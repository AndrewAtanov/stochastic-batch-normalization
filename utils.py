import numpy as np
import tempfile
import itertools as IT
import os
from torch.nn.parallel import DataParallel
import torch
from models.simple import LeNet
from models.vgg import VGG
from models.resnet import ResNet18
from models.stochbn import _MyBatchNorm
import importlib
import sys
import pickle
import torchvision
from torchvision import transforms
import PIL
from torch.nn import Dropout
from torch.autograd import Variable


def uniquify(path, sep=''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s=sep, n=next(count))
    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir=dirname, prefix=filename, suffix=ext)
        tempfile._name_sequence = orig
    return filename


def make_description(args):
    return '{}'.format(vars(args))


class Ensemble:
    """
    Ensemble for classification. Take logits and average probabilities using softmax.
    """
    def __init__(self, save_logits=False):
        self.__n_estimators = 0
        self.cum_proba = 0
        self.logits = None
        if save_logits:
            self.logits = []

    def add_estimator(self, logits):
        """
        Add estimator to current ensemble. First call define number of objects (N) and number of classes (K).
        :param logits: ndarray of logits with shape (N, K)
        """
        if self.logits is not None:
            self.logits.append(np.copy(logits))
        l = np.exp(logits - logits.max(1)[:, np.newaxis])

        assert not np.isnan(l).any(), 'NaNs while computing softmax'
        self.cum_proba += l / l.sum(1)[:, np.newaxis]
        assert not np.isnan(self.cum_proba).any(), 'NaNs while computing softmax'

        self.__n_estimators += 1

    def get_proba(self):
        """
        :return: ndarray with probabilities of shape (N, K)
        """
        return self.cum_proba / self.__n_estimators

    def get_logits(self):
        return np.array(self.logits)


class AccCounter:
    """
    Class for count accuracy during pass through data with mini-batches.
    """
    def __init__(self):
        self.__n_objects = 0
        self.__sum = 0

    def add(self, outputs, targets):
        """
        Compute and save stats needed for overall accuracy.
        :param outputs: ndarray of predicted values (logits or probabilities)
        :param targets: ndarray of labels with the same length as first dimension of _outputs_
        """
        self.__sum += np.sum(outputs.argmax(axis=1) == targets)
        self.__n_objects += outputs.shape[0]

    def acc(self):
        """
        Compute current accuracy.
        :return: float accuracy.
        """
        return self.__sum * 1. / self.__n_objects

    def flush(self):
        """
        Flush stats.
        :return:
        """
        self.__n_objects = 0
        self.__sum = 0


def softmax(logits, temp=1.):
    assert not np.isnan(logits).any(), 'NaNs in logits for softmax'
    if len(logits.shape) == 2:
        l = np.exp((logits - logits.max(1)[:, np.newaxis]) / temp)
        try:
            assert not np.isnan(l).any(), 'NaNs while computing softmax'
            return l / l.sum(1)[:, np.newaxis]
        except Exception as e:
            raise e
    elif len(logits.shape) == 4:
        return de_hbn_ensemble(logits, temp=temp)
    else:
        l = np.exp((logits - logits.max(2)[:, :, np.newaxis]) / temp)
        assert not np.isnan(l).any(), 'NaNs while computing softmax with temp={}'.format(temp)
        l /= l.sum(2)[:, :, np.newaxis]
        return np.mean(l, axis=0)


def entropy_plot_xy(p):
    e = entropy(p)
    n = len(e)
    return sorted(e), np.arange(1, n + 1) / 1. / n


def entropy_plot_with_proba(p):
    e = entropy(p)
    n = len(e)
    return sorted(e), np.arange(1, n + 1) / 1. / n


def entropy_plot_with_logits(logits, adjust_t=False, k=0.2,
                             labels=None):
    if len(logits.shape) == 2:
        logits = logits[np.newaxis]

    temp = 1.
    if adjust_t:
        k = int(logits.shape[1] * k)
        val_logits = logits[:, :k]
        logits = logits[:, k:]
        temp = adjust_temp(val_logits, labels[:k])
    return entropy_plot_with_proba(softmax(logits, temp=temp))


def set_strategy(net, strategy):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            m.strategy = strategy


def set_do_to_train(net):
    have_do = False
    for m in net.modules():
        if isinstance(m, Dropout):
            m.train()
            have_do = True
    return have_do


def get_model(model='ResNet18', **kwargs):
    if model == 'ResNet18':
        return ResNet18(n_classes=kwargs.get('n_classes', 10))
    elif 'VGG' in model:
        return VGG(vgg_name=model, k=kwargs['k'], dropout=kwargs.get('dropout', None),
                   n_classes=kwargs.get('n_classes', 10), )
    elif 'LeNet' == model:
        return LeNet(dropout=kwargs.get('dropout', None))
    else:
        raise NotImplementedError('unknown {} model'.format(model))


def get_dataloader(data='cifar', train_bs=128, test_bs=200, augmentation=True,
                   noiid=False, shuffle=True, data_root='./data',
                   drop_last_train=False, drop_last_test=False,
                   random_labeling=False):
    transform_train = transforms.Compose([
        MyPad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if data == 'cifar':
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True,
                                                transform=transform_train if augmentation else transform_test)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    elif data == 'mnist':
        trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)
    elif data == 'not-mnist':
        trainset = torchvision.datasets.MNIST(root=os.path.join(data_root, 'not-mnist'), train=False,
                                              download=True, transform=transform_test)
        testset = torchvision.datasets.MNIST(root=os.path.join(data_root, 'not-mnist'), train=False,
                                             download=True, transform=transform_test)
    elif data == 'cifar5':
        CIFAR5_CLASSES = [0, 1, 2, 3, 4]
        trainset = CIFAR(root=data_root, train=True, download=True,
                         transform=transform_train if augmentation else transform_test, classes=CIFAR5_CLASSES,
                         random_labeling=random_labeling)
        testset = CIFAR(root=data_root, train=False, download=True, transform=transform_test, classes=CIFAR5_CLASSES,
                        random_labeling=random_labeling)
    elif data == 'cifar5-rest':
        CIFAR5_CLASSES = [5, 6, 7, 8, 9]
        trainset = CIFAR(root=data_root, train=True, download=True,
                         transform=transform_train if augmentation else transform_test, classes=CIFAR5_CLASSES)
        testset = CIFAR(root=data_root, train=False, download=True, transform=transform_test, classes=CIFAR5_CLASSES)
    else:
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=shuffle,
                                              num_workers=2, drop_last=drop_last_train)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False,
                                             num_workers=2, drop_last=drop_last_test)

    return trainloader, testloader


def load_model(filename, print_info=False):
    use_cuda = torch.cuda.is_available()
    chekpoint = torch.load(filename)
    net = get_model(**chekpoint.get('script_args', {}))
    net.load_state_dict(chekpoint['state_dict'])
    if use_cuda:
        net = DataParallel(net, device_ids=range(torch.cuda.device_count()))
    return net


def pad(img, size, mode):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    return np.pad(img, [(size, size), (size, size), (0, 0)], mode)


class MyPad(object):
    def __init__(self, size, mode='reflect'):
        self.mode = mode
        self.size = size
        self.topil = transforms.ToPILImage()

    def __call__(self, img):
        return self.topil(pad(img, self.size, self.mode))


def to_np(x):
    return x.data.cpu().numpy()


def entropy(p):
    eps = 1e-8
    assert np.all(p >= 0)
    return np.apply_along_axis(lambda x: -np.sum(x[x > eps] * np.log(x[x > eps])), 1, p)


def ensemble(net, data, bs, n_infer=50, return_logits=False):
    """ Ensemble for net training with Vanilla BN """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    ens = Ensemble(save_logits=return_logits)
    acc_data = np.array(list(map(lambda x: transform_test(x).numpy(), data)))
    logits = []
    for _ in range(n_infer):
        logits = np.zeros([acc_data.shape[0], 5])
        perm = np.random.permutation(np.arange(acc_data.shape[0]))

        for i in range(0, len(perm), bs):
            idxs = perm[i: i + bs]
            inputs = Variable(torch.Tensor(acc_data[idxs]).cuda(async=True))
            outputs = net(inputs)
            assert np.allclose(logits[idxs], 0.)
            logits[idxs] = outputs.cpu().data.numpy()

        ens.add_estimator(logits)
    return ens.get_proba(), ens.get_logits()


def predict_proba(dataloader, net, ensembles=1, n_classes=10, return_logits=False):
    proba = np.zeros((len(dataloader.dataset), n_classes))
    labels = []
    logits = []
    p = 0
    for img, label in dataloader:
        ens = Ensemble(save_logits=return_logits)
        img = Variable(img).cuda()
        for _ in range(ensembles):
            pred = net(img).data.cpu().numpy()
            ens.add_estimator(pred)
        proba[p: p + pred.shape[0]] = ens.get_proba()
        p += pred.shape[0]
        labels += label.tolist()
        if return_logits:
            logits.append(ens.get_logits())

    if return_logits:
        logits = np.stack(logits)
        logits = logits.transpose(0, 2, 1, 3)
        logits = np.concatenate(logits, axis=0)
        logits = logits.transpose(1, 0, 2)
        return proba, np.array(labels), logits
    return proba, np.array(labels)


class CIFAR(torchvision.datasets.CIFAR10):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset with several classes.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, classes=None, random_labeling=False):

        if classes is None:
            classes = np.arange(10).tolist()

        self.classes = classes[:]

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.random_labeling = random_labeling

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            mask = np.isin(self.train_labels, classes)
            self.train_labels = [classes.index(l) for l, cond in zip(self.train_labels, mask) if cond]
            if self.random_labeling:
                self.train_labels = np.random.permutation(self.train_labels)

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))[mask]
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

            mask = np.isin(self.test_labels, classes)
            self.test_labels = [classes.index(l) for l, cond in zip(self.test_labels, mask) if cond]

            self.test_data = self.test_data.reshape((10000, 3, 32, 32))[mask]
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
