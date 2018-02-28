'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.optim as optim
from torch import nn
import utils
import numpy as np
import os
import argparse
from torch.autograd import Variable
import shutil
from time import time


parser = argparse.ArgumentParser(description='Trin DNN with SBN modeules')
parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--bs', default=16, type=int, help='Batch size')
parser.add_argument('--test_bs', default=200, type=int, help='Batch size for test dataloader')
parser.add_argument('--decrease_from', default=1, type=int, help='Epoch to decrease learning_rate linear to 0 from')
parser.add_argument('--log_dir', help='Directory for logging')
parser.add_argument('--augmentation', dest='augmentation', action='store_true',
                    help='Eigther use or not data augmentation for cifar')
parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
parser.set_defaults(augmentation=True)
parser.add_argument('--model', '-m', default='ResNet18', help='Model to train')
parser.add_argument('--k', '-k', default=1, type=float, help='Model size for VGG')
parser.add_argument('--dropout', type=float, nargs='+', default=None,
                    help='Dropout rate, include dropout layer if > 0.')
parser.add_argument('--data', default='cifar', help='Dataset, one of: cifar, cifar5, mnist, not-mnist')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--epoch_tune_bn', default=5, type=int,
                    help='Number of epochs to tune SBN approximation parameters')
args = parser.parse_args()
args.script = os.path.basename(__file__)


def save_checkpoint(state, is_best, epoch=None):
    if epoch:
        torch.save(state, '{}/model-{}'.format(args.log_dir, epoch))
        if is_best:
            shutil.copyfile('{}/model-{}'.format(args.log_dir, epoch), '{}/best_model'.format(args.log_dir))
    else:
        torch.save(state, '{}/model'.format(args.log_dir))
        if is_best:
            shutil.copyfile('{}/model'.format(args.log_dir), '{}/best_model'.format(args.log_dir))


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_linear(epoch):
    return max(0, (args.lr * np.minimum((args.decrease_from - epoch) * 1. / (args.epochs - args.decrease_from) + 1, 1.)))


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

NCLASSES = 10
if args.data == 'cifar5':
    NCLASSES = 5

print('==> Preparing data..')

trainloader, testloader = utils.get_dataloader(data=args.data, train_bs=args.bs, test_bs=args.test_bs,
                                               augmentation=args.augmentation)

print('==> Building model..')
net = utils.get_model(n_classes=NCLASSES, **vars(args))


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))


criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

with open('{}/log'.format(args.log_dir), 'w') as f:
    f.write('#{}\n'.format(utils.make_description(args)))
    f.write('epoch,loss,train_acc,test_acc\n')

best_test_acc = 0
counter = utils.AccCounter()

model_args = vars(args)
model_args['n_classes'] = NCLASSES

print(net)

for epoch in range(args.epochs):
    counter.flush()

    t0 = time()
    lr = lr_linear(epoch)
    adjust_learning_rate(optimizer, lr)

    net.train()
    utils.set_strategy(net, 'batch')

    training_loss = 0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
        optimizer.zero_grad()

        outputs = net(inputs)
        counter.add(outputs.data.cpu().numpy(), labels.data.cpu().numpy())
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        training_loss += loss.cpu().data.numpy()[0] * float(inputs.size(0))

    train_acc = counter.acc()
    counter.flush()
    test_loss = 0

    net.eval()
    utils.set_strategy(net, 'running')
    for _, (inputs, labels) in enumerate(testloader):
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += utils.to_np(loss) * float(inputs.size(0))
        counter.add(utils.to_np(outputs), utils.to_np(labels))

    print(' -- Epoch %d | time: %.4f | loss: %.4f | training acc: %.4f validation accuracy: %.4f | lr %.6f --' %
          (epoch, time() - t0, training_loss, train_acc, counter.acc(), lr))

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.module.state_dict() if use_cuda else net.state_dict(),
        'test_accuracy': counter.acc(),
        'optimizer': optimizer.state_dict(),
        'name': args.model,
        'model_args': model_args,
        'script_args': vars(args)
    }, best_test_acc < counter.acc())

    with open('{}/log'.format(args.log_dir), 'a') as f:
        f.write('{},{},{},{}\n'.format(epoch, training_loss, train_acc, counter.acc()))

    if best_test_acc < counter.acc():
        best_test_acc = counter.acc()

print('==> Finish Training')

print('==> Tune SBN parameters')

net.train()
utils.set_strategy(net, 'batch')
for _ in range(args.epoch_tune_bn):
    for inputs, _ in trainloader:
        inputs = Variable(inputs.cuda(async=True))
        net(inputs)

print('==> Done.')


