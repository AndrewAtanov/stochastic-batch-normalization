import os
import torch
import utils
import argparse
import numpy as np
from itertools import product
from torch.nn import CrossEntropyLoss


parser = argparse.ArgumentParser()
parser.add_argument('--models_dir', help='Direcotry with models, should contain subdirectorie with models: 0/model, 1/model etc.')
parser.add_argument('--n_models', type=int, help='Number of models in Deep Ensembles')
parser.add_argument('--n_tries', default=50, type=int, help='Number of samples from MC estimation')
parser.add_argument('--log_file', default='de_eval_data', help='File to log predictions')
parser.add_argument('--data_kn', default='cifar5', help='Known data')
parser.add_argument('--data_ukn', default='cifar5-rest', help='Unknown data')
parser.add_argument('--test_bs', default=500, type=int)
parser.add_argument('--n_classes', default=5, type=int)
parser.add_argument('--data_root', default='/home/andrew/StochBN/data')
args = parser.parse_args()

_, testloader_kn = utils.get_dataloader(data=args.data_kn, test_bs=args.test_bs, data_root=args.data_root,
                                        shuffle=False)
_, testloader_ukn = utils.get_dataloader(data=args.data_ukn, test_bs=args.test_bs, data_root=args.data_root,
                                         shuffle=False)

eval_data = {
    'known': {
        'eval/logits': [],
        'ensemble/logits': [],
        'labels': [],
    },

    'unknown': {
        'eval/logits': [],
        'ensemble/logits': [],
    }
}

criterion = CrossEntropyLoss().cuda()

for i in range(args.n_models):
    net = utils.load_model(os.path.join(args.models_dir, str(i), 'model'))
    net.eval()
    for s, mode, n in zip(['running', 'sample'], ['eval', 'ensemble'], [1, args.n_tries]):
        utils.set_strategy(net, strategy=s)

        _, labels, logits = utils.predict_proba(testloader_kn, net, ensembles=n,
                                                n_classes=args.n_classes, return_logits=True)
        eval_data['known']['{}/logits'.format(mode)].append(logits)
        eval_data['known']['labels'] = labels

        _, _, logits = utils.predict_proba(testloader_ukn, net, ensembles=n,
                                           n_classes=args.n_classes, return_logits=True)
        eval_data['unknown']['{}/logits'.format(mode)].append(logits)

for d, t in product(['known', 'unknown'], ['eval', 'ensemble']):
    eval_data[d]['{}/logits'.format(t)] = np.squeeze(np.stack(eval_data[d]['{}/logits'.format(t)]))

torch.save(eval_data, os.path.join(args.models_dir, args.log_file))
