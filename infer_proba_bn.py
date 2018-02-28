import torch
import argparse
import utils
from models.stochbn import _MyBatchNorm
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--data_kn', default='cifar5')
parser.add_argument('--data_ukn', default='cifar5-rest')
parser.add_argument('--log_file', default='eval_data')
parser.add_argument('--test_bs', default=500, type=int)
parser.add_argument('--n_classes', default=5, type=int)
parser.add_argument('--data_root', default='/home/andrew/StochBN/data')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--n_tries', default=50, type=int)
# parser.add_argument('--model_types', nargs='+', default=['eval', 'ensemble'])
args = parser.parse_args()

torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

ckpt = torch.load(args.model)
log_file = utils.uniquify(os.path.join(os.path.dirname(args.model), args.log_file), sep='-')

_, dataloader_known = utils.get_dataloader(data=args.data_kn, test_bs=args.test_bs, data_root=args.data_root)
_, dataloader_unknown = utils.get_dataloader(data=args.data_ukn, test_bs=args.test_bs,
                                                         data_root=args.data_root, drop_last_test=True)

eval_data = {}

net = utils.load_model(args.model)
net.eval()
utils.set_strategy(net, 'sample')
have_do =  utils.set_do_to_train(net)

res = utils.predict_proba(dataloader_unknown, net, n_classes=args.n_classes, return_logits=True, ensembles=args.n_tries)

eval_data['unknown'] = {
    'ensemble/proba': res[0],
    'ensemble/logits': res[2],
    'ensemble/labels': res[1]
}

res = utils.predict_proba(dataloader_known, net, n_classes=args.n_classes, return_logits=True, ensembles=args.n_tries)
eval_data['known'] = {
    'ensemble/proba': res[0],
    'ensemble/logits': res[2],
    'ensemble/labels': res[1]
}

net.eval()
utils.set_strategy(net, 'running')
have_do = utils.set_do_to_train(net)

res = utils.predict_proba(dataloader_unknown, net, n_classes=args.n_classes, return_logits=True, ensembles=args.n_tries if have_do else 1)
eval_data['unknown'].update({
    'eval/proba': res[0],
    'eval/logits': res[2],
    'eval/labels': res[1]
})

res = utils.predict_proba(dataloader_known, net, n_classes=args.n_classes, return_logits=True, ensembles=args.n_tries if have_do else 1)
eval_data['known'].update({
    'eval/proba': res[0],
    'eval/logits': res[2],
    'eval/labels': res[1]
})

torch.save(eval_data, log_file)
