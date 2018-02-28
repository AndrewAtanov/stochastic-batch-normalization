'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from stochbn import MyBatchNorm2d
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, k=1, dropout=None, n_classes=10, learn_bn_stats=False):
        super(VGG, self).__init__()
        self.use_dropout = not (dropout is None)
        self.dropout_rate = dropout
        self.learn_bn_stats = learn_bn_stats

        self.features = self._make_layers(cfg[vgg_name], k)
        if self.use_dropout:
            self.dropout = nn.Dropout(self.dropout_rate[0])
        self.classifier = nn.Linear(int(512 * k), n_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, config, k):
        layers = []
        in_channels = 3
        for i, x in enumerate(config):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = int(x * k)
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           MyBatchNorm2d(x, learn_stats=self.learn_bn_stats),
                           nn.ReLU(inplace=True)]
                if self.use_dropout and config[i + 1] != 'M':
                    layers += [nn.Dropout(self.dropout_rate[1])]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
