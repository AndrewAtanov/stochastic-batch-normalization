'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from stochbn import MyBatchNorm2d, MyBatchNorm1d
from collections import OrderedDict
import numpy as np


class LeNet(nn.Module):
    def __init__(self, dropout=None):
        super(LeNet, self).__init__()
        if dropout is None:
            dropout = [0.]
        self.conv_features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 20, 5, 1)),
            ('bn1', MyBatchNorm2d(20)),
            # ('bn1', nn.BatchNorm2d(20)),
            ('relu1', nn.ReLU()),
            ('do1', nn.Dropout(p=dropout[0])),
            ('pool1', nn.MaxPool2d(2)),

            ('conv2', nn.Conv2d(20, 50, 5, 1)),
            ('bn2', MyBatchNorm2d(50)),
            # ('bn2', nn.BatchNorm2d(50)),
            ('relu2', nn.ReLU()),
            ('do2', nn.Dropout(p=dropout[0])),
            ('pool2', nn.MaxPool2d(2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16 * 50, 500)),
            ('bn3', nn.BatchNorm1d(500)),
            ('relu3', nn.ReLU()),
            ('do3', nn.Dropout(p=dropout[0])),
            ('fc2', nn.Linear(500, 10)),
        ]))

    def forward(self, x):
        out = self.conv_features(x)
        out = out.view(out.size(0), 16 * 50)
        out = self.classifier(out)
        return out


class FC(nn.Module):
    def __init__(self, n_hidden=100, n_classes=10):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(28**2, n_hidden)
        self.bn1 = MyBatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.bn2 = MyBatchNorm1d(n_hidden)
        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x, return_activations=False):
        activations = []
        out = x.view(x.size(0), 28**2)
        out = self.fc1(out)
        out = self.bn1(out)
        if return_activations:
            activations.append(out.data.cpu().numpy())
        out = F.relu()
        out = self.fc2(out)
        out = F.relu(self.bn2(out))
        if return_activations:
            activations.append(out.data.cpu().numpy())
        out = self.classifier(out)
        return out
