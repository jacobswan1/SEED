import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

__all__ = ['simclr_resnet50']


class SimCLRResNet(nn.Module):
    def __init__(self, base_model, num_classes):

        super(SimCLRResNet, self).__init__()
        resnet = self._get_basemodel(base_model, num_classes)
        num_ftrs = resnet.fc.in_features

        resnet.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
                                  nn.BatchNorm1d(num_ftrs),
                                  nn.ReLU(),
                                  resnet.fc)
        self.module = resnet
        self.iter = 0

    def _get_basemodel(self, model_name, num_classes):
        model = models.__dict__[model_name](num_classes=num_classes)
        return model

    def forward_one(self, x):
        x = self.module(x)
        x = F.normalize(x, dim=1)
        return x

    def forward(self, x):
        z = self.forward_one(x)
        return z


def simclr_resnet50(num_classes, **kwargs):
    return SimCLRResNet(base_model='resnet50', num_classes=num_classes)

