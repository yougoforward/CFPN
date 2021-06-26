###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division

import torch
import torch.nn as nn

from torch.nn.functional import interpolate

from .base import BaseNet
from .fcn import FCNHead


__all__ = ['dfcn', 'get_dfcn']

class dfcn(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(dfcn, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = dfcnHead(2048, nclass, norm_layer)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)

        
class dfcnHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(dfcnHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


def get_dfcn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = dfcn(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

