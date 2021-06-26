from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['fpn', 'get_fpn']


class fpn(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(fpn, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = fpnHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)
        x = self.head(c1,c2,c3,c4)
        x = F.interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)



class fpnHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(fpnHead, self).__init__()
        self.se_loss = se_loss
        self._up_kwargs = up_kwargs

        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))

        self.localUp3=localUp(512, inter_channels, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, inter_channels, norm_layer, up_kwargs)

    def forward(self, c1,c2,c3,c4):
        _,_, h,w = c2.size()
        out = self.conv5(c4)
               
        out3 = self.localUp4(c3, out)  
        out = self.localUp3(c2, out3)
        
        return self.conv6(out)

class localUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(localUp, self).__init__()
        self.connect = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU())

        self._up_kwargs = up_kwargs

    def forward(self, c1,c2):
        n,c,h,w =c1.size()
        c1p = self.connect(c1) # n, 64, h, w
        c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
        out = c1p + c2
        return out


def get_fpn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = fpn(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model


