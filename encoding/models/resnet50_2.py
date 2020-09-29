from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

from .fcn import FCNHead
from .base import BaseNet
from ..dilated import resnet2 as resnet

__all__ = ['resnet50_2', 'get_resnet50_2']

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class resnet50_2(nn.Module):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], **kwargs):
        super(resnet50_2, self).__init__()
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        self._up_kwargs = up_kwargs
        self.base = resnet.resnet50(pretrained=False, norm_layer=norm_layer)
        self.head = FCNHead(2048, nclass, norm_layer)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)
            
    def forward(self, x):
        imsize = x.size()[2:]
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        c1 = self.base.layer1(x)
        c2 = self.base.layer2(c1)
        c3 = self.base.layer3(c2)
        c4 = self.base.layer4(c3)
        
        x = self.head(c4)
        x = interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)
    
    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union


def get_resnet50_2(dataset='pascal_voc', backbone='resnet50_2', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = resnet50_2(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
