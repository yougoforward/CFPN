from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['vgg_full_dilated', 'get_vgg_full_dilated']

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class vgg_full_dilated(nn.Module):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], **kwargs):
        super(vgg_full_dilated, self).__init__()
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        self._up_kwargs = up_kwargs
        self.base = vgg_full_dilated_base(norm_layer)
        self.head = vgg_full_dilatedHead(512, nclass, norm_layer, up_kwargs=self._up_kwargs)

    def forward(self, x):
        imsize = x.size()[2:]
        x = self.base(x)
        x = self.head(x)
        x = F.interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
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



class vgg_full_dilatedHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs=None):
        super(vgg_full_dilatedHead, self).__init__()
        self._up_kwargs = up_kwargs

        inter_channels = 512
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        out = self.conv5(x)
        return self.conv6(out)

class vgg_full_dilated_base(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(vgg_full_dilated_base, self).__init__()
        self.layer1 = vgg_full_dilated_layer(3,64,1,1,norm_layer)
        self.layer2 = vgg_full_dilated_layer(64,64,1,1,norm_layer)
        # self.pool = nn.MaxPool2d(2)
        
        self.layer3 = vgg_full_dilated_layer(64,128,2,1,norm_layer)
        self.layer4 = vgg_full_dilated_layer(128,128,2,1,norm_layer)

        self.layer5 = vgg_full_dilated_layer(128,256,4,1,norm_layer)
        self.layer6 = vgg_full_dilated_layer(256,256,4,1,norm_layer)
        self.layer7 = vgg_full_dilated_layer(256,256,4,1,norm_layer)
        
        self.layer8 = vgg_full_dilated_layer(256,512,8,1,norm_layer)
        self.layer9 = vgg_full_dilated_layer(512,512,8,1,norm_layer)
        self.layer10 = vgg_full_dilated_layer(512,512,8,1,norm_layer)
        
        self.layer11 = vgg_full_dilated_layer(512,512,16,1,norm_layer)
        self.layer12 = vgg_full_dilated_layer(512,512,16,1,norm_layer)
        self.layer13 = vgg_full_dilated_layer(512,512,16,1,norm_layer)


    def forward(self, x):
        x1=self.layer1(x)
        x2=self.layer2(x1)
        # x_pool1=self.pool(x2)
        
        x3=self.layer3(x2)
        x4=self.layer4(x3)
        # x_pool2=self.pool(x4)
        
        x5=self.layer5(x4)
        x6=self.layer6(x5)
        x7=self.layer7(x6)
        # x_pool3=self.pool(x7)
        
        x8=self.layer8(x7)
        x9=self.layer9(x8)
        x10=self.layer10(x9)
        # x_pool4=self.pool(x10)
        
        x11=self.layer11(x10)
        x12=self.layer12(x11)
        x13=self.layer13(x12)
        return x13



class vgg_full_dilated_layer(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, tl_size=1, norm_layer=nn.BatchNorm2d):
        super(vgg_full_dilated_layer, self).__init__()
        self.tl_size = tl_size
        self.inplanes = in_planes
        self.outplanes = out_planes
        self.conv = nn.Sequential(nn.Conv2d(in_planes*tl_size*tl_size, out_planes*tl_size*tl_size, 3, padding=dilation, dilation=dilation, bias=False),
                                   norm_layer(out_planes*tl_size*tl_size), nn.ReLU())

    def forward(self, x):
        out = self.conv(x)
        return out


def get_vgg_full_dilated(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = vgg_full_dilated(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
