from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['refinenet', 'get_refinenet']


class refinenet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(refinenet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = refinenetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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



class refinenetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(refinenetHead, self).__init__()
        self.se_loss = se_loss
        self._up_kwargs = up_kwargs

        self.conv6 = nn.Conv2d(256, out_channels, kernel_size=3, stride=1,
                                  padding=1)

        self.rf2=localUp3(256, 256, norm_layer, up_kwargs)
        self.rf3=localUp(512, 256, norm_layer, up_kwargs)
        self.rf4=localUp4(1024, 256, norm_layer, up_kwargs)
        self.rf5=localUp2(2048, 512, norm_layer, up_kwargs)

    def forward(self, c1,c2,c3,c4):
        _,_, h,w = c2.size()
        out = self.rf5(c4,c4)
               
        out = self.rf4(c3, out)  
        out = self.rf3(c2, out)
        out = self.rf2(c1, out)
        
        return self.conv6(out)

class RCU(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(RCU, self).__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        self._up_kwargs = up_kwargs

    def forward(self, x):
        residual = x
        out = residual+self.conv2(self.relu(self.conv1(self.relu(x))))
        return out

class CRP(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(CRP, self).__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self._up_kwargs = up_kwargs

    def forward(self, x):
        x = self.relu(x)
        top1 = self.conv1(self.maxpool(x))
        top2 = self.conv2(self.maxpool(top1))
        top3 = self.conv2(self.maxpool(top2))
        top4 = self.conv2(self.maxpool(top3))
        out = x + top1+ top2 +top3 +top4       
        return out

class localUp3(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(localUp3, self).__init__()
        self.connect = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.rcu1 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu2 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.project = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        self.rcu21 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu22 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.project2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self._up_kwargs = up_kwargs
        
        self.crp = CRP(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu31 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu32 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu33 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        # self.project3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        
        
        

    def forward(self, c1,c2):
        n,c,h,w =c1.size()
        c1 = self.project(self.rcu2(self.rcu1(self.connect(c1))))
        c2 = self.project2(self.rcu22(self.rcu21(c2)))
        
        c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
        sum = c1 + c2
        out = self.crp(sum)
        out = self.rcu33(self.rcu32(self.rcu31(sum)))
        # out = self.rcu31(out)
        # out = self.project3(self.rcu33(self.rcu32(self.rcu31(sum))))
        return out
    
class localUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(localUp, self).__init__()
        self.connect = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.rcu1 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu2 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.project = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        self.rcu21 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu22 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.project2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self._up_kwargs = up_kwargs
        
        self.crp = CRP(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu31 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        # self.rcu32 = RCU(out_channels, out_channels)
        # self.rcu33 = RCU(out_channels, out_channels)
        # self.project3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        
        
        

    def forward(self, c1,c2):
        n,c,h,w =c1.size()
        c1 = self.project(self.rcu2(self.rcu1(self.connect(c1))))
        c2 = self.project2(self.rcu22(self.rcu21(c2)))
        
        c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
        sum = c1 + c2
        out = self.crp(sum)
        out = self.rcu31(out)
        # out = self.project3(self.rcu33(self.rcu32(self.rcu31(sum))))
        return out

class localUp4(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(localUp4, self).__init__()
        self.connect = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.rcu1 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu2 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.project = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        self.rcu21 = RCU(2*out_channels, 2*out_channels, norm_layer, up_kwargs)
        self.rcu22 = RCU(2*out_channels, 2*out_channels, norm_layer, up_kwargs)
        self.project2 = nn.Conv2d(2*out_channels, out_channels, 3, padding=1, bias=False)
        self._up_kwargs = up_kwargs
        
        self.crp = CRP(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu31 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        # self.rcu32 = RCU(out_channels, out_channels)
        # self.rcu33 = RCU(out_channels, out_channels)
        # self.project3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        
        
        

    def forward(self, c1,c2):
        n,c,h,w =c1.size()
        c1 = self.project(self.rcu2(self.rcu1(self.connect(c1))))
        c2 = self.project2(self.rcu22(self.rcu21(c2)))
        
        c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
        sum = c1 + c2
        out = self.crp(sum)
        out = self.rcu31(out)
        # out = self.project3(self.rcu33(self.rcu32(self.rcu31(sum))))
        return out
    
class localUp2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(localUp2, self).__init__()
        self.connect = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.rcu1 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu2 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        self.project = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        # self.rcu21 = RCU(out_channels, out_channels)
        # self.rcu22 = RCU(out_channels, out_channels)
        # self.project2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self._up_kwargs = up_kwargs
        
        self.crp = CRP(out_channels, out_channels, norm_layer, up_kwargs)
        self.rcu31 = RCU(out_channels, out_channels, norm_layer, up_kwargs)
        # self.rcu32 = RCU(out_channels, out_channels)
        # self.rcu33 = RCU(out_channels, out_channels)
        # self.project3 = nn.Conv2d(out_channels, out_channels//2, 3, padding=1, bias=False)
        
        
        
        

    def forward(self, c1,c2):
        n,c,h,w =c1.size()
        c1 = self.project(self.rcu2(self.rcu1(self.connect(c1))))
        # c2 = self.project2(self.rcu22(self.rcu21(self.connect2(c2))))
        
        # c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
        # sum = c1 + c2
        sum = c1
        out = self.rcu31(sum)
        # out = self.project3(self.rcu33(self.rcu32(self.rcu31(sum))))
        return out

def get_refinenet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = refinenet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model


