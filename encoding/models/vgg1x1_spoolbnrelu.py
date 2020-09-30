from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['vgg1x1_spoolbnrelu', 'get_vgg1x1_spoolbnrelu']

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class vgg1x1_spoolbnrelu(nn.Module):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], **kwargs):
        super(vgg1x1_spoolbnrelu, self).__init__()
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        self._up_kwargs = up_kwargs
        self.base = vgg1x1_spoolbnrelu_base(norm_layer)
        self.head = vgg1x1_spoolbnreluHead(512, nclass, norm_layer, up_kwargs=self._up_kwargs)

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



class vgg1x1_spoolbnreluHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs=None):
        super(vgg1x1_spoolbnreluHead, self).__init__()
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

class vgg1x1_spoolbnrelu_base(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(vgg1x1_spoolbnrelu_base, self).__init__()
        self.layer1 = vgg1x1_spoolbnrelu_layer(3,64,1,1,norm_layer)
        self.layer2 = vgg1x1_spoolbnrelu_layer(64,64,1,1,norm_layer)
        # self.pool = nn.MaxPool2d(2)
        
        # self.layer3 = vgg1x1_spoolbnrelu_layer2(64,128,1,1,norm_layer)
        # self.layer4 = vgg1x1_spoolbnrelu_layer2(128,128,1,1,norm_layer)

        # self.layer5 = vgg1x1_spoolbnrelu_layer2(128,256,1,1,norm_layer)
        # self.layer6 = vgg1x1_spoolbnrelu_layer2(256,256,1,1,norm_layer)
        # self.layer7 = vgg1x1_spoolbnrelu_layer2(256,256,1,1,norm_layer)
        
        # self.layer8 = vgg1x1_spoolbnrelu_layer2(256,512,1,1,norm_layer)
        # self.layer9 = vgg1x1_spoolbnrelu_layer2(512,512,1,1,norm_layer)
        # self.layer10 = vgg1x1_spoolbnrelu_layer2(512,512,1,1,norm_layer)
        
        # self.layer11 = vgg1x1_spoolbnrelu_layer3(512,512,1,1,256,256,norm_layer)
        # self.layer12 = vgg1x1_spoolbnrelu_layer3(512,512,1,1,256,256,norm_layer)
        # self.layer13 = vgg1x1_spoolbnrelu_layer3(512,512,1,1,256,256,norm_layer)
        self.layer11 = vgg1x1_spoolbnrelu_layer4(64,512,1,1,256,256,norm_layer)
        self.layer12 = vgg1x1_spoolbnrelu_layer4(512,512,1,1,256,256,norm_layer)
        self.layer13 = vgg1x1_spoolbnrelu_layer4(512,512,1,1,256,256,norm_layer)

    def forward(self, x):
        x1=self.layer1(x)
        x2=self.layer2(x1)
        # x_pool1=self.pool(x2)
        
        # x3=self.layer3(x2)
        # x4=self.layer4(x3)
        # # x_pool2=self.pool(x4)
        
        # x5=self.layer5(x4)
        # x6=self.layer6(x5)
        # x7=self.layer7(x6)
        # # x_pool3=self.pool(x7)
        
        # x8=self.layer8(x7)
        # x9=self.layer9(x8)
        # x10=self.layer10(x9)
        # # x_pool4=self.pool(x10)
        
        # x11=self.layer11(x10)
        # x12=self.layer12(x11)
        # x13=self.layer13(x12)
        
        x11=self.layer11(x2)
        x12=self.layer12(x11)
        x13=self.layer13(x12)
        return x13



class vgg1x1_spoolbnrelu_layer(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, tl_size=1, norm_layer=nn.BatchNorm2d):
        super(vgg1x1_spoolbnrelu_layer, self).__init__()
        self.tl_size = tl_size
        self.inplanes = in_planes
        self.outplanes = out_planes
        self.conv = nn.Sequential(nn.Conv2d(in_planes*tl_size*tl_size, out_planes*tl_size*tl_size, 3, padding=1, dilation=1, bias=False),
                                   norm_layer(out_planes*tl_size*tl_size), nn.ReLU())

    def forward(self, x):
        out = self.conv(x)
        return out
    
class vgg1x1_spoolbnrelu_layer2(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, tl_size=1, norm_layer=nn.BatchNorm2d):
        super(vgg1x1_spoolbnrelu_layer2, self).__init__()
        self.tl_size = tl_size
        self.inplanes = in_planes
        self.outplanes = out_planes
        self.conv = nn.Sequential(nn.Conv2d(in_planes*tl_size*tl_size, out_planes*tl_size*tl_size, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(out_planes*tl_size*tl_size), nn.ReLU())

    def forward(self, x):
        out = self.conv(x)
        return out

class vgg1x1_spoolbnrelu_layer3(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, tl_size=1, height=256, weight=256, norm_layer=nn.BatchNorm2d):
        super(vgg1x1_spoolbnrelu_layer3, self).__init__()
        self.tl_size = tl_size
        self.inplanes = in_planes
        self.outplanes = out_planes
        self.spool = SPool(height, weight, norm_layer)
        self.conv = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(out_planes))
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(in_planes//4), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(out_planes))
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.spool(x1)
        x1 = self.conv2(x1)
        out = self.conv(x)
        out = self.relu(x1+out)
        return out

class vgg1x1_spoolbnrelu_layer4(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, tl_size=1, height=256, weight=256, norm_layer=nn.BatchNorm2d):
        super(vgg1x1_spoolbnrelu_layer4, self).__init__()
        self.tl_size = tl_size
        self.inplanes = in_planes
        self.outplanes = out_planes
        self.spool = SPool(height, weight, norm_layer)
        self.conv = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(out_planes),nn.ReLU())
        

    def forward(self, x):
        x1 = self.spool(x)
        out = self.conv(x1)
        return out
    
class SPool(nn.Module):
    def __init__(self, height, width, norm_layer):
        super(SPool, self).__init__()
        self.conv_h = nn.Sequential(nn.Conv2d(height, height, 1, padding=0, dilation=1, bias=False), norm_layer(height),nn.ReLU())
        self.conv_w = nn.Sequential(nn.Conv2d(width, width, 1, padding=0, dilation=1, bias=False), norm_layer(width),nn.ReLU())

    def forward(self, x):
        n,c,h,w = x.size()
        x_h = x.permute(0,2,1,3).contiguous()#n,h,c,w
        x_w = x.permute(0,3,1,2).contiguous()#n,w,c,h
        
        x_h = self.conv_h(x_h)
        x_w = self.conv_w(x_w)
        
        x_h = x_h.permute(0,2,1,3)
        x_w = x_w.permute(0,2,3,1)
        
        out = x_h+x_w
        return out
        
def get_vgg1x1_spoolbnrelu(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = vgg1x1_spoolbnrelu(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
