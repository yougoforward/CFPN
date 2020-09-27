from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['vgg_sa_spool', 'get_vgg_sa_spool']

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class vgg_sa_spool(nn.Module):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], **kwargs):
        super(vgg_sa_spool, self).__init__()
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        self._up_kwargs = up_kwargs
        self.base = vgg_sa_spool_base(norm_layer)
        self.head = vgg_sa_spoolHead(512, nclass, norm_layer, up_kwargs=self._up_kwargs)

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



class vgg_sa_spoolHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs=None):
        super(vgg_sa_spoolHead, self).__init__()
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

class vgg_sa_spool_base(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(vgg_sa_spool_base, self).__init__()
        self.layer1 = vgg_sa_spool_layer(3,64,1,1,norm_layer)
        self.layer2 = vgg_sa_spool_layer(64,64,1,1,norm_layer)
        self.layer2_spool = vgg_sa_spool_layer3(64,64,1,1,256,256,norm_layer)
        self.pool = nn.MaxPool2d(2)
        
        self.layer3 = vgg_sa_spool_layer(64,128,1,1,norm_layer)
        self.layer4 = vgg_sa_spool_layer(128,128,1,1,norm_layer)
        self.layer4_spool = vgg_sa_spool_layer3(128,128,1,1,128,128,norm_layer)

        self.layer5 = vgg_sa_spool_layer(128,256,1,1,norm_layer)
        self.layer6 = vgg_sa_spool_layer(256,256,1,1,norm_layer)
        self.layer7 = vgg_sa_spool_layer(256,256,1,1,norm_layer)
        self.layer7_spool = vgg_sa_spool_layer3(256,256,1,1,64,64,norm_layer)
        
        self.layer8 = vgg_sa_spool_layer(256,512,1,1,norm_layer)
        self.layer9 = vgg_sa_spool_layer(512,512,1,1,norm_layer)
        self.layer10 = vgg_sa_spool_layer(512,512,1,1,norm_layer)
        self.layer10_spool = vgg_sa_spool_layer3(512,512,1,1,32,32,norm_layer)
        
        self.layer11 = vgg_spool_layer(512,512,1,1,norm_layer)
        self.layer12 = vgg_spool_layer(512,512,1,1,norm_layer)
        self.layer13 = vgg_spool_layer(512,512,1,1,norm_layer)
        self.layer13_spool = vgg_spool_layer3(512,512,1,1,16,16,norm_layer)

    def forward(self, x):
        x1=self.layer1(x)
        x2=self.layer2(x1)
        x2=self.layer2_spool(x2)
        x2=self.pool(x2)
        
        x3=self.layer3(x2)
        x4=self.layer4(x3)
        x4=self.layer4_spool(x4)
        x4=self.pool(x4)
        
        x5=self.layer5(x4)
        x6=self.layer6(x5)
        x7=self.layer7(x6)
        x7=self.layer7_spool(x7)
        x7=self.pool(x7)
        
        x8=self.layer8(x7)
        x9=self.layer9(x8)
        x10=self.layer10(x9)
        x10=self.layer10_spool(x10)
        x10=self.pool(x10)
        
        x11=self.layer11(x10)
        x12=self.layer12(x11)
        x13=self.layer13(x12)
        x13=self.layer13_spool(x13)
        return x13



class vgg_sa_spool_layer(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, tl_size=1, norm_layer=nn.BatchNorm2d):
        super(vgg_sa_spool_layer, self).__init__()
        self.tl_size = tl_size
        self.inplanes = in_planes
        self.outplanes = out_planes
        self.conv = nn.Sequential(nn.Conv2d(in_planes*tl_size*tl_size, out_planes*tl_size*tl_size, 3, padding=1, dilation=1, bias=False),
                                   norm_layer(out_planes*tl_size*tl_size), nn.ReLU())

    def forward(self, x):
        out = self.conv(x)
        return out
    
class vgg_sa_spool_layer2(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, tl_size=1, norm_layer=nn.BatchNorm2d):
        super(vgg_sa_spool_layer2, self).__init__()
        self.tl_size = tl_size
        self.inplanes = in_planes
        self.outplanes = out_planes
        self.conv = nn.Sequential(nn.Conv2d(in_planes*tl_size*tl_size, out_planes*tl_size*tl_size, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(out_planes*tl_size*tl_size), nn.ReLU())

    def forward(self, x):
        out = self.conv(x)
        return out

class vgg_sa_spool_layer3(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, tl_size=1, height=256, weight=256, norm_layer=nn.BatchNorm2d):
        super(vgg_sa_spool_layer3, self).__init__()
        self.tl_size = tl_size
        self.inplanes = in_planes
        self.outplanes = out_planes
        self.sa = SA_Module(in_planes//4, in_planes//8, norm_layer)
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
        x1 = self.sa(x1)
        x1 = self.spool(x1)
        x1 = self.conv2(x1)
        out = self.conv(x)
        out = self.relu(x1+out)
        return out

class SA_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, key_dim, norm_layer):
        super(SA_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        # self.project = nn.Sequential(nn.Conv2d(in_dim, value_dim, 1, padding=0, dilation=1, bias=False),
        #                            norm_layer(value_dim), nn.ReLU())
        self.key_dim = key_dim


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        n, c, h, w = x.size()
        c2 = self.key_dim
        query = self.query_conv(x)
        key = self.key_conv(x)
        # value = self.project(x)
        value = x
        
        key_h = query.permute(0,3,1,2)#n,w,c2,h
        key_w = query.permute(0,2,1,3)#n,h,c2,w
        query_h = key.permute(0,3,2,1)#n,w,h,c2
        query_w = key.permute(0,2,3,1)#n,h,w,c2
        
        #h attention
        energy_h = torch.matmul(query_h, key_h)#n,w,h,h
        attention_h = torch.softmax(energy_h, -1)
        value_h = value.permute(0,3,2,1)#n,w,h,c
        value_h = torch.matmul(attention_h,value_h)#n,w,h,c
        value_h = value_h.permute(0,3,2,1)
        
        #w attention
        energy_w = torch.matmul(query_w, key_w)#n,h,w,w
        attention_w = torch.softmax(energy_w, -1)
        value_w = value.permute(0,2,3,1)#n,h,w,c
        value_w = torch.matmul(attention_w,value_w)#n,h,w,c
        value_w = value_w.permute(0,3,1,2)
        
        out = value_h+value_w
        return out
    
class SPool(nn.Module):
    def __init__(self, height, width, norm_layer):
        super(SPool, self).__init__()
        self.conv_h = nn.Sequential(nn.Conv2d(height, height, 1, padding=0, dilation=1, bias=False))
        self.conv_w = nn.Sequential(nn.Conv2d(width, width, 1, padding=0, dilation=1, bias=False))

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
        
def get_vgg_sa_spool(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = vgg_sa_spool(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
