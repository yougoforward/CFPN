from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet
from ..nn import PyramidPooling
__all__ = ['Upernet', 'get_Upernet']


class Upernet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Upernet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = UpernetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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



class UpernetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(UpernetHead, self).__init__()
        self.se_loss = se_loss
        self._up_kwargs = up_kwargs

        inter_channels = in_channels // 8
        self.nr_scene_class, self.nr_object_class, self.nr_part_class, self.nr_material_class = \
           10, 20, 7, out_channels
        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))

        self.localUp2=localUp(256, inter_channels, norm_layer, up_kwargs)
        self.localUp3=localUp(512, inter_channels, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, inter_channels, norm_layer, up_kwargs)
        self.psp = nn.Sequential(PyramidPooling(in_channels, norm_layer, up_kwargs),
                                   nn.Conv2d(in_channels*2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True))
        
        self.conv_fusion = nn.Sequential(
                                   nn.Conv2d(inter_channels*4, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True))
        self.material_head = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
            nn.Conv2d(inter_channels, self.nr_material_class, kernel_size=1, bias=True)
        )
        # self.scene_head = nn.Sequential(
        #     nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU(True),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(inter_channels, self.nr_scene_class, kernel_size=1, bias=True)
        # )
        # self.object_head = nn.Sequential(
        #     nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU(True),
        #     nn.Conv2d(inter_channels, self.nr_object_class, kernel_size=1, bias=True)
        # )

        # input: Fusion out, input_dim: fpn_dim
        # self.part_head = nn.Sequential(
        #     nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU(True),
        #     nn.Conv2d(inter_channels, self.nr_part_class, kernel_size=1, bias=True)
        # )
        
    def forward(self, c1,c2,c3,c4):
        _,_, h,w = c1.size()
        p5 = self.psp(c4)
        
        
        p4 = self.localUp4(c3, p5)  
        p3 = self.localUp3(c2, p4)
        p2 = self.localUp2(c1, p3)
        
        p5 = F.interpolate(p5, (h,w), **self._up_kwargs)
        p4 = F.interpolate(p4, (h,w), **self._up_kwargs)
        p3 = F.interpolate(p3, (h,w), **self._up_kwargs)
        fpn_list = [p2, p3, p4, p5]
        
        fuse = self.conv_fusion(torch.cat(fpn_list, dim=1))
        out = self.material_head(fuse)
        # scene = self.scene_head(p5)
        # object = self.scene_head(fuse)
        # part = self.scene_head(fuse)
        
        return out

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

def get_Upernet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = Upernet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model

