from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['sfpn', 'get_sfpn']


class sfpn(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(sfpn, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = sfpnHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        c0, c1, c2, c3, c4 = self.base_forward(x)
        x = self.head(c0,c1,c2,c3,c4)
        x = F.interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)



class sfpnHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(sfpnHead, self).__init__()
        self.se_loss = se_loss
        self._up_kwargs = up_kwargs

        inter_channels = in_channels // 4
        # self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU(),
        #                            )

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(2*inter_channels, out_channels, 1))

        self.localUp3=localUp(512, in_channels, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, in_channels, norm_layer, up_kwargs)
        
        self.context4 = Context(in_channels, inter_channels, inter_channels, 8, norm_layer)
        self.context3 = Context(in_channels, inter_channels, inter_channels, 8, norm_layer)
        self.context2 = Context(in_channels, inter_channels, inter_channels, 8, norm_layer)
        self.project = nn.Sequential(nn.Conv2d(6*inter_channels, inter_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                            norm_layer(inter_channels),
                            nn.ReLU(True))
        self.se = nn.Sequential(
                            nn.Conv2d(inter_channels, inter_channels, 1, bias=True),
                            nn.Sigmoid())
        

    def forward(self, c0,c1,c2,c3,c4):
        _,_, h,w = c2.size()
               
        out3 = self.localUp4(c3, c4)  
        out2 = self.localUp3(c2, out3)
        
        cat4, p4_1, p4_8=self.context4(c4)
        cat3, p3_1, p3_8=self.context3(out3)
        cat2, p2_1, p2_8=self.context2(out2)
        
        p4_1 = F.interpolate(p4_1, (h,w), **self._up_kwargs)
        p4_8 = F.interpolate(p4_8, (h,w), **self._up_kwargs)
        p3_1 = F.interpolate(p3_1, (h,w), **self._up_kwargs)
        p3_8 = F.interpolate(p3_8, (h,w), **self._up_kwargs)
        out = self.project(torch.cat([p2_1,p2_8,p3_1,p3_8,p4_1,p4_8], dim=1))
        
        #gp
        gp = self.gap(c4)    
        # se
        se = self.se(gp)
        out = out + se*out
        #
        out = torch.cat([out, gp.expand_as(out)], dim=1)
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
    
class Context(nn.Module):
    def __init__(self, in_channels, width, out_channels, dilation_base, norm_layer):
        super(Context, self).__init__()
        self.dconv0 = nn.Sequential(nn.Conv2d(in_channels, width, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(width), nn.ReLU())
        self.dconv1 = nn.Sequential(nn.Conv2d(in_channels, width, 3, padding=dilation_base, dilation=dilation_base, bias=False),
                                   norm_layer(width), nn.ReLU())

    def forward(self, x):
        feat0 = self.dconv0(x)
        feat1 = self.dconv1(x)
        cat = torch.cat([feat0, feat1], dim=1)  
        return cat, feat0, feat1
       
# class localUp(nn.Module):
#     def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
#         super(localUp, self).__init__()
#         self.connect = nn.Sequential(nn.Conv2d(in_channels, out_channels//4, 1, padding=0, dilation=1, bias=False),
#                                    norm_layer(out_channels//4),
#                                    nn.ReLU())
#         self.project = nn.Sequential(nn.Conv2d(out_channels, out_channels//4, 1, padding=0, dilation=1, bias=False),
#                                    norm_layer(out_channels//4),
#                                    nn.ReLU())

#         self._up_kwargs = up_kwargs
#         self.refine = nn.Sequential(nn.Conv2d(out_channels//2, out_channels//4, 3, padding=1, dilation=1, bias=False),
#                                    norm_layer(out_channels//4),
#                                    nn.ReLU(),
#                                     )
#         self.project2 = nn.Sequential(nn.Conv2d(out_channels//4, out_channels, 1, padding=0, dilation=1, bias=False),
#                                    norm_layer(out_channels),
#                                    )
#         self.relu = nn.ReLU()
        
#     def forward(self, c1,c2):
#         n,c,h,w =c1.size()
#         c1p = self.connect(c1) # n, 64, h, w
#         c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
#         c2p = self.project(c2)
#         out = torch.cat([c1p,c2p], dim=1)
#         out = self.refine(out)
#         out = self.project2(out)
#         out = self.relu(c2+out)
#         return out

def get_sfpn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = sfpn(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model