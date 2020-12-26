from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['fpn_psaa', 'get_fpn_psaa']


class fpn_psaa(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(fpn_psaa, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = fpn_psaaHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        c0, c1, c2, c3, c4 = self.base_forward(x)
        x = self.head(c0, c1,c2,c3,c4)
        x = F.interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)



class fpn_psaaHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(fpn_psaaHead, self).__init__()
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
        # self.aspp = ASPP_Module(in_channels, 256, inter_channels, atrous_rates, norm_layer, up_kwargs)
        self.psaa = PSAA_Module(in_channels, 256, inter_channels, atrous_rates, norm_layer, up_kwargs)

    def forward(self, c0, c1,c2,c3,c4):
        _,_, h,w = c2.size()
               
        out3 = self.localUp4(c3, c4)  
        out2 = self.localUp3(c2, out3)
        
        # out = self.conv5(out2)
        # out = self.aspp(out2)
        out = self.psaa(out2)
        
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

def get_fpn_psaa(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = fpn_psaa(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model

def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    # block = nn.Sequential(
    #     nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
    #               dilation=atrous_rate, bias=False),
    #     norm_layer(out_channels),
    #     nn.ReLU(True))
    block = nn.Sequential(
        nn.Conv2d(in_channels, 512, 1, padding=0,
                  dilation=1, bias=False),
        norm_layer(512),
        nn.ReLU(True),
        nn.Conv2d(512, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block

class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h,w), **self._up_kwargs)

class ASPP_Module(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, inter_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, inter_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, inter_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, inter_channels, norm_layer, up_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5*inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))


    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)

        return self.project(y)
    
class PSAA_Module(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(PSAA_Module, self).__init__()
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, inter_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, inter_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, inter_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(4*inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.psaa_conv = nn.Sequential(nn.Conv2d(in_channels+inter_channels*4, inter_channels, 1, padding=0, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(inter_channels, 4, 1, bias=True),
                                    nn.Sigmoid())
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_channels, out_channels, 1, bias=False),
                            norm_layer(out_channels),
                            nn.ReLU(True))
        self.se = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, 1, bias=True),
                            nn.Sigmoid())

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        # feat4 = self.b4(x)
        n, c, h, w = feat0.size()
        #gp
        gp = self.gap(x)
        se = self.se(gp)
        feat4 = gp.expand(n, 512, h, w)
        
        # psaa_att = self.psaa_conv(torch.cat([x, feat0, feat1, feat2, feat3], dim=1))
        # psaa_att_list = torch.split(psaa_att, 1, dim=1)

        # y = torch.cat((psaa_att_list[0] * feat0, psaa_att_list[1] * feat1, psaa_att_list[2] * feat2,
        #                 psaa_att_list[3] * feat3), 1)

        # y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        y = torch.cat((feat0, feat1, feat2, feat3), 1)
        out = self.project(y)
        out = torch.cat([out+out*se, feat4], dim=1)

        return out