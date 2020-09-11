from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['fatnet', 'get_fatnet']


class fatnet(nn.Module):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(fatnet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = fatnetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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



class fatnetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(fatnetHead, self).__init__()
        self.se_loss = se_loss
        self._up_kwargs = up_kwargs

        inter_channels = in_channels // 4
        # self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU(),
        #                            )
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                            norm_layer(inter_channels),
                            nn.ReLU(True))
        self.se = nn.Sequential(
                            nn.Conv2d(inter_channels, inter_channels, 1, bias=True),
                            nn.Sigmoid())
        self.gff = PAM_Module(in_dim=inter_channels, key_dim=inter_channels//8,value_dim=inter_channels,out_dim=inter_channels,norm_layer=norm_layer)

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(2*inter_channels, out_channels, 1))

        self.localUp3=localUp(512, inter_channels, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, inter_channels, norm_layer, up_kwargs)

        self.context4 = Context(in_channels, inter_channels, inter_channels, 8, norm_layer)
        self.project4 = nn.Sequential(nn.Conv2d(2*inter_channels, inter_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(inter_channels), nn.ReLU())
        self.context3 = Context(inter_channels, inter_channels, inter_channels, 8, norm_layer)
        self.project3 = nn.Sequential(nn.Conv2d(2*inter_channels, inter_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(inter_channels), nn.ReLU())
        self.context2 = Context(inter_channels, inter_channels, inter_channels, 8, norm_layer)

        self.project = nn.Sequential(nn.Conv2d(6*inter_channels, inter_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   )
    def forward(self, c1,c2,c3,c4):
        _,_, h,w = c2.size()
        cat4, p4_1, p4_8=self.context4(c4)
        p4 = self.project4(cat4)
                
        out3 = self.localUp4(c3, p4)
        cat3, p3_1, p3_8=self.context3(out3)
        p3 = self.project3(cat3)
        
        out2 = self.localUp3(c2, p3)
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
        out = self.gff(out)

        #
        out = torch.cat([out, gp.expand_as(out)], dim=1)

        return self.conv6(out)

class fatnet_base(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(fatnet_base, self).__init__()
        self.layer1 = fatnet_layer(3,64,1,1,norm_layer)
        self.layer2 = fatnet_layer(64,64,1,1,norm_layer)
        
        self.layer3 = fatnet_layer(64,48,1,2,norm_layer)
        self.layer4 = fatnet_layer(48,48,1,2,norm_layer)
        
        self.layer5 = fatnet_layer(48,48,1,4,norm_layer)
        self.layer6 = fatnet_layer(48,48,1,4,norm_layer)
        self.layer7 = fatnet_layer(48,48,1,4,norm_layer)
        
        self.layer8 = fatnet_layer(48,48,1,8,norm_layer)
        self.layer9 = fatnet_layer(48,48,1,8,norm_layer)
        self.layer10 = fatnet_layer(48,48,1,8,norm_layer)
        
        self.layer11 = fatnet_layer(48,48,1,16,norm_layer)
        self.layer12 = fatnet_layer(48,48,1,16,norm_layer)
        self.layer13 = fatnet_layer(48,48,1,16,norm_layer)


    def forward(self, x):
        
        return out



class fatnet_layer(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, tl_size=1, norm_layer=nn.BatchNorm2d):
        super(fatnet_layer, self).__init__()
        self.tl_size = tl_size
        self.out_planes = out_planes
        self.conv = nn.Sequential(nn.Conv2d(in_planes*tl_size^2, out_planes*tl_size^2, 3, padding=1, dilation=1, groups=tl_size^2, bias=False),
                                   norm_layer(out_planes*tl_size^2), nn.ReLU())

    def forward(self, x):
        x_fat = pixelshuffle_invert(x, (self.tl_size, self.tl_size))
        out = self.conv(x_fat)
        out = pixelshuffle(out, (self.tl_size, self.tl_size))
        return out


def get_fatnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = fatnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model

class ASPP_TLConv(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=[1], tl_size=1):
        super(ASPP_TLConv, self).__init__()
        self.conv_list = nn.ModuleList()
        self.tl_size = tl_size
        for i in range(tl_size * tl_size):
            self.conv_list.append(
                ASPP(in_planes, out_planes, dilation, stride=tl_size)
            )

    def forward(self, x):
        out = []
        conv_id = 0
        for i in range(self.tl_size):
            for j in range(self.tl_size):
                y = F.pad(x, pad=(-j, j, -i, i))
                out.append(self.conv_list[conv_id](y))
                conv_id += 1

        outs = torch.cat(out, 1)
        outs = F.pixel_shuffle(outs, upscale_factor=self.tl_size)

        return outs
    
def pixelshuffle(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC//(pH*pW), iH*pH, iW*pW
    y = y.reshape(B, oC, pH, pW, iH, iW)
    y = y.permute(0, 1, 4, 2, 5, 3)     # B, oC, iH, pH, iW, pW
    y = y.reshape(B, oC, oH, oW)
    return y


def pixelshuffle_invert(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC*(pH*pW), iH//pH, iW//pW
    y = y.reshape(B, iC, oH, pH, oW, pW)
    y = y.permute(0, 1, 3, 5, 2, 4)     # B, iC, pH, pW, oH, oW
    y = y.reshape(B, oC, oH, oW)
    return y