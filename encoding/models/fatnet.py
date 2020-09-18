from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['fatnet', 'get_fatnet']

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
class fatnet(nn.Module):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(fatnet, self).__init__()
        self._up_kwargs = up_kwargs

        self.base = fatnet_base(norm_layer)
        self.head = fatnetHead(48, nclass, norm_layer, up_kwargs=self._up_kwargs)

    def forward(self, x):
        imsize = x.size()[2:]
        x = self.base(x)
        x = self.head(x)
        x = F.interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        return tuple(outputs)



class fatnetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs=None):
        super(fatnetHead, self).__init__()
        self.se_loss = se_loss
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
        x1=self.layer1(x)
        x2=self.layer1(x)
        x3=self.layer1(x)
        x4=self.layer1(x)
        x5=self.layer1(x)
        x6=self.layer1(x)
        x7=self.layer1(x)
        x8=self.layer1(x)
        x9=self.layer1(x)
        x10=self.layer1(x)
        x11=self.layer1(x)
        x12=self.layer1(x)
        x13=self.layer1(x)
        return x13



class fatnet_layer(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, tl_size=1, norm_layer=nn.BatchNorm2d):
        super(fatnet_layer, self).__init__()
        self.tl_size = tl_size
        self.inplanes = in_planes
        self.outplanes = out_planes
        self.conv = nn.Sequential(nn.Conv2d(in_planes*tl_size^2, out_planes*tl_size^2, 3, padding=1, dilation=1, groups=tl_size^2, bias=False),
                                   norm_layer(out_planes*tl_size^2), nn.ReLU())
        # self.conv_list = nn.ModuleList([nn.Sequential(nn.Conv2d(in_planes, out_planes, 3, padding=1, dilation=1, bias=False),
        #                            norm_layer(out_planes), nn.ReLU()) for i in range(tl_size^2)])

    def forward(self, x):
        x_fat = pixelshuffle_invert(x, (self.tl_size, self.tl_size))
        # x_fat_list = torch.split(x_fat, self.inplanes, dim=1)
        # out = torch.cat([self.conv_list[i](x_fat_list[i]) for i in range(self.tl_size^2)], dim=1)
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


def pixelshuffle(x, factor_hw):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC//(pH*pW), iH*pH, iW*pW
    y = y.reshape(B, oC, pH, pW, iH, iW)
    y = y.permute(0, 1, 4, 2, 5, 3)     # B, oC, iH, pH, iW, pW
    y = y.reshape(B, oC, oH, oW)
    return y


def pixelshuffle_invert(x, factor_hw):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC*(pH*pW), iH//pH, iW//pW
    y = y.reshape(B, iC, oH, pH, oW, pW)
    y = y.permute(0, 1, 3, 5, 2, 4)     # B, iC, pH, pW, oH, oW
    y = y.reshape(B, oC, oH, oW)
    return y

# def pixelshuffle(x: torch.Tensor, factor_hw: tuple[int, int]):
#     pH = factor_hw[0]
#     pW = factor_hw[1]
#     y = x
#     B, iC, iH, iW = y.shape
#     oC, oH, oW = iC//(pH*pW), iH*pH, iW*pW
#     y = y.reshape(B, oC, pH, pW, iH, iW)
#     y = y.permute(0, 1, 4, 2, 5, 3)     # B, oC, iH, pH, iW, pW
#     y = y.reshape(B, oC, oH, oW)
#     return y


# def pixelshuffle_invert(x: torch.Tensor, factor_hw: tuple[int, int]):
#     pH = factor_hw[0]
#     pW = factor_hw[1]
#     y = x
#     B, iC, iH, iW = y.shape
#     oC, oH, oW = iC*(pH*pW), iH//pH, iW//pW
#     y = y.reshape(B, iC, oH, pH, oW, pW)
#     y = y.permute(0, 1, 3, 5, 2, 4)     # B, iC, pH, pW, oH, oW
#     y = y.reshape(B, oC, oH, oW)
#     return y