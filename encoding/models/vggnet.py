from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['vggnet', 'get_vggnet']


class vggnet(nn.Module):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(vggnet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.base = vggnet_base(norm_layer)
        self.head = vggnetHead(512, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)

    def forward(self, x):
        imsize = x.size()[2:]
        x = self.base(x)
        x = self.head(x)
        x = F.interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        return tuple(outputs)



class vggnetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(vggnetHead, self).__init__()
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

class vggnet_base(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(vggnet_base, self).__init__()
        self.layer1 = vggnet_layer(3,64,1,1,norm_layer)
        self.layer2 = vggnet_layer(64,64,1,1,norm_layer)
        self.pool = nn.MaxPool2d(2)
        
        self.layer3 = vggnet_layer(64,128,1,1,norm_layer)
        self.layer4 = vggnet_layer(128,128,1,1,norm_layer)

        self.layer5 = vggnet_layer(128,256,1,1,norm_layer)
        self.layer6 = vggnet_layer(256,256,1,1,norm_layer)
        self.layer7 = vggnet_layer(256,256,1,1,norm_layer)
        
        self.layer8 = vggnet_layer(256,512,1,1,norm_layer)
        self.layer9 = vggnet_layer(512,512,1,1,norm_layer)
        self.layer10 = vggnet_layer(512,512,1,1,norm_layer)
        
        self.layer11 = vggnet_layer(512,512,1,1,norm_layer)
        self.layer12 = vggnet_layer(512,512,1,1,norm_layer)
        self.layer13 = vggnet_layer(512,512,1,1,norm_layer)


    def forward(self, x):
        x1=self.layer1(x)
        x2=self.layer1(x1)
        x_pool1=self.pool(x2)
        
        x3=self.layer1(x_pool1)
        x4=self.layer1(x3)
        x_pool2=self.pool(x4)
        
        x5=self.layer1(x_pool2)
        x6=self.layer1(x5)
        x7=self.layer1(x6)
        x_pool3=self.pool(x7)
        
        x8=self.layer1(x_pool3)
        x9=self.layer1(x8)
        x10=self.layer1(x9)
        x_pool1=self.pool(x10)
        
        x11=self.layer1(x)
        x12=self.layer1(x)
        x13=self.layer1(x)
        return x13



class vggnet_layer(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, tl_size=1, norm_layer=nn.BatchNorm2d):
        super(vggnet_layer, self).__init__()
        self.tl_size = tl_size
        self.inplanes = in_planes
        self.outplanes = out_planes
        self.conv = nn.Sequential(nn.Conv2d(in_planes*tl_size^2, out_planes*tl_size^2, 3, padding=1, dilation=1, bias=False),
                                   norm_layer(out_planes*tl_size^2), nn.ReLU())

    def forward(self, x):
        out = self.conv(x)
        return out


def get_vggnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = vggnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
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