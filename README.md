# Context Feature Pyramid Network
[[arXiv Paper]](https://arxiv.org/abs/1903.11816) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fastfcn-rethinking-dilated-convolution-in-the/semantic-segmentation-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-pascal-context?p=fastfcn-rethinking-dilated-convolution-in-the)

Official implementation of **CFPN: Context Feature Pyramid Network**.   
A **Faster**, **Stronger** and **Lighter** framework for semantic segmentation, achieving the state-of-the-art performance and more than **2x** acceleration compared to deep dilation based models.
```
@inproceedings{zhu2021cfpn,
  title     = {CFPN: Context Feature Pyramid Network},
  author    = {Zhu, Hailong, Pang, Yanwei},
  booktitle = {arXiv preprint arXiv:},
  year = {2021}
}
```
Contact: Hailong Zhu (hlzhu2009@outlook.com)

## Usage Tips:
## Training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.train --dataset [pcontext|ade20k|cocostuff] \
    --model [cfpn_gsf] --aux --base-size 520 --crop-size 520 \
    --backbone [resnet50|resnet101] --checkname [cfpn_gsf_res101_pcontext]
```

## validation:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test_whole --dataset [pcontext|ade20k|cocostuff] \
    --model [cfpn_gsf] --base-size 520 --crop-size 520\
    --backbone [resnet50|resnet101] --resume {MODEL_PATH .pth} --split val --mode testval
```
## testing:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test_whole --dataset [pcontext|ade20k|cocostuff] \
    --model [cfpn_gsf] --base-size 520 --crop-size 520\
    --backbone [resnet50|resnet101] --resume {MODEL_PATH .pth} --split [val|test] --mode test [--ms]
```
## Test on a single image:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test_single_image  --dataset [pcontext|ade20k|cocostuff] \
    --model [cfpn_gsf] --base-size 520 --crop-size 520\
    --backbone [resnet50|resnet101] [--ms] --resume {MODEL_PATH .pth} --input-path {INPUT} --save-path {OUTPUT}
```

## Supporting:
1.Python>= 3.6
2.Pytorch>=1.0

## Overview
### Architecture
![](images/Framework.png)

## Install
1. [PyTorch >= 1.1.0](https://pytorch.org/get-started/locally) (Note: The code is test in the environment with `python=3.6, cuda=9.0`)
2. Download **CFPN**
   ```
   git clone https://github.com/yougoforward/CFPN.git
   cd CFPN
   ```
3. Install Requirements
   ```
   nose
   tqdm
   scipy
   cython
   requests
   ```

## Train and Test results on benchmarks
### PContext
```
python -m scripts.prepare_pcontext
```
| Method | Backbone | mIoU | FPS | Model | Scripts |
|:----|:----|:---:|:---:|:---:|:---:|
| CFPN | ResNet-101 | **57.0 (MS)** | **41.5** | [BaiduNetDisk]() | [bash](experiments/segmentation/scripts/cfpn_gsf_res101_pcontext.sh) |

### ADE20K
```
python -m scripts.prepare_ade20k
```
#### Val Set
| Method | Backbone | mIoU (MS) | Model | Scripts |
|:----|:----|:---:|:---:|:---:|
| CFPN | ResNet-101 | 47.5 | [BaiduNetDisk]() | [bash](experiments/segmentation/scripts/cfpn_gsf_res101_ade20k.sh) |

### Cocostuff 10k
#### Val Set
| Method | Backbone | mIoU (MS) | Model | Scripts |
|:----|:----|:---:|:---:|:---:|
| CFPN | ResNet-101 | 42.8 | [BaiduNetDisk]() | [bash](experiments/segmentation/scripts/cfpn_gsf_res101_cocostuff.sh) |

**Note:** All models are trained with `crop_size=520 and short_size=520` and tested with 'long_size=520'.

## Acknowledgement
Code borrows heavily from [FastFCN](https://github.com/wuhuikai/FastFCN) and [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)..
