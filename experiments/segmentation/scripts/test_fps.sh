#!/usr/bin/env bash

#fps
python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model fcn  \
    --backbone resnet101

python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model fpn  \
    --backbone resnet101

python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model fcn  --dilated \
    --backbone resnet101

python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model deeplab --dilated \
    --backbone resnet101

python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model psp  --dilated \
    --backbone resnet101 

python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model pam  --dilated \
    --backbone resnet101

python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model cfpn_gsf  \
    --backbone resnet101

