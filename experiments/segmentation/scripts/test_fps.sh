#!/usr/bin/env bash

#fps
CUDA_VISIBLE_DEVICES=0 python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model fpn  \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=0 python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model fpn_aspp   \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=0 python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model fpn_pam   \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=0 python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model fcn  --dilated \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=0 python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model deeplab --dilated \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=0 python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model pam  --dilated \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=0 python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model cfpn  \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=0 python -m experiments.segmentation.test_fps_params --dataset pcontext \
    --model cfpn_gsf  \
    --backbone resnet101