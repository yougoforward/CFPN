# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model dfpn84_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname dfpn84_gsf_res50_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model dfpn84_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/dfpn84_gsf/dfpn84_gsf_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model dfpn84_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/dfpn84_gsf/dfpn84_gsf_res50_pcontext/model_best.pth.tar --split val --mode testval --ms