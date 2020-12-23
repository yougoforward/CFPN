# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_dpcan --dataset pcontext \
    --model fpn_dpcan --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname fpn_dpcan_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fpn_dpcan --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/fpn_dpcan/fpn_dpcan_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fpn_dpcan --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/fpn_dpcan/fpn_dpcan_res101_pcontext/model_best.pth.tar --split val --mode testval --ms