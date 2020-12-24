# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_dpcan --dataset pcontext \
    --model dpcan --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname dpcan_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model dpcan --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/dpcan/dpcan_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model dpcan --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/dpcan/dpcan_res101_pcontext/model_best.pth.tar --split val --mode testval --ms