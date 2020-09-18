# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model fatnet --base-size 512 --crop-size 512 \
    --checkname fatnet_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fatnet --base-size 512 --crop-size 512 \
    --resume experiments/segmentation/runs/pcontext/fatnet/fatnet_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fatnet --base-size 512 --crop-size 512 \
    --resume experiments/segmentation/runs/pcontext/fatnet/fatnet_pcontext/model_best.pth.tar --split val --mode testval --ms