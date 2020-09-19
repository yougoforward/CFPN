# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model fatnet1 --base-size 512 --crop-size 512 \
    --checkname fatnet1_pcontext --epochs 150

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fatnet1 --base-size 512 --crop-size 512 \
    --resume experiments/segmentation/runs/pcontext/fatnet1/fatnet1_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fatnet1 --base-size 512 --crop-size 512 \
    --resume experiments/segmentation/runs/pcontext/fatnet1/fatnet1_pcontext/model_best.pth.tar --split val --mode testval --ms