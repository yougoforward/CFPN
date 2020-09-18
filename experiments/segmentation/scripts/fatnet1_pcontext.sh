# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model fatnet1 --base-size 520 --crop-size 520 \
    --checkname fatnet1_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fatnet1 --base-size 520 --crop-size 520 \
    --resume experiments/segmentation/runs/pcontext/fatnet1/fatnet1_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fatnet1 --base-size 520 --crop-size 520 \
    --resume experiments/segmentation/runs/pcontext/fatnet1/fatnet1_pcontext/model_best.pth.tar --split val --mode testval --ms