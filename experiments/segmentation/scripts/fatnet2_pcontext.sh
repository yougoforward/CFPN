# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model fatnet2 --base-size 320 --crop-size 320 \
    --checkname fatnet2_pcontext --epochs 150

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fatnet2 --base-size 320 --crop-size 320 \
    --resume experiments/segmentation/runs/pcontext/fatnet2/fatnet2_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fatnet2 --base-size 320 --crop-size 320 \
    --resume experiments/segmentation/runs/pcontext/fatnet2/fatnet2_pcontext/model_best.pth.tar --split val --mode testval --ms