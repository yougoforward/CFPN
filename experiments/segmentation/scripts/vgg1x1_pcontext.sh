# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model vgg1x1 --base-size 320 --crop-size 320 \
    --checkname vgg1x1_pcontext --epochs 150

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model vgg1x1 --base-size 320 --crop-size 320 \
    --resume experiments/segmentation/runs/pcontext/vgg1x1/vgg1x1_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model vgg1x1 --base-size 320 --crop-size 320 \
    --resume experiments/segmentation/runs/pcontext/vgg1x1/vgg1x1_pcontext/model_best.pth.tar --split val --mode testval --ms