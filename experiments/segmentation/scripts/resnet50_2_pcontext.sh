# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model resnet50_2 --base-size 256 --crop-size 256 \
    --checkname resnet50_2_pcontext --epochs 150

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model resnet50_2 --base-size 256 --crop-size 256 \
    --resume experiments/segmentation/runs/pcontext/resnet50_2/resnet50_2_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model resnet50_2 --base-size 256 --crop-size 256 \
    --resume experiments/segmentation/runs/pcontext/resnet50_2/resnet50_2_pcontext/model_best.pth.tar --split val --mode testval --ms