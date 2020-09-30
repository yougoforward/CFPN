# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model resnet50_5 --base-size 256 --aux --crop-size 256 \
    --checkname resnet50_5_pcontext --epochs 150

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model resnet50_5 --base-size 256 --aux --crop-size 256 \
    --resume experiments/segmentation/runs/pcontext/resnet50_5/resnet50_5_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model resnet50_5 --base-size 256 --aux --crop-size 256 \
    --resume experiments/segmentation/runs/pcontext/resnet50_5/resnet50_5_pcontext/model_best.pth.tar --split val --mode testval --ms