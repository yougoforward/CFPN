# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model vgg1x1_spool --base-size 320 --crop-size 320 \
    --checkname vgg1x1_spool_pcontext --epochs 150

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model vgg1x1_spool --base-size 320 --crop-size 320 \
    --resume experiments/segmentation/runs/pcontext/vgg1x1_spool/vgg1x1_spool_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model vgg1x1_spool --base-size 320 --crop-size 320 \
    --resume experiments/segmentation/runs/pcontext/vgg1x1_spool/vgg1x1_spool_pcontext/model_best.pth.tar --split val --mode testval --ms