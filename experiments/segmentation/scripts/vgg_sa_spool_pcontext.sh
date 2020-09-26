# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model vgg_sa_spool --base-size 256 --crop-size 256 \
    --checkname vgg_sa_spool_pcontext --epochs 150

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model vgg_sa_spool --base-size 256 --crop-size 256 \
    --resume experiments/segmentation/runs/pcontext/vgg_sa_spool/vgg_sa_spool_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model vgg_sa_spool --base-size 256 --crop-size 256 \
    --resume experiments/segmentation/runs/pcontext/vgg_sa_spool/vgg_sa_spool_pcontext/model_best.pth.tar --split val --mode testval --ms