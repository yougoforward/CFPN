# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model vgg1x1_spool2bnrelu --base-size 256 --crop-size 256 \
    --checkname vgg1x1_spool2bnrelu_pcontext --epochs 150

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model vgg1x1_spool2bnrelu --base-size 256 --crop-size 256 \
    --resume experiments/segmentation/runs/pcontext/vgg1x1_spool2bnrelu/vgg1x1_spool2bnrelu_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model vgg1x1_spool2bnrelu --base-size 256 --crop-size 256 \
    --resume experiments/segmentation/runs/pcontext/vgg1x1_spool2bnrelu/vgg1x1_spool2bnrelu_pcontext/model_best.pth.tar --split val --mode testval --ms