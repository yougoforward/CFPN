# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model psp --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname psp_res101_pcontext --dilated

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model psp --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/psp/psp_res101_pcontext/model_best.pth.tar --split val --mode testval --dilated

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model psp --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/psp/psp_res101_pcontext/model_best.pth.tar --split val --mode testval --ms --dilated