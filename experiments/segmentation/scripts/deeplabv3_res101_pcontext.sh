# !/usr/bin/env bash
# train
# python -m experiments.segmentation.train --dataset pcontext \
#     --model deeplab --aux --dilated --base-size 520 --crop-size 520 \
#     --backbone resnet101 --checkname deeplabv3_res101_pcontext

# # #test [single-scale]
# python -m experiments.segmentation.test_whole --dataset pcontext \
#     --model deeplab --aux --dilated --base-size 520 --crop-size 520 \
#     --backbone resnet101 --resume experiments/segmentation/runs/pcontext/deeplabv3/deeplabv3_res101_pcontext/model_best.pth.tar --split val --mode testval

# #test [multi-scale]
# python -m experiments.segmentation.test --dataset pcontext \
#     --model deeplab --aux --dilated --base-size 520 --crop-size 520 \
#     --backbone resnet101 --resume experiments/segmentation/runs/pcontext/deeplabv3/deeplabv3_res101_pcontext/model_best.pth.tar --split val --mode testval --ms

# python -m experiments.segmentation.test_whole --dataset pcontext \
#     --model deeplab --aux --dilated --base-size 520 --crop-size 520 \
#     --backbone resnet101 --resume experiments/segmentation/runs/pcontext/deeplabv3/deeplabv3_res101_pcontext/model_best.pth.tar --split val --mode test --ms

python -m experiments.segmentation.test_whole --dataset pcontext \
    --model deeplab --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/deeplabv3/deeplabv3_res101_pcontext/model_best.pth.tar --split val --mode test