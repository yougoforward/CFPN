# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_object --dataset pcontext \
    --model object_center_cut_gsnet --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname object_center_cut_gsnet_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model object_center_cut_gsnet --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/object_center_cut_gsnet/object_center_cut_gsnet_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model object_center_cut_gsnet --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/object_center_cut_gsnet/object_center_cut_gsnet_res101_pcontext/model_best.pth.tar --split val --mode testval --ms