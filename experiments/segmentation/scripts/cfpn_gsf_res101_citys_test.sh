# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset citys  --train-split trainval \
    --model cfpn_gsf --aux --base-size 1024 --crop-size 769 \
    --backbone resnet101 --checkname cfpn_gsf_res101_citys_test --batch-size 8 --lr 0.002 --epochs 300

#test [single-scale]
python -m experiments.segmentation.test --dataset citys \
    --model cfpn_gsf --aux --base-size 2048 --crop-size 769 \
    --backbone resnet101 --resume experiments/segmentation/runs/citys/cfpn_gsf/cfpn_gsf_res101_citys_test/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset citys \
    --model cfpn_gsf --aux --base-size 2048 --crop-size 769 \
    --backbone resnet101 --resume experiments/segmentation/runs/citys/cfpn_gsf/cfpn_gsf_res101_citys_test/model_best.pth.tar --split test --mode test --ms