# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset citys  --train-split trainval \
    --model cfpn_gsf --aux --base-size 1024 --crop-size 768 \
    --backbone resnet101 --checkname cfpn_gsf_res101_citys_test3 --batch-size 8 --lr 0.003 --epochs 240 --resume experiments/segmentation/runs/citys/cfpn_gsf/cfpn_gsf_res101_citys/model_best.pth.tar --ft

#test [single-scale]
python -m experiments.segmentation.test --dataset citys \
    --model cfpn_gsf --aux --base-size 2048 --crop-size 768 \
    --backbone resnet101 --resume experiments/segmentation/runs/citys/cfpn_gsf/cfpn_gsf_res101_citys_test3/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset citys \
    --model cfpn_gsf --aux --base-size 2048 --crop-size 768 \
    --backbone resnet101 --resume experiments/segmentation/runs/citys/cfpn_gsf/cfpn_gsf_res101_citys_test3/model_best.pth.tar --split test --mode test --ms --save-folder experiments/segmentation/results/cfpn_gsf_res101_citys_test3