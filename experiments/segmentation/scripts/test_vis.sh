# # !/usr/bin/env bash
#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset cocostuff \
    --model cfpn_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/cocostuff/cfpn_gsf/cfpn_gsf_res101_cocostuff/model_best.pth.tar --split val --mode test --ms

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset ade20k \
    --model cfpn_gsf --aux --base-size 576 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/ade20k/cfpn_gsf/cfpn_gsf_res101_ade20k/model_best.pth.tar --split val --mode test --ms

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model cfpn_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/cfpn_gsf/cfpn_gsf_res101_pcontext/model_best.pth.tar --split val --mode test --ms