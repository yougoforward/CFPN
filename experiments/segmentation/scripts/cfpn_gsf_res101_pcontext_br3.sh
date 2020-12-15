# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_br3 --dataset pcontext3 \
    --model cfpn_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname cfpn_gsf_res101_pcontext_br3

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model cfpn_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext3/cfpn_gsf/cfpn_gsf_res101_pcontext_br3/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model cfpn_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext3/cfpn_gsf/cfpn_gsf_res101_pcontext_br3/model_best.pth.tar --split val --mode testval --ms