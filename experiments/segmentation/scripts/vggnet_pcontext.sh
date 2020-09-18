# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model vggnet --base-size 520 --crop-size 520 \
    --checkname vggnet_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model vggnet --base-size 520 --crop-size 520 \
    --resume experiments/segmentation/runs/pcontext/vggnet/vggnet_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model vggnet --base-size 520 --crop-size 520 \
    --resume experiments/segmentation/runs/pcontext/vggnet/vggnet_pcontext/model_best.pth.tar --split val --mode testval --ms