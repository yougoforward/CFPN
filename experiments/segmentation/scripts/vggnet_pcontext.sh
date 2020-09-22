# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model vggnet --base-size 320 --crop-size 320 \
    --checkname vggnet_pcontext --epochs 150

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model vggnet --base-size 320 --crop-size 320 \
    --resume experiments/segmentation/runs/pcontext/vggnet/vggnet_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model vggnet --base-size 320 --crop-size 320 \
    --resume experiments/segmentation/runs/pcontext/vggnet/vggnet_pcontext/model_best.pth.tar --split val --mode testval --ms