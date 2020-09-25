# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_scratch --dataset pcontext \
    --model vggnet --base-size 256 --crop-size 256 \
    --checkname vggnet80_pcontext --epochs 80

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model vggnet --base-size 256 --crop-size 256 \
    --resume experiments/segmentation/runs/pcontext/vggnet/vggnet80_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model vggnet --base-size 256 --crop-size 256 \
    --resume experiments/segmentation/runs/pcontext/vggnet/vggnet80_pcontext/model_best.pth.tar --split val --mode testval --ms