#!/usr/bin/env bash

# test training on ITSC services
# 23 Aug 2019.

#cp /home/hlc/Data/deeplab/v3+/pytorch-deeplab-xception/deeplab-resnet.pth.tar deeplab-resnet.pth.tar


# test training on ITSC services with 8 GPU
#export CUDA_VISIBLE_DEVICES=1
python train.py --backbone resnet --lr 0.007 --workers 4 \
--use-sbd --epochs 50 --batch-size 64 --gpu-ids 0,1,2,3,4,5,6,7 \
--checkname deeplab-resnet --eval-interval 1 --dataset pascal


#  False