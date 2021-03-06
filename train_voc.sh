#!/usr/bin/env bash


#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone resnet --lr 0.007 --workers 4 --use-sbd True --epochs 50 --batch-size 16 --gpu-ids 0,1,2,3 --checkname deeplab-resnet --eval-interval 1 --dataset pascal


# copy tar to pth file. It is strange that the tar file is not a compression file
# so just rename it.
# saved model is *pth.tar

#cp /home/hlc/Data/deeplab/v3+/pytorch-deeplab-xception/deeplab-resnet.pth.tar deeplab-resnet.pth.tar


# test training on Cryo06
export CUDA_VISIBLE_DEVICES=1
python train.py --backbone resnet --lr 0.007 --workers 4 \
--use-sbd --epochs 50 --batch-size 8 --gpu-ids 0 \
--checkname deeplab-resnet --eval-interval 1 --dataset pascal


# --use-sbd False