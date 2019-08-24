#!/usr/bin/env bash

# test training on ITSC services
# 23 Aug 2019.

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# use python 3 with pytorch 1.2
export PATH=~/programs/anaconda3/bin:$PATH

deeplab_dir=~/codes/PycharmProjects/yghlc_pytorch-deeplab-xception
para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py


para_file=para.ini

if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

# copy pretrained model
if [ ! -f deeplab-resnet.pth.tar ]; then
   echo "Copy pretrained model"
   cp ~/Data/deeplab/v3+/pytorch-deeplab-xception/deeplab-resnet.pth.tar deeplab-resnet.pth.tar
fi

expr_name=$(python2 ${para_py} -p ${para_file} expr_name)

# test training on ITSC services with 8 GPU
#export CUDA_VISIBLE_DEVICES=1

if [ -f ${expr_name}/checkpoint.pth.tar ]; then

    # resume training
    python ${deeplab_dir}/train.py --backbone resnet --lr 0.007 --workers 4 \
    --use-sbd --epochs 50 --batch-size 32 --gpu-ids 0,1,2,3,4,5,6,7 \
    --checkname deeplab-resnet --eval-interval 1 --dataset pascal \
    --resume ${expr_name}/checkpoint.pth.tar

else
    python ${deeplab_dir}/train.py --backbone resnet --lr 0.007 --workers 4 \
    --use-sbd --epochs 50 --batch-size 32 --gpu-ids 0,1,2,3,4,5,6,7 \
    --checkname deeplab-resnet --eval-interval 1 --dataset pascal

fi


#  False