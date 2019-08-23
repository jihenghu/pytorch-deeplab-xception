#!/usr/bin/env python
# Filename: inference 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 August, 2019
"""

from modeling.deeplab import DeepLab
import torch

import numpy as np

import os
import cv2

import dataloaders.custom_transforms as custom_transforms

def inference_A_sample_image(img_path, model_path,num_classes,backbone,output_stride,sync_bn,freeze_bn):


    # read image
    image = cv2.imread(img_path)

    # print(image.shape)
    image = np.array(image).astype(np.float32)
    # Normalize pascal image (mean and std is from pascal.py)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image /=255
    image -= mean
    image /= std

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))

    # to 4D, N=1
    image = image.reshape(1, image.shape[0],image.shape[1],image.shape[2])
    image = torch.from_numpy(image) #.float()


    model = DeepLab(num_classes=num_classes,
                    backbone=backbone,
                    output_stride=output_stride,
                    sync_bn=sync_bn,
                    freeze_bn=freeze_bn,
                    pretrained=True) # False

    if torch.cuda.is_available() is False:
        device = torch.device('cpu')
    else:
        device = None # need added

    # checkpoint = torch.load(model_path,map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])
    checkpoint = torch.load('resnet101-5d3b4d8f.pth', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # for set dropout and batch normalization layers to evaluation mode before running inference.
    #  Failing to do this will yield inconsistent inference results.
    model.eval()

    with torch.no_grad():
        output = model(image)

        out_np = output.cpu().data.numpy()

        pred = np.argmax(out_np, axis=1)

        pred = pred.reshape(pred.shape[1],pred.shape[2])

        # save result
        cv2.imwrite('output.jpg',pred)

        test = 1



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Inference")
    parser.add_argument('--img_path', type=str,
                        help='the path to an image (not big image like a remote sensing image)')

    parser.add_argument('--model_path', type=str,default='model_best.pth.tar', #deeplab-resnet.pth.tar
                        help='the path to an image')

    args = parser.parse_args()

    inference_A_sample_image('2009_001138.jpg',args.model_path,21,'resnet',16,False,False)

    # inference_A_sample_image('2011_002730.jpg', args.model_path, 21, 'resnet', 16, False, False)

    # inference_A_sample_image('2010_004365.jpg', args.model_path, 21, 'resnet', 16, False, False)





