from modeling.backbone import resnet, xception, drn, mobilenet

# don't check CERTIFICATE, this cause issue on ITSC services
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def build_backbone(backbone, output_stride, BatchNorm,pretrained=True):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm,pretrained=pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
