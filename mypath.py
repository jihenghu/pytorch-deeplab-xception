class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            # return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
            return '/home/hlc/Data/experiment/darknet_expr/VOCdevkit/VOC2012'  # VOC2012 on Cryo06
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
