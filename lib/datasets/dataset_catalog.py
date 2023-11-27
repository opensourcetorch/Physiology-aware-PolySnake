from lib.config import cfg
from volume.transforms import CentralCrop, Rescale, Gray2Mask, ToTensor, Identical, HU2Gray, RandomFlip, Gray2Triple
from volume.transforms import RandomTranslate, RandomCentralCrop, AddNoise, RandomRotation, Gray2InnerOuterBound



class DatasetCatalog(object):

    dataset_attrs = {
        'CocoTrain': {
            'id': 'coco',
            'data_root': 'data/coco/train2017',
            'ann_file': 'data/coco/annotations/instances_train2017.json',
            'split': 'train'
        },
        'CocoVal': {
            'id': 'coco',
            'data_root': 'data/coco/val2017',
            'ann_file': 'data/coco/annotations/instances_val2017.json',
            'split': 'test'
        },
        'CocoMini': {
            'id': 'coco',
            'data_root': 'data/coco/val2017',
            'ann_file': 'data/coco/annotations/instances_val2017.json',
            'split': 'mini'
        },
        'CocoTest': {
            'id': 'coco_test',
            'data_root': 'data/coco/test2017',
            'ann_file': 'data/coco/annotations/image_info_test-dev2017.json',
            'split': 'test'
        },
        'CityscapesTrain': {
            'id': 'cityscapes',
            'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': ('data/cityscapes/annotations/train', 'data/cityscapes/annotations/train_val'),
            'split': 'train'
        },
        'CityscapesVal': {
            'id': 'cityscapes',
            'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': 'data/cityscapes/annotations/val',
            'split': 'val'
        },
        'CityscapesCocoVal': {
            'id': 'cityscapes_coco',
            'data_root': 'data/cityscapes/leftImg8bit/val',
            'ann_file': 'data/cityscapes/coco_ann/instance_val.json',
            'split': 'val'
        },
        'CityCocoBox': {
            'id': 'cityscapes_coco',
            'data_root': 'data/cityscapes/leftImg8bit/val',
            'ann_file': 'data/cityscapes/coco_ann/instance_box_val.json',
            'split': 'val'
        },
        'CityscapesMini': {
            'id': 'cityscapes',
            'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': 'data/cityscapes/annotations/val',
            'split': 'mini'
        },
        'CityscapesTest': {
            'id': 'cityscapes_test',
            'data_root': 'data/cityscapes/leftImg8bit/test'
        },
        'SbdTrain': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_train_instance.json',
            'split': 'train'
        },
        'SbdVal': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'val'
        },
        'SbdMini': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'mini'
        },
        'VocVal': {
            'id': 'voc',
            'data_root': 'data/voc/JPEGImages',
            'ann_file': 'data/voc/annotations/voc_val_instance.json',
            'split': 'val'
        },
        'KinsTrain': {
            'id': 'kins',
            'data_root': '/data/training/image_2',
            'ann_file': 'data/training/instances_train.json',
            'split': 'train'
        },
        'KinsVal': {
            'id': 'kins',
            'data_root': 'data/kitti/testing/image_2',
            'ann_file': 'data/kitti/testing/instances_val.json',
            'split': 'val'
        },
        'KinsMini': {
            'id': 'kins',
            'data_root': 'data/testing/image_2',
            'ann_file': 'data/testing/instances_val.json',
            'split': 'mini'
        },
        'MedicalTrain': {
            'id': 'medical',
            'data_root': '/data/ugui0/antonio-t/CPR_multiview_interp2_huang',
            'split':'train',
            'ann_file': '/data/ugui0/antonio-t/CPR_multiview_interp2_huang/annotations/medical_train_instance_9010.json',





        },

        'MedicalTest': {
            'id': 'medical',
            'data_root': '/data/ugui0/antonio-t/CPR_multiview_interp2_huang',
            'split': 'test',
            'ann_file': '/data/ugui0/antonio-t/CPR_multiview_interp2_huang/annotations/medical_val_instance_demo.json',


        }
    }
    #  ann_file, data_root,split,data_dir, metric_prev_epoch = None, phases_prev_epoch = None, transform = None, mode = 'train',
    #                 is_hard_mining = False, percentile = 85, multi_view = False, interval=32, down_sample=1, config='config'
    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()

