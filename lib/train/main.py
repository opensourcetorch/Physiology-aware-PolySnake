# _*_ coding: utf-8 _*_

""" main function """

from __future__ import print_function

import sys
sys.path.append("..")

import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import argparse
import shutil
from loss import WeightedCrossEntropy, FocalLoss, DiceLoss, WeightedHausdorffDistanceDoubleBoundLoss
import os.path as osp
from train import train_model, model_reference
from torchvision import transforms
from lr_scheduler import PolyLR

from torch.optim import lr_scheduler

import matplotlib as mpl
mpl.use('Agg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, help="from where to read data")
    parser.add_argument('--central_crop', type=int, default=192)
    parser.add_argument('--rescale', type=int, default=96)
    parser.add_argument('--output_channel', type=int, default=5, choices=(2, 3, 4, 5))
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--w_decay', type=float, default=0.005)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--criterion', type=str, default='nll')
    parser.add_argument('--opt', type=str, default='Adam', help="optimizer")
    parser.add_argument('--weight', type=lambda x: True if x.lower()=='true' else None, default=True)
    parser.add_argument('--weight_type', type=lambda x: None if x.lower()=='none' else x, default=None)
    parser.add_argument('--only_test', type=lambda x: x.lower()=='true')
    parser.add_argument('--rotation', type=lambda x: x.lower()=='true')
    parser.add_argument('--flip', type=lambda x: x.lower()=='true')
    parser.add_argument('--r_central_crop', type=lambda x: x.lower()=='true')
    parser.add_argument('--random_trans', type=lambda x: x.lower()=='true')
    parser.add_argument('--noise', type= lambda x: x.lower()=='true', help="whether add Gaussian noise or not")
    parser.add_argument('--use_pre_train', type=lambda x: x.lower()=='true')
    parser.add_argument('--fig_dir', type=str, help="directory for saving segmentation results")
    parser.add_argument('--pre_train_path', type=str)
    parser.add_argument('--with_shallow_net', type= lambda x: x.lower()=='true')
    parser.add_argument('--n_epoch_hardmining', type=int, default=15, help="every how many epochs for hard mining")
    parser.add_argument('--percentile', type=int, default=85, help="how much percent samples to save for hard mining")
    parser.add_argument('--plot_data', type=str, default='test', help="what data to plot")
    parser.add_argument('--do_plot', type=lambda x: x.lower()=='true', help="whether plot test results or not")
    parser.add_argument('--multi_view', type=lambda x: x.lower()=='true', help="whether to use multi-view inputs")
    parser.add_argument('--model', type=str, choices=('tiramisu', 'unet', 'res_unet', 'hyper_tiramisu', 'deeplab_resnet',
                                                      'res_unet_dp', 'res_unet_reg'), help="which model to use")
    parser.add_argument('--theta', type=float, help="compression ratio for DenseNet")
    parser.add_argument('--interval', type=int, help="interval of slices in volume")
    parser.add_argument('--down_sample', type=int, default=1, help="down sampling step")
    parser.add_argument('--model_type', type=str, default='2d', help="use 2D or 3D model")
    parser.add_argument('--config', type=str, default='config', help="config file name for train/val/test data split")
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help="learning scheduler")
    parser.add_argument('--cal_zerogt', type= lambda x: x.lower() == 'true', default=False,
                        help= "whether calculate F1 score for case of all GT pixels are zero")
    parser.add_argument('--drop_out', type=float, default=0.0,
                        help= "drop out rate for Res-UNet model")
    parser.add_argument('--ignore_index', type=lambda x: None if x.lower()=='none' else int(x),
                        help= "ignore index")
    parser.add_argument('--bound_out', type=lambda x: x.lower()=='true', default=False,
                        help="whether output with bound")
    parser.add_argument('--whd_alpha', default=4, type=int, help="alpha in WHD loss")
    parser.add_argument('--whd_beta', default=1, type=float, help="beta in WHD loss")
    parser.add_argument('--whd_ratio', default=0.5, type=float, help="ratio in WHD loss")
    parser.add_argument('--avg_pool', default=False, type= lambda x: x.lower() == 'true',
                        help="whether use average pooling when calculating WHD loss")
    parser.add_argument('--sample_stack_rows', default=50, type=int,
                        help="how many samples to save per file")

    args = parser.parse_args()
    shutil.copy('./main.sh', './{}'.format(args.fig_dir)) # save current bash file for replicating experiment results

    args.model_save_name = "./{}/model.pth".format(args.fig_dir)

    # transforms and augmentations
    if args.model_type == '2d':
        from image.transforms import Gray2InnerOuterBound
        from image.transforms import CentralCrop, Rescale, Gray2Triple, Gray2Mask, ToTensor, Identical, HU2Gray, RandomFlip
        from image.transforms import RandomTranslate, RandomCentralCrop, AddNoise, RandomRotation
    else:  # 2.5D and 3D
        from volume.transforms import CentralCrop, Rescale, Gray2Mask, ToTensor, Identical, HU2Gray, RandomFlip, Gray2Triple
        from volume.transforms import RandomTranslate, RandomCentralCrop, AddNoise, RandomRotation, Gray2InnerOuterBound

    # choose transforms of annotation under different settings
    if args.output_channel == 3: # 2 options: (1) triple class seg (2) inner + outer boundary detection
        if args.bound_out:
            ToMask = Gray2InnerOuterBound()
        else:
            ToMask = Gray2Triple()

    elif args.output_channel == 5 and not args.bound_out:
        ToMask = Gray2Mask()

    args.compose = {'train': transforms.Compose([HU2Gray(),
                                                 RandomRotation() if args.rotation else Identical(),
                                                 RandomFlip() if args.flip else Identical(),
                                                 RandomCentralCrop() if args.r_central_crop else CentralCrop(args.central_crop),
                                                 Rescale((args.rescale)),
                                                 RandomTranslate() if args.random_trans else Identical(),
                                                 AddNoise() if args.noise else Identical(),
                                                 ToMask,
                                                 ToTensor(norm=True)]),

                    'test': transforms.Compose([HU2Gray(),
                                                CentralCrop(args.central_crop),
                                                Rescale(args.rescale),
                                                ToMask,
                                                ToTensor(norm=True)])}

    # whether use pre_train model or not
    if args.use_pre_train:
        model = torch.load("{}/model.pth".format(args.pre_train_path),
                           map_location=lambda storage, loc: storage)
    else:
        args.color_channel = 3 if args.multi_view else 1

        if args.model_type == '2d':
            if args.model == 'unet':
                if args.with_shallow_net:
                    from image.models.unet import UNet18 as UNet
                else:
                    from image.models.unet import UNet28 as UNet
                model = UNet(args.color_channel, args.output_channel)

            elif args.model == 'res_unet':
                print("res_unet is called")
                if args.with_shallow_net:
                    from image.models.res_unet import ResUNet18 as ResUNet
                else:
                    from image.models.res_unet import ResUNet28 as ResUNet
                model = ResUNet(args.color_channel, args.output_channel, args.drop_out)

            elif args.model == 'tiramisu':
                if args.with_shallow_net:
                    from image.models.tiramisu import FCDenseNet43 as FCDenseNet
                else:
                    from image.models.tiramisu import FCDenseNet67 as FCDenseNet
                model = FCDenseNet(args.color_channel, args.output_channel, args.theta)

        elif args.model_type == '3d':
            if args.model == 'unet':
                if args.with_shallow_net:
                    from lib.networks.snake.hybrid.models.unet import UNet18 as UNet
                else:
                    from lib.networks.snake.hybrid.models.unet import UNet28 as UNet

                model = UNet(args.color_channel, args.output_channel)

            elif args.model == 'res_unet': # for 3D network, Res-UNet and Res-UNet with dropout is not distinguished
                if args.with_shallow_net:
                    from volume.models.res_unet import ResUNet18 as ResUNet
                else:
                    from volume.models.res_unet import ResUNet28 as ResUNet
                model = ResUNet(args.color_channel, args.output_channel, args.drop_out)

            elif args.model == 'tiramisu':
                if args.with_shallow_net:
                    from volume.models.tiramisu import FCDenseNet43 as FCDenseNet
                else:
                    from volume.models.tiramisu import FCDenseNet67 as FCDenseNet
                model = FCDenseNet(args.color_channel, args.output_channel, args.theta)

        elif args.model_type == "2.5d": # Hybrid model with 3D input and 2D output
            if args.model == 'res_unet':
                if args.with_shallow_net:
                    from hybrid.models.hybrid_res_unet import ResUNet18 as ResUNet # 15 slices
                else:
                    from hybrid.models.hybrid_res_unet import ResUNet23 as ResUNet # 31 slices
                model = ResUNet(args.color_channel, args.output_channel, args.interval, args.rescale)

    # whether use gpu or not
    if args.use_gpu:
        model = model.cuda()

    # whether introduce prior weight into loss function or not
    if args.weight:
        if args.weight_type is None:
            if args.bound_out:
                weight = torch.from_numpy(np.load('../class_weights/nlf_weight_all_bound_{}.npy'.format(args.output_channel))).float()
            else:
                if args.output_channel == 5:
                    weight = torch.from_numpy(np.load('../class_weights/class_weight.npy')).float()
                else:
                    weight = torch.from_numpy(np.load('../class_weights/nlf_weight_all_{}.npy'.format(args.output_channel))).float()

        weight = Variable(weight.cuda())

    else: # no prior weight, especially for bound detection
        weight = None
    print("weight: {}".format(weight))

    # criterion
    if args.criterion == 'nll':
        criterion = nn.NLLLoss(weight=weight)
    elif args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif args.criterion == 'dice':
        criterion = DiceLoss(weight=weight, ignore_index=None, weight_type=args.weight_type, cal_zerogt=args.cal_zerogt)
    elif args.criterion == 'focal':
        criterion = FocalLoss()
    elif args.criterion == 'wce':
        criterion = WeightedCrossEntropy()
    elif args.criterion == 'whddb': # whd with double bounds
        criterion = WeightedHausdorffDistanceDoubleBoundLoss(return_boundwise_loss=False, alpha=args.whd_alpha,
                                 beta=args.whd_beta, ratio=args.whd_ratio)
    # optimizer
    if args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)

    # learning schedule
    if args.lr_scheduler == 'StepLR':
        my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'PolyLR':
        my_lr_scheduler = PolyLR(optimizer, max_iter=args.num_train_epochs, power=0.9)

    # print arguments setting
    for arg in vars(args):
        print("{} : {}".format(arg, getattr(args, arg)))

    # plot samples used for train, val and test respectively
    print("Dataset:")
    for mode in ['train', 'val', 'test']:
        config_file = osp.join('../configs/{}'.format(args.config), mode+'2.txt')
        print(mode)
        with open(config_file, 'r') as reader:
            for line in reader.readlines():
                print(line.strip('\n'))

    since = time.time()
    if not args.only_test:
        train_model(model, criterion, optimizer, my_lr_scheduler, args)

    # model reference
    model_reference(args)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))