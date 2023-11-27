from .yacs import CfgNode as CN
import argparse
import os
from torchvision import transforms
from volume.transforms import CentralCrop, Rescale, Gray2Mask, ToTensor, Identical, HU2Gray, RandomFlip, Gray2Triple
from volume.transforms import RandomTranslate, RandomCentralCrop, AddNoise, RandomRotation, Gray2InnerOuterBound

cfg = CN()

# model
cfg.model = 'hello'
cfg.model_dir = 'data/model'
cfg.pretrain = 'classifier'
cfg.pretrain_dir = 'data/model'
cfg.im_kernel_ratio = 0.5
cfg.cpr_dir = 'data/model'
cfg.dis = 68
# network
cfg.network = 'dla_34'

# network heads
cfg.heads = CN()

# task
cfg.task = 'snake'

# gpus
cfg.gpus = [0]

# if load the pretrained network
cfg.resume = True

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()
ToMask = Gray2Mask()

cfg.train.dataset = 'CocoTrain'
cfg.train.dataset_dir = "/data/ugui0/antonio-t/CPR_multiview_interp2_huang"
cfg.train.epoch = 140
cfg.train.num_workers = 8

# use adam as default
cfg.train.optim = 'sgd'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.scheduler = ''
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.state='test'

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False

cfg.save_ep = 5
cfg.eval_ep = 5

# -----------------------------------------------------------------------------
# snake
# -----------------------------------------------------------------------------
cfg.ct_score = 0.05
cfg.demo_path = ''


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256



    cfg.pretrain_dir = os.path.join(cfg.model_dir, cfg.task, cfg.pretrain)

    cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.model)

    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.percentile = args.percentile
    cfg.multi_view = args.multi_view
    cfg.interval = args.interval
    cfg.down_sample = args.down_sample
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.config = args.config
    cfg.rotation=args.rotation
    cfg.flip=args.flip
    cfg.r_central_crop=args.r_central_crop
    cfg.central_crop=args.central_crop
    cfg.rescale=args.rescale
    cfg.random_trans=args.random_trans
    cfg.noise=args.noise
    # cfg.compose = {'train': transforms.Compose([HU2Gray(),
    #                                             RandomRotation() if args.rotation else Identical(),
    #                                             RandomFlip() if args.flip else Identical(),
    #                                             RandomCentralCrop() if args.r_central_crop else CentralCrop(
    #                                                 args.central_crop),
    #                                             Rescale((args.rescale)),
    #                                             RandomTranslate() if args.random_trans else Identical(),
    #                                             AddNoise() if args.noise else Identical(),
    #                                             ToMask
    #                                             # ToTensor(norm=True)
    #                                             ]),
    #
    #                'test': transforms.Compose([HU2Gray(),
    #                                            CentralCrop(args.central_crop),
    #                                            Rescale(args.rescale),
    #                                            ToMask
    #                                            # ToTensor(norm=True)
    #                                            ])}
    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
parser.add_argument('--central_crop', type=int, default=192)
parser.add_argument('--rescale', type=int, default=96)
parser.add_argument('--rotation', type=lambda x: x.lower() == 'true',default=True)
parser.add_argument('--flip', type=lambda x: x.lower() == 'true',default=True)
parser.add_argument('--r_central_crop', type=lambda x: x.lower() == 'true',default=192)
parser.add_argument('--random_trans', type=lambda x: x.lower() == 'true',default=False)
parser.add_argument('--noise', type=lambda x: x.lower() == 'true',default=False, help="whether add Gaussian noise or not")
parser.add_argument('--percentile', type=int, default=85, help="how much percent samples to save for hard mining")
parser.add_argument('--multi_view', type=lambda x: x.lower() == 'true',default=False, help="whether to use multi-view inputs")
parser.add_argument('--interval', type=int,default=15, help="interval of slices in volume")
parser.add_argument('--down_sample', type=int, default=1, help="down sampling step")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--config', type=str, default='config', help="config file name for train/val/test data split")
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
