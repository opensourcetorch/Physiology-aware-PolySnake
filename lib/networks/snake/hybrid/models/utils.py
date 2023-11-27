# _*_ coding: utf-8 _*_


import math
import torch
torch.manual_seed(42) # make random weight fixed for every running
import argparse
import torch.nn as nn

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count_parameters(model):
    """ count number of parameters """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _initialize_weights_3d(model):
    """ model weight initialization """
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            m.weight.data.normal_(0, 0.01)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def _initialize_weights_2d(model):
    """ model weight initialization """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
