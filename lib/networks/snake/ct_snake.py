import torch.nn as nn
import torch.nn.functional as F
from .dla import DLASeg
from .evolve import Evolution
from .ICD import RAFT
from lib.utils import net_utils, data_utils
from lib.utils.snake import snake_decode
from lib.networks.snake.snake import Thickness_Classifier
import torch
# from .hybrid.models.hybrid_res_unet import ResUNet18_5 as ResUNet
from .hybrid.models.hybrid_res_unet import ResUNet18 as ResUNet
from .hybrid.models.unext import Unext_1,Unext_heads,Unext_2d
from .hybrid.models.unext import Unext_double
from .hybrid.models.unext import Unext_double_64
from .hybrid.models.unext import Unext_Branch_64
from .hybrid.models.unext import Unext_cat
from .hybrid.models.unext3d import  Unext_3d
from .hybrid.models.unext3d import  Unext_3d_hybrid_unext
from .hybrid.models.unext3d import  Unext_3d_hybrid_unext_add
from .hybrid.models.unext3d import  Unext_3d_hybrid_unext_cat
from .hybrid.models.attentiionConvnet import Attentionconvnet
from .hybrid.models.attentiionConvnet import MIAunext,MIAunext_test,Hybridresunet_test,MIAHybridresreunet_test,MIAHybrid
from .hybrid.models.PAUnext import PAUnext
# from .hybrid.models.hybrid_res_unet import ResUNet18_64out as ResUNet
from .hybrid.models.hybrid_res_unet import ResUNet18_tiny as ResUNet_ct
from .hybrid.models.hybrid_res_unet import ResUNet18_Cls as ResUNet_cls
from .hybrid.models.hybrid_res_unet import ResUNet18_3Cls as ResUNet_3cls
from .hybrid.models.hybrid_res_unet import SkipResUNet18_3Cls as SkipResUNet_3cls
# from .hybrid.models.res_unet import ResUNet_cls3d as ResUNet_3cls
from .hybrid.models.hybrid_res_unet import ResUNet18_block as ResUNet_block
from .hybrid.models.hybrid_res_unet import ResUNet_block_32_3 as ResUNet_block_3out
from .hybrid.models.hybrid_res_unet import ResUNet_block_32_5 as ResUNet_block_5out
# from .hybrid.models.hybrid_res_unet import ResUNet18_Double_Skip as ResUNet_2
from .hybrid.models.hybrid_res_unet import ResUNet18_Double_Skip as ResUNet_2
from .hybrid.models.hybrid_res_unet import Time_Series as Time_Series
from .hybrid.models.hybrid_res_unet import ResTime_Series as ResTime_Series
# from .hybrid.models.hybrid_res_unet import ResUNet18_Double as ResUNet_2
from .hybrid.models.transunet import TransUnet
import json
import sys
import numpy as np

from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import cycle
import os
from termcolor import colored
import SimpleITK as sitk

mean = snake_config.mean
std = snake_config.std
import nibabel as nib
import pandas as pd
import numpy as np
import scipy
from scipy import ndimage
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import skimage
from skimage import feature
from scipy import spatial
import ipdb

class Network(nn.Module):

    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()
        # self.Timeseries = Time_Series()
        # self.Timeseries = ResTime_Series()
        # self.transunet = TransUnet()
        # self.hybrid = ResUNet()# hybrid Resunet

        # self.unext_double = Unext_double()# unext-double
        # self.unext_double_64 = Unext_double_64() # unext-double-64
        # self.unext_branch_64 = Unext_Branch_64() # unext-branch-64
        # self.hybrid_double = ResUNet_2() # hybrid Resunet
        # self.unext = Unext_1()# unext
        # self.unext_2d = Unext_2d()# unext-2d
        self.unext_heads = Unext_heads()# unext-multiheads
        # self.unext_cat = Unext_cat()# unext-cat
        # self.unext_3d = Unext_3d()# unext-3d
        # self.unext_3d_hybrid = Unext_3d_hybrid_unext()# unext-3d-hybrid
        # self.unext_3d_hybrid_add = Unext_3d_hybrid_unext_add()# unext-3d-hybrid-add
        # self.unext_3d_hybrid_cat = Unext_3d_hybrid_unext_cat()# unext-3d-hybrid-cat
        # self.attentionconvnet = Attentionconvnet()# attentionconvnet
        # self.MIAunext = MIAunext()# imbalanceawareunext
        # self.MIAunext_test = MIAunext_test()# imbalanceawareunext
        # self.Hybridtest = Hybridresunet_test()
        # self.MIAhybridtest = MIAHybridresreunet_test()
        # self.MIAHybrid = MIAHybrid()
        # self.PAUnext = PAUnext()


        # self.hybrid_1 = ResUNet_2() # physiology-aware
        # self.hybrid_2 = ResUNet_2() # physiology-aware
        # # # self.hybrid_1 = ResUNet()
        # # # self.hybrid_2 = ResUNet()
        # # self.hybrid_cls = ResUNet_cls()
        # self.hybrid_cls = ResUNet_3cls()# physiology-aware
        # # self.hybrid_cls = SkipResUNet_3cls()# physiology-aware
        # self.hybrid_block = ResUNet_block_3out()# physiology-aware
        # self.hybrid_block_seg = ResUNet_block_5out()# physiology-aware
        # self.hybrid_block_seg = ResUNet_block_seg_2out()



        # self.dla = DLASeg('dla{}'.format(num_layers), heads,
        #                   pretrained=False,
        #                   down_ratio=down_ratio,
        #                   final_kernel=1,
        #                   last_level=5,
        #                   head_conv=head_conv)
        # self.gcn = Evolution()
        self.gcn_1 = RAFT()
        # self.thickness = Thickness_Classifier(state_dim=128)

    def decode_detection(self, output, h, w):
        ct_hm = output['ct_hm']
        wh = output['wh']
        ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return ct, detection


    # physiology-aware boundary-detection  polysnake
    def forward(self, x, batch=None):

        output_f = {}
        x = torch.tensor(x, dtype=torch.float32)


        x = torch.unsqueeze(x, 1)

        output,cnn_feature,output_wh = self.unext_heads(x)

        output_f["prob_map_boundary"] = output
        output_f["wh"] = output_wh



        output_f = self.gcn_1(output_f, cnn_feature, batch)

        return output_f

    




def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network
