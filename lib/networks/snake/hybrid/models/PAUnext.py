import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
# from utils_3d import *
__all__ = ['UNext']
from lib.config import cfg, args

import timm
from timm.models.layers import DropPath,to_2tuple, to_3tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
import ipdb
from .unext import ResBlock3D,conv_333,shiftedBlock,OverlapPatchEmbed
from .hybrid_res_unet import ResBlock2D
class UNext_Core(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        # ipdb.set_trace()

        # self.encoder1 = nn.Conv3d(input_channels, 32, 3, stride=1, padding=1)
        # self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        # self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        #
        # self.ebn1 = nn.BatchNorm3d(32)
        # self.ebn2 = nn.BatchNorm3d(64)
        # self.ebn3 = nn.BatchNorm3d(128)
        self.encoder1 = ResBlock3D(input_channels, 64, stride=2, p=0.5)
        self.encoder2 = ResBlock3D(64, 128, stride=2, p=0.5)
        self.encoder3 = ResBlock3D(128, 256, stride=2, p=0.5)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(384)
        self.dnorm4 = norm_layer(256)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(512, 384, 3, stride=1,padding=1)
        self.decoder2 =   nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.decoder4 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(384)
        self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(128)
        self.dbn4 = nn.BatchNorm2d(64)


    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        out = self.encoder1(x)
        #get middle tensor
        idx1 = out.shape[2]//2
        t1 = out[:,:,idx1,:,:]

        ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out = self.encoder2(out)
        idx2 = out.shape[2]//2
        t2 = out[:,:,idx2,:,:]
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out = self.encoder3(out)
        idx3 = out.shape[2]//2
        t3 = out[:,:,idx3,:,:]
        out = t3

        # ipdb.set_trace()


        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))

        out = torch.add(out,t4)
        # print("stage4:",out.shape)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        # ipdb.set_trace()

        return out



class PAUNext(nn.Module):
    def __init__(self,  num_classes_bound=3, num_classes_patch=3, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        # PA-Unext is the abbreviation of physiology-aware Unext that allows network to concerntate on the boundary refination nearby plaque region.
        # PA-Unext is based on Unext model, it contains two parts: 1) the boundary refination module, 2) the plaque-patch-segmentation  module.
        # physiology-aware loss is used to  refine the boundary nearby plaque region based on hausdorff distance loss
        super(PAUNext, self).__init__()
        #
        self.bound_net = UNext_Core( input_channels=input_channels, deep_supervision=deep_supervision,img_size=img_size, patch_size=patch_size, in_chans=in_chans,  embed_dims=embed_dims,
                                 num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                 attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                                 depths=depths, sr_ratios=sr_ratios)
        self.patch_net = UNext_Core(input_channels=input_channels, deep_supervision=deep_supervision,img_size=img_size, patch_size=patch_size, in_chans=in_chans,  embed_dims=embed_dims,
                                 num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                 attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                                 depths=depths, sr_ratios=sr_ratios)
        self.bound_net_final = nn.Conv2d(64, num_classes_bound, kernel_size=1)
        self.patch_net_final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, num_classes_patch, kernel_size=1)
        )


    def forward(self,x):
        out_b = self.bound_net(x) # (B,64,H,W)
        out_bound = self.bound_net_final(out_b)# (B,num_classes_bound,H,W)
        out_bound = F.softmax(out_bound,dim=1)

        out_p = self.patch_net(x) # (B,64,H,W)
        out_patch = self.patch_net_final(out_p) # (B,num_classes_patch,H,W)
        out_patch = F.softmax(out_patch,dim=1)
        # ipdb.set_trace()

        return out_bound,out_b, out_patch

def PAUnext():
    return PAUNext(num_classes_bound=3, num_classes_patch=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
    num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
    depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])