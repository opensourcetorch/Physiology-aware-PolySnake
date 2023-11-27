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
import os
import matplotlib.pyplot as plt
from utils import *
# from utils_3d import *
__all__ = ['UNext']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_,to_3tuple
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
import ipdb
from .unext import shiftedBlock,OverlapPatchEmbed


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
    x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, self.pad, H)
    x_cat = torch.narrow(x_cat, 3, self.pad, W)
    return x_cat

class shiftmlp3d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features[0])
        self.dwconv1 = DWConv3d(hidden_features[0])
        self.fc2 = nn.Linear(hidden_features[0], hidden_features[1])
        self.dwconv2 = DWConv3d(hidden_features[1])
        self.act = act_layer()
        self.fc3 = nn.Linear(hidden_features[1], out_features)

        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x,D, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, D, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad,self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)

        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        # ipdb.set_trace()
        x_cat = torch.cat(x_shift, 1)

        x_cat = torch.narrow(x_cat, 2, self.pad, D)
        x_cat = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_cat, 4, self.pad, W)


        x_s = x_s.reshape(B,C,D*H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv1(x, D,H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, D,H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad,self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, D)
        x_cat = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_cat, 4, self.pad, W)
        x_s = x_s.reshape(B,C,D*H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.dwconv2(x, D,H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, D,H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad,self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 4) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, D)
        x_cat = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_cat, 4, self.pad, W)
        x_s = x_s.reshape(B,C,D*H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc3(x_shift_c)
        x= self.drop(x)
        # ipdb.set_trace()
        return x

class shiftedBlock3d(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = [int(dim * mlp_ratio),int(dim * mlp_ratio)]
        self.mlp = shiftmlp3d(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1]* m.kernel_size[2]* m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x,D, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x),D, H, W))
        return x



class DWConv3d(nn.Module):
    def __init__(self, dim=768):
        super(DWConv3d, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, D,H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C,D, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed3d(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_3tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.D,self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]
        self.num_patches = self.D * self.H * self.W
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(0, patch_size[1] // 2, patch_size[2] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, D,H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # ipdb.set_trace()

        return x,D, H, W


def conv_333(in_channels, out_channels, stride=1, padding=1):
    # here only the X and Y directions are padded and no padding along Z direction
    # in this way, we can make sure the central slice of the input volume will remain central
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, bias=True)
def conv_111(in_channels, out_channels, stride=1):
    # since BN is used, bias is not necessary
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=False)
class ResBlock3D(nn.Module):
    """ residual block """
    def __init__(self, in_channels, out_channels, stride=1, p=0.5, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(in_channels)
        padding = 1
        self.conv1 = conv_333(in_channels, out_channels, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = conv_333(out_channels, out_channels, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout3d(p=p)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=3, stride=stride, bias=False, padding=padding),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        # print("input residual size: {}".format(residual.size()))
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dp(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
            # print("output residual size: {}".format(residual.size()))
        # print("output size: {}".format(out.size()))
        out += residual
        # ipdb.set_trace()

        return out
class ResBlock3D_111(nn.Module):
    """ residual block """
    def __init__(self, in_channels, out_channels, stride=1, p=0.5, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = conv_111(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = conv_111(out_channels, out_channels, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout3d(p=p)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False, padding=0),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        # print("input residual size: {}".format(residual.size()))
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dp(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
            # print("output residual size: {}".format(residual.size()))
        # print("output size: {}".format(out.size()))
        out += residual
        # ipdb.set_trace()

        return out
class UNext3d(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
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
        self.encoder1 = ResBlock3D(input_channels, 64, stride=[1,2,2], p=0.5)
        self.encoder2 = ResBlock3D(64, 128, stride=[1,2,2], p=0.5)
        self.encoder3 = ResBlock3D(128, 256, stride=[1,2,2], p=0.5)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(384)
        self.dnorm4 = norm_layer(256)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock3d(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock3d(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock3d(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock3d(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed3d(img_size= [size // factor for size, factor in zip(img_size, [1, 4, 4])], patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed3d(img_size= [size // factor for size, factor in zip(img_size, [2, 8, 8])], patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv3d(512, 384, 3, stride=1,padding=1)
        self.decoder2 =   nn.Conv3d(384, 256, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv3d(256, 128, 3, stride=1, padding=1)
        self.decoder4 =   nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv3d(64, 64, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm3d(384)
        self.dbn2 = nn.BatchNorm3d(256)
        self.dbn3 = nn.BatchNorm3d(128)
        self.dbn4 = nn.BatchNorm3d(64)

        self.final = nn.Conv3d(64, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        out = self.encoder1(x)
        #get middle tensor
        t1 = out

        ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out = self.encoder2(out)
        t2 = out
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out = self.encoder3(out)
        t3 = out

        # ipdb.set_trace()


        ### Tokenized MLP Stage
        ### Stage 4

        out,D,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out,D, H, W)
        out = self.norm3(out)
        out = out.reshape(B,D, H, W, -1).permute(0, 4, 1, 2,3).contiguous()
        t4 = out
        # ipdb.set_trace()

        ### Bottleneck

        out ,D,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out,D, H, W)
        out = self.norm4(out)
        out = out.reshape(B, D,H, W, -1).permute(0,4, 1, 2,3).contiguous()


        ### Stage 4
        out = self.decoder1(out)
        # ipdb.set_trace()

        out = F.relu(F.interpolate(self.dbn1(out),size=[out.shape[-3]*2+1,out.shape[-2]*2,out.shape[-1]*2],mode ='trilinear'))


        out = torch.add(out,t4)
        # ipdb.set_trace()
        # print("stage4:",out.shape)
        _,_,D,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, D,H, W)



        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B,D, H, W, -1).permute(0, 4, 1, 2,3).contiguous()
        out = self.decoder2(out)
        out = F.relu(F.interpolate(self.dbn2(out),size=[out.shape[-3]*2+1,out.shape[-2]*2,out.shape[-1]*2],mode ='trilinear'))
        out = torch.add(out,t3)
        _,_,D,H,W = out.shape
        out = out.flatten(2).transpose(1,2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, D,H, W)

        out = self.dnorm4(out)
        out = out.reshape(B,D, H, W, -1).permute(0, 4, 1, 2,3).contiguous()

        out = self.decoder3(out)
        out = F.relu(F.interpolate(self.dbn3(out),size=[out.shape[-3],out.shape[-2]*2,out.shape[-1]*2],mode ='trilinear'))
        out = torch.add(out,t2)
        out = self.decoder4(out)
        out = F.relu(F.interpolate(self.dbn4(out),size=[out.shape[-3],out.shape[-2]*2,out.shape[-1]*2],mode ='trilinear'))
        out = torch.add(out,t1)
        out = self.decoder5(out)
        out = F.relu(F.interpolate(out,size=[out.shape[-3],out.shape[-2]*2,out.shape[-1]*2],mode ='trilinear'))
        # ipdb.set_trace()

        return self.final(out),out

class UNext3d_Hybrid_Unext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.unext3d=UNext3d(num_classes=5, input_channels=1, deep_supervision=False,img_size=[15,96,96], patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                             num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                             attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                             depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
        self.encoder1 = ResBlock3D(input_channels+64, 64, stride=2, p=0.5)
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

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        out_3d,feature_3d = self.unext3d(x)
        x=torch.cat([x,feature_3d],dim=1)

        B = x.shape[0]
        # ipdb.set_trace()
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

        return self.final(out),out,out_3d

class UNext3d_Hybrid_Unext_add(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.unext3d=UNext3d(num_classes=5, input_channels=1, deep_supervision=False,img_size=[15,96,96], patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                             num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                             attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                             depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
        self.encoder_seg = ResBlock3D_111(64, 1, stride=1, p=0.5)
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

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        out_3d,feature_3d = self.unext3d(x)
        feature_3d = self.encoder_seg(feature_3d)
        # ipdb.set_trace()
        x=torch.add(x,feature_3d)

        B = x.shape[0]
        # ipdb.set_trace()
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

        return self.final(out),out,out_3d
class UNext3d_Hybrid_Unext_cat(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.unext3d=UNext3d(num_classes=5, input_channels=1, deep_supervision=False,img_size=[15,96,96], patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                             num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                             attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                             depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
        self.encoder_seg = ResBlock3D_111(64, 1, stride=1, p=0.5)
        self.encoder1 = ResBlock3D(input_channels+1, 64, stride=2, p=0.5)
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

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        out_3d,feature_3d = self.unext3d(x)
        feature_3d = self.encoder_seg(feature_3d)
        # ipdb.set_trace()
        out=torch.cat((x,feature_3d),dim=1)

        B = x.shape[0]
        # ipdb.set_trace()
        ### Encoder
        ### Conv Stage

        ### Stage 1
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        out = self.encoder1(out)
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

        return self.final(out),out,out_3d

def Unext_3d():
    return UNext3d(num_classes=3, input_channels=1, deep_supervision=False,img_size=[15,96,96], patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                          num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                          depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])

def Unext_3d_hybrid_unext():

    return UNext3d_Hybrid_Unext(num_classes=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                       num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                       attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                       depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
def Unext_3d_hybrid_unext_add():

    return UNext3d_Hybrid_Unext_add(num_classes=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                                num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                                attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                                depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
def Unext_3d_hybrid_unext_cat():

    return UNext3d_Hybrid_Unext_cat(num_classes=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                                    num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                                    depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])