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
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
import ipdb



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
    x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, self.pad, H)
    x_cat = torch.narrow(x_cat, 3, self.pad, W)
    return x_cat

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)


        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        # ipdb.set_trace()
        return x



class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
class ImbalanceAwareshiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # ipdb.set_trace()
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # ipdb.set_trace()

        return x, H, W


class UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        # ipdb.set_trace()

        self.encoder1 = nn.Conv2d(input_channels, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

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

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)
        self.decoder2 =   nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.decoder6 =  nn.Conv2d(16, 64, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)


        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

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
        out = F.relu(self.decoder6(out))
        # ipdb.set_trace()

        return self.final(out),out
def conv_333(in_channels, out_channels, stride=1, padding=1):
    # here only the X and Y directions are padded and no padding along Z direction
    # in this way, we can make sure the central slice of the input volume will remain central
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, bias=True)

class ResBlock3D(nn.Module):
    """ residual block """
    def __init__(self, in_channels, out_channels, stride=1, p=0.5, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(in_channels)
        padding = 1 if stride == 1 else (0, 1, 1)
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

        return out
class UNext_t(nn.Module):

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

        return self.final(out),out


class UNext_multiheads(nn.Module):

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

        self.head_wh_1 = nn.Conv2d(64, 64, kernel_size=2,stride=2)
        self.head_norm_1 = nn.BatchNorm2d(64)
        self.head_wh_2 = nn.Conv2d(64, 512, kernel_size=2,stride=2)

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
        out_wh = self.head_wh_1(out)
        out_wh = self.head_norm_1(out_wh)
        out_wh = self.head_wh_2(out_wh)
        out_wh = out_wh.view(-1,2,256,24,24)

        # ipdb.set_trace()

        return self.final(out),out,out_wh

class ConvSkip(nn.Module):
    """ up convolution """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                            stride=1, padding=1)

    def forward(self, skip, x):
        """ skip is 3D volume and x is 2D slice, central slice of skip is concatenated with x """
        # print("skip shape: {}".format(skip.shape))
        # print("x shape: {}".format(x.shape))
        # print("x:",x.shape)
        # print("skip_slice:",skip_slice.shape)

        out = self.conv(x)
        # print("out:", out.shape)
        # ipdb.set_trace()
        out = torch.cat([skip, out], 1)
        # print("out_cat:", out.shape)
        # x: torch.Size([16, 256, 12, 12])
        # skip_slice: torch.Size([16, 128, 24, 24])
        # out: torch.Size([16, 128, 24, 24])
        # out_cat: torch.Size([16, 256, 24, 24])

        return out

class UNext_t_cat(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP+skip connection cat

    def __init__(self,  num_classes, input_channels=3,encode_dims =[32,64,128], deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[  128,320, 512],decode_dims=[512,320,256,128,64],
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
        self.encoder1 = ResBlock3D(input_channels, encode_dims[0], stride=2, p=0.5)
        self.encoder2 = ResBlock3D(encode_dims[0], encode_dims[1], stride=2, p=0.5)
        self.encoder3 = ResBlock3D(encode_dims[1], encode_dims[2], stride=2, p=0.5)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0]*2)

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
            dim=embed_dims[0]*2, num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(decode_dims[0],decode_dims[1], 3, stride=1,padding=1)
        self.decoder2 =   nn.Conv2d(decode_dims[1], decode_dims[2], 3, stride=1, padding=1)
        self.upconv2 =ConvSkip(decode_dims[2], decode_dims[2]//2)
        self.decoder3 =   nn.Conv2d(decode_dims[2], decode_dims[3], 3, stride=1, padding=1)
        self.upconv3 =ConvSkip(decode_dims[3], decode_dims[3]//2)
        self.decoder4 =   nn.Conv2d(decode_dims[3], decode_dims[4], 3, stride=1, padding=1)
        self.upconv4 =ConvSkip(decode_dims[4], decode_dims[4]//2)
        self.decoder5 =   nn.Conv2d(decode_dims[4], decode_dims[4], 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(decode_dims[1])
        self.dbn2 = nn.BatchNorm2d(decode_dims[2])
        self.dbn3 = nn.BatchNorm2d(decode_dims[3])
        self.dbn4 = nn.BatchNorm2d(decode_dims[4])

        self.final = nn.Conv2d(decode_dims[4], num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

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
        out = self.upconv2(t3,out)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = self.upconv3(t2,out)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = self.upconv4(t1,out)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        # ipdb.set_trace()

        return self.final(out),out
class UNext_Double_t(nn.Module):

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
        self.encoder1 = ResBlock3D(input_channels, 32, stride=2, p=0.5)
        self.encoder2 = ResBlock3D(32, 64, stride=2, p=0.5)
        self.encoder3 = ResBlock3D(64, 128, stride=2, p=0.5)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)
        self.encoder1_1 = ResBlock3D(input_channels, 32, stride=2, p=0.5)
        self.encoder2_1 = ResBlock3D(32, 64, stride=2, p=0.5)
        self.encoder3_1 = ResBlock3D(64, 128, stride=2, p=0.5)

        self.norm3_1 = norm_layer(embed_dims[1])
        self.norm4_1 = norm_layer(embed_dims[2])

        self.dnorm3_1 = norm_layer(160)
        self.dnorm4_1 = norm_layer(128)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_1 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

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



        self.block1_1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_1[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2_1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_1[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1_1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_1[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2_1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_1[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed3_1 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4_1 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)
        self.decoder2 =   nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.decoder1_1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)
        self.decoder2_1 =   nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3_1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder4_1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5_1 =   nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dbn4 = nn.BatchNorm2d(32)
        self.dbn1_1 = nn.BatchNorm2d(160)
        self.dbn2_1 = nn.BatchNorm2d(128)
        self.dbn3_1 = nn.BatchNorm2d(64)
        self.dbn4_1 = nn.BatchNorm2d(32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        self.final_1 = nn.Conv2d(32, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

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

        ### Stage 1_1
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        out_1 = self.encoder1_1(x)
        #get middle tensor
        idx1_1 = out_1.shape[2]//2
        t1_1 = out_1[:,:,idx1_1,:,:]

        ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out = self.encoder2(out)
        idx2 = out.shape[2]//2
        t2 = out[:,:,idx2,:,:]

        ### Stage 2_1
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out_1 = self.encoder2_1(out_1)
        idx2_1 = out_1.shape[2]//2
        t2_1 = out_1[:,:,idx2_1,:,:]
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out = self.encoder3(out_1)
        idx3 = out.shape[2]//2
        t3 = out[:,:,idx3,:,:]
        out = t3
        ### Stage 3_1
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out_1 = self.encoder3_1(out_1)
        idx3_1 = out_1.shape[2]//2
        t3_1 = out_1[:,:,idx3_1,:,:]
        out_1 = t3_1

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
        t4 = out
        # print("stage4:",out.shape)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)


        ### Tokenized MLP Stage_1
        ### Stage 4_1

        out_1,H_1,W_1 = self.patch_embed3(out_1)
        for i, blk in enumerate(self.block1_1):
            out_1 = blk(out_1, H_1, W_1)
        out_1 = self.norm3_1(out_1)
        out_1 = out_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        t4_1 = out_1

        ### Bottleneck_1

        out_1 ,H_1,W_1= self.patch_embed4_1(out_1)
        for i, blk in enumerate(self.block2_1):
            out_1 = blk(out_1, H_1, W_1)
        out_1 = self.norm4_1(out_1)
        out_1 = out_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4_1

        out_1 = F.relu(F.interpolate(self.dbn1_1(self.decoder1_1(out_1)),scale_factor=(2,2),mode ='bilinear'))

        out_1 = torch.add(out_1,t4_1)
        out_1 = torch.add(out_1,t4)

        # print("stage4:",out.shape)
        _,_,H_1,W_1 = out_1.shape
        out_1 = out_1.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1_1):
            out_1 = blk(out_1, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        t3 = out
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        t2 = out
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        t1 = out
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        ### Stage 3_1

        out_1 = self.dnorm3_1(out_1)
        out_1 = out_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        out_1 = F.relu(F.interpolate(self.dbn2_1(self.decoder2_1(out_1)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t3_1)
        out_1 = torch.add(out_1,t3)

        _,_,H_1,W_1 = out_1.shape
        out_1 = out_1.flatten(2).transpose(1,2)

        for i, blk in enumerate(self.dblock2_1):
            out_1 = blk(out_1, H_1, W_1)

        out_1 = self.dnorm4_1(out_1)
        out_1 = out_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()

        out_1 = F.relu(F.interpolate(self.dbn3_1(self.decoder3_1(out_1)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t2_1)
        out_1 = torch.add(out_1,t2)
        out_1 = F.relu(F.interpolate(self.dbn4_1(self.decoder4_1(out_1)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t1_1)
        out_1 = torch.add(out_1,t1)

        out_1 = F.relu(F.interpolate(self.decoder5_1(out_1),scale_factor=(2,2),mode ='bilinear'))
        # ipdb.set_trace()

        return self.final_1(out_1),out_1,self.final(out),out


class UNext_Double_t_64(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 256, 320, 512],
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

        self.dnorm3 = norm_layer(320)
        self.dnorm4 = norm_layer(256)
        self.encoder1_1 = ResBlock3D(input_channels, 64, stride=2, p=0.5)
        self.encoder2_1 = ResBlock3D(64, 128, stride=2, p=0.5)
        self.encoder3_1 = ResBlock3D(128, 256, stride=2, p=0.5)

        self.norm3_1 = norm_layer(embed_dims[1])
        self.norm4_1 = norm_layer(embed_dims[2])

        self.dnorm3_1 = norm_layer(320)
        self.dnorm4_1 = norm_layer(256)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_1 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

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



        self.block1_1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_1[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2_1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_1[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1_1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_1[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2_1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_1[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed3_1 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                                embed_dim=embed_dims[1])
        self.patch_embed4_1 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                                embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(512, 320, 3, stride=1,padding=1)
        self.decoder2 =   nn.Conv2d(320, 256, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.decoder4 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.decoder1_1 = nn.Conv2d(512, 320, 3, stride=1,padding=1)
        self.decoder2_1 =   nn.Conv2d(320, 256, 3, stride=1, padding=1)
        self.decoder3_1 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.decoder4_1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder5_1 =   nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(320)
        self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(128)
        self.dbn4 = nn.BatchNorm2d(64)
        self.dbn1_1 = nn.BatchNorm2d(320)
        self.dbn2_1 = nn.BatchNorm2d(256)
        self.dbn3_1 = nn.BatchNorm2d(128)
        self.dbn4_1 = nn.BatchNorm2d(64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_1 = nn.Conv2d(64, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

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

        ### Stage 1_1
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        out_1 = self.encoder1_1(x)
        #get middle tensor
        idx1_1 = out_1.shape[2]//2
        t1_1 = out_1[:,:,idx1_1,:,:]

        ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out = self.encoder2(out)
        idx2 = out.shape[2]//2
        t2 = out[:,:,idx2,:,:]

        ### Stage 2_1
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out_1 = self.encoder2_1(out_1)
        idx2_1 = out_1.shape[2]//2
        t2_1 = out_1[:,:,idx2_1,:,:]
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out = self.encoder3(out)
        idx3 = out.shape[2]//2
        t3 = out[:,:,idx3,:,:]
        out = t3
        ### Stage 3_1
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out_1 = self.encoder3_1(out_1)
        idx3_1 = out_1.shape[2]//2
        t3_1 = out_1[:,:,idx3_1,:,:]
        out_1 = t3_1

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
        t4 = out
        # print("stage4:",out.shape)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)


        ### Tokenized MLP Stage_1
        ### Stage 4_1

        out_1,H_1,W_1 = self.patch_embed3(out_1)
        for i, blk in enumerate(self.block1_1):
            out_1 = blk(out_1, H_1, W_1)
        out_1 = self.norm3_1(out_1)
        out_1 = out_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        t4_1 = out_1

        ### Bottleneck_1

        out_1 ,H_1,W_1= self.patch_embed4_1(out_1)
        for i, blk in enumerate(self.block2_1):
            out_1 = blk(out_1, H_1, W_1)
        out_1 = self.norm4_1(out_1)
        out_1 = out_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4_1

        out_1 = F.relu(F.interpolate(self.dbn1_1(self.decoder1_1(out_1)),scale_factor=(2,2),mode ='bilinear'))

        out_1 = torch.add(out_1,t4_1)
        out_1 = torch.add(out_1,t4)

        # print("stage4:",out.shape)
        _,_,H_1,W_1 = out_1.shape
        out_1 = out_1.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1_1):
            out_1 = blk(out_1, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        t3 = out
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        t2 = out
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        t1 = out
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        ### Stage 3_1

        out_1 = self.dnorm3_1(out_1)
        out_1 = out_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        out_1 = F.relu(F.interpolate(self.dbn2_1(self.decoder2_1(out_1)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t3_1)
        out_1 = torch.add(out_1,t3)

        _,_,H_1,W_1 = out_1.shape
        out_1 = out_1.flatten(2).transpose(1,2)

        for i, blk in enumerate(self.dblock2_1):
            out_1 = blk(out_1, H_1, W_1)

        out_1 = self.dnorm4_1(out_1)
        out_1 = out_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()

        out_1 = F.relu(F.interpolate(self.dbn3_1(self.decoder3_1(out_1)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t2_1)
        out_1 = torch.add(out_1,t2)
        out_1 = F.relu(F.interpolate(self.dbn4_1(self.decoder4_1(out_1)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t1_1)
        out_1 = torch.add(out_1,t1)

        out_1 = F.relu(F.interpolate(self.decoder5_1(out_1),scale_factor=(2,2),mode ='bilinear'))
        # ipdb.set_trace()

        return self.final_1(out_1),out_1,self.final(out),out
class UNext_Branch_t_64(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 256, 320, 512],
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

        self.dnorm3 = norm_layer(320)
        self.dnorm4 = norm_layer(256)



        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_1 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

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
        self.patch_embed3_1 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                                embed_dim=embed_dims[1])
        self.patch_embed4_1 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                                embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(512, 320, 3, stride=1,padding=1)
        self.decoder2 =   nn.Conv2d(320, 256, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.decoder4 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)

        self.decoder1_1 = nn.Conv2d(512, 320, 3, stride=1,padding=1)
        self.decoder2_1 =   nn.Conv2d(320, 256, 3, stride=1, padding=1)
        self.decoder3_1 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.decoder4_1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder5_1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(320)
        self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(128)
        self.dbn4 = nn.BatchNorm2d(64)
        self.dbn1_1 = nn.BatchNorm2d(320)
        self.dbn2_1 = nn.BatchNorm2d(256)
        self.dbn3_1 = nn.BatchNorm2d(128)
        self.dbn4_1 = nn.BatchNorm2d(64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_1 = nn.Conv2d(32, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

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
        # t4 = out
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
        out_1 =out
        H_1 = H
        W_1 = W

        ### Stage 2_1

        out_1 = out_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()

        out_1 = F.relu(F.interpolate(self.dbn3_1(self.decoder3_1(out_1)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t2)
        t2 = out_1
        out_1 = F.relu(F.interpolate(self.dbn4_1(self.decoder4_1(out_1)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t1)
        t1 = out_1

        out_1 = F.relu(F.interpolate(self.decoder5_1(out_1),scale_factor=(2,2),mode ='bilinear'))

        ### Stage 2
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)

        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)

        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))



        out = torch.cat([out,out_1],dim=1)
        # ipdb.set_trace()

        return self.final(out),out,self.final_1(out_1),out_1




#EOF
def Unext_double():
    return UNext_Double_t(num_classes=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                          num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                          depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
def Unext_Branch_64():
    return UNext_Branch_t_64(num_classes=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 320, 512],
                             num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                             attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                             depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
def Unext_double_64():
    return UNext_Double_t_64(num_classes=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 320, 512],
                          num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                          depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
def Unext_1():
    return UNext_t(num_classes=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
def Unext_2d():
    return UNext(num_classes=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                   num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                   attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                   depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
def Unext_heads():
    return UNext_multiheads(num_classes=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                   num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                   attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                   depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
def Unext_cat():
    return UNext_t_cat(num_classes=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3, encode_dims=[32,64,128], embed_dims=[ 128,320, 512], decode_dims=[512,320,256,128,64],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])