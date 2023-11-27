
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

class ResBlock3D_Attention(nn.Module):
    """ residual block """
    def __init__(self, in_channels, out_channels, stride=[[2,2,2],[2,2,2]], p=0.5, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(in_channels)
        padding = [[0,1,1],[0,1,1]]

        for i in range(len(stride)):
            if stride[i][0] == 1:
                padding[i][0] = 1
            else:
                padding[i][0] = 0
        self.conv1 = conv_333(in_channels, out_channels, stride=stride[0], padding=padding[0])
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = conv_333(out_channels, out_channels, stride=stride[1], padding=padding[1])
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout3d(p=p)


    def forward(self, x):
        # print("input residual size: {}".format(residual.size()))
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dp(out)
        out = F.sigmoid(out)


        return out

class ResBlock3D_Imbalance_Aware(nn.Module):
    """ residual block """
    def __init__(self, in_channels, out_channels, stride=[[2,2,2],[2,2,2]], p=0.5, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(in_channels)
        padding = [[0,1,1],[0,1,1]]

        for i in range(len(stride)):
            if stride[i][0] == 1:
                padding[i][0] = 1
            else:
                padding[i][0] = 0
        self.conv1 = conv_333(in_channels, out_channels, stride=stride[0], padding=padding[0])
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = conv_333(out_channels, out_channels, stride=stride[1], padding=padding[1])
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout3d(p=p)


    def forward(self, x):
        # print("input residual size: {}".format(residual.size()))
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dp(out)
        out = F.sigmoid(out)


        return out


class ResBlock3D_Attention_4stride(nn.Module):
    """ residual block """
    def __init__(self, in_channels, out_channels, stride=[[1,2,2],[1,2,2],[2,2,2],[2,2,2]], p=0.5):
        super().__init__()
        padding = [[0,1,1],[0,1,1],[0,1,1],[0,1,1]]

        for i in range(len(stride)):
            if stride[i][0] == 1:
                padding[i][0] = 1
            else:
                padding[i][0] = 0
        mid_channels = (in_channels + out_channels) // 2
        self.bn1 = nn.BatchNorm3d(in_channels)

        self.conv1 = conv_333(in_channels, mid_channels, stride=stride[0], padding=padding[0])
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.conv2 = conv_333(mid_channels, mid_channels, stride=stride[1], padding=padding[1])
        self.bn3 = nn.BatchNorm3d(mid_channels)
        self.conv3 = conv_333(mid_channels, mid_channels, stride=stride[2], padding=padding[2])
        self.bn4 = nn.BatchNorm3d(mid_channels)
        self.conv4 = conv_333(mid_channels, out_channels, stride=stride[3], padding=padding[3])
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout3d(p=p)


    def forward(self, x):
        # print("input residual size: {}".format(residual.size()))
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.dp(out)
        out = F.sigmoid(out)


        return out

class AttentionConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels,in_size,stride=1, stride_attention=[[2,2,2],[2,2,2]],stride_attenconv = 2, bias=True):
        super(AttentionConvBlock3d, self).__init__()
        kernel_size = [in_size[0]//(stride_attention[0][0]*stride_attention[1][0]),in_size[1]//(stride_attention[0][1]*stride_attention[1][1]),
                       in_size[2]//(stride_attention[0][2]*stride_attention[1][2])]
        padding = [(kernel_size[0]-stride_attenconv)//2,(kernel_size[1]-stride_attenconv)//2 if not kernel_size[1]%2 else (kernel_size[1]-stride_attenconv)//2+1,
                   (kernel_size[2]-stride_attenconv)//2 if not kernel_size[2]%2 else (kernel_size[2]-stride_attenconv)//2+1]
        # self.attenconv = AttentionConv3D(in_channels, out_channels,kernel_size, stride_attenconv, padding, bias)
        self.resblock3d_attention = ResBlock3D_Attention(in_channels,out_channels,stride_attention,p=0.5)
        self.attenconv = AttentionConv3D_einsum(in_channels, out_channels, kernel_size, stride_attenconv, padding, bias)
        self.resblock3d= ResBlock3D(in_channels, in_channels, stride, p=0.5)

        # self.convCheck = Conv_check(in_channels, out_channels, kernel_size, stride_attenconv, padding, bias)
    def forward(self, x):
        attention_x = self.resblock3d_attention(x)
        out = self.resblock3d(x)
        # check = self.convCheck(out)
        # ipdb.set_trace()
        out = self.attenconv(out,attention_x)
        return out

class AttentionConvBlock3d_4stride(nn.Module):
    def __init__(self, in_channels, out_channels,in_size,stride=1, stride_attention=[[1,2,2],[1,2,2],[2,2,2],[2,2,2]],stride_attenconv = 2, bias=True):
        super(AttentionConvBlock3d_4stride, self).__init__()

        kernel_size = [in_size[0]//(np.prod(stride_attention,axis =0)[0]),in_size[1]//(np.prod(stride_attention,axis =0)[1]),
                       in_size[2]//(np.prod(stride_attention,axis =0)[2])]

        # ipdb.set_trace()
        # padding = [(kernel_size[0]-stride_attenconv)//2,(kernel_size[1]-stride_attenconv)//2,(kernel_size[2]-stride_attenconv)//2]
        padding = [(kernel_size[0]-stride_attenconv)//2,(kernel_size[1]-stride_attenconv)//2 if not kernel_size[1]%2 else (kernel_size[1]-stride_attenconv)//2+1,
                   (kernel_size[2]-stride_attenconv)//2 if not kernel_size[2]%2 else (kernel_size[2]-stride_attenconv)//2+1]
        self.resblock3d_attention = ResBlock3D_Attention_4stride(in_channels,out_channels,stride_attention,p=0.5)
        # print("in_channels, out_channels, kernel_size, stride_attenconv, padding, bias:",in_channels, out_channels, kernel_size, stride_attenconv, padding, bias)
        self.attenconv = AttentionConv3D_einsum(in_channels, out_channels, kernel_size, stride_attenconv, padding, bias)
        self.resblock3d= ResBlock3D(in_channels, in_channels, stride, p=0.5)

        # self.convCheck = Conv_check(in_channels, out_channels, kernel_size, stride_attenconv, padding, bias)
    def forward(self, x):
        attention_x = self.resblock3d_attention(x)
        out = self.resblock3d(x)
        # check = self.convCheck(out)
        # ipdb.set_trace()
        out = self.attenconv(out,attention_x)
        return out




#
#
#
# class AttentionConv3D(nn.Module):
#     def __init__(self, in_channels, out_channels,kernel_size, stride=1, padding=0, bias=True):
#         super(AttentionConv3D, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         if isinstance(stride, int):
#             stride = to_3tuple(stride)
#         self.stride = stride
#         self.padding = padding
#         self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]))
#         self.bn = nn.BatchNorm3d(in_channels)
#     def forward(self, x, attention_matrix):
#         out = self.bn(x)
#         batch_size, _, D, H, W = x.size()
#
#         D_new = (D + 2 * self.padding[0] - self.weights.size(2)) // self.stride[0] + 1
#         H_new = (H + 2 * self.padding[1] - self.weights.size(3)) // self.stride[1] + 1
#         W_new = (W + 2 * self.padding[2] - self.weights.size(4)) // self.stride[2] + 1
#         output = torch.zeros(batch_size, self.out_channels, D_new, H_new, W_new, device=x.device)  # Use x.device to ensure the same device
#
#         for b in range(batch_size):
#             for c_out in range(self.out_channels):
#                 for c_in in range(self.in_channels):
#                     weight = self.weights[c_out, c_in] * attention_matrix[b, c_out]
#                     conv_out = torch.nn.functional.conv3d(out[b, c_in].unsqueeze(0).unsqueeze(0),
#                                                           weight.unsqueeze(0).unsqueeze(0),
#                                                           stride=self.stride, padding=self.padding)
#                     output[b, c_out] += conv_out.squeeze(0).squeeze(0)
#         return output
class AttentionConv3D_einsum(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size, stride=1, padding=0, bias=True):
        super(AttentionConv3D_einsum, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(stride, int):
            stride = to_3tuple(stride)
        self.stride = stride
        self.padding = padding
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]))
        self.bn = nn.BatchNorm3d(in_channels)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))

        # self.bias = bias
    def forward(self, x, attention_matrix):

        out = self.bn(x)

        N, C, D, H, W = out.shape
        expanded_weights = self.weights.unsqueeze(0).expand(N, -1, -1, -1, -1, -1)
        weight = expanded_weights * attention_matrix.unsqueeze(2)
        # weight = expanded_weights
        # ipdb.set_trace()
        N, C_out, _, K_D, K_H, K_W = weight.shape

        # 添加padding
        input_padded = torch.nn.functional.pad(out, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        D_, H_, W_ = input_padded.shape[2:]
        new_stride = (
        C* D_ * H_ * W_,
        D_ * H_ * W_,
        self.stride[0] * W_ * H_,
        self.stride[1]*W_,
        self.stride[2],
        W_*H_,
        W_,
        1
        )
        # input_strided = input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1, (H_- K_H) // self.stride[1] + 1,(W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
        #                      new_stride)
        #
        # 执行卷积操作
        output = torch.einsum('bidhwjkl,boijkl->bodhw', input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1,
                                                                                 (H_- K_H) // self.stride[1] + 1,
                                                                                 (W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
                                                                                new_stride), weight)
        # ipdb.set_trace()

        # 添加偏置项（如果有）
        if self.bias is not None:
            output += self.bias[None, :, None, None, None]

        return output
class MIAOutConv3D(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,im_class_num=1,tasks_num =2,im_kernel_ratio=0.5, stride=1, padding=0, bias=True):
        #MIAOutConv3D means Multitask-available Convolution 3D with Imbalance Aware in Output channel
        super(MIAOutConv3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = to_3tuple(kernel_size)
        if isinstance(stride, int):
            stride = to_3tuple(stride)
        if isinstance(padding, int):
            padding = to_3tuple(padding)
        self.fan_out = out_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.im_kernel_ratio = im_kernel_ratio
        self.im_kernel_num = int(out_channels*im_kernel_ratio)
        if (out_channels-self.im_kernel_num)>0:
            self.weights_common = nn.Parameter(torch.randn(out_channels-self.im_kernel_num, in_channels,
                                                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
            nn.init.kaiming_normal_(self.weights_common, mode='fan_out', nonlinearity='relu')
            # self.weights_common = nn.Parameter(torch.ones(out_channels-self.im_kernel_num, in_channels,
            #                                               self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
        if self.im_kernel_num>0:
            self.weights_imbalance = nn.Parameter(torch.randn(1+im_class_num,self.im_kernel_num, in_channels,
                                                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
            for i in range(1+im_class_num):
                nn.init.kaiming_normal_(self.weights_imbalance[i], mode='fan_out', nonlinearity='relu')
        if tasks_num>1:
            if  self.im_kernel_num>0:
                self.weights_multi = nn.Parameter(torch.randn(tasks_num-1,self.im_kernel_num, in_channels,
                                                          self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
                for i in range(tasks_num-1):
                    nn.init.kaiming_normal_(self.weights_multi[i], mode='fan_out', nonlinearity='relu')
                # self.weights_multi = nn.Parameter(torch.zeros(tasks_num-1,self.im_kernel_num, in_channels,
                #                                               self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
        self.bias = bias
        if bias:
            if self.im_kernel_num>0:
                self.bias_im = nn.Parameter(torch.zeros(im_class_num+1,self.im_kernel_num))
                self.bias_multi = nn.Parameter(torch.zeros(tasks_num-1,self.im_kernel_num))
            if (out_channels-self.im_kernel_num)>0:
                self.bias_common = nn.Parameter(torch.zeros(out_channels-self.im_kernel_num))



        # self.bias = bias
    def forward(self, x, conv_attention=None,task_indx=1):
        # print("task_indx: ",task_indx)
        # print("self.im_kernel_num: ",self.im_kernel_num)
        # conv_attention: N, C


        N, C, D, H, W = x.shape
        if task_indx>0:
            if not self.im_kernel_num:
                weight = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
            elif not (self.out_channels-self.im_kernel_num):
                weight =self.weights_multi[task_indx-1].unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
            else:
                weight_multi = self.weights_multi[task_indx-1].unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
                weight_common = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
                weight = torch.cat((weight_common, weight_multi), dim=1)
                weight_ =torch.cat((self.weights_common, self.weights_multi[task_indx-1]), dim=0)
        else:
            _ , C_atten= conv_attention.shape
            if not self.im_kernel_num:
                weight = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
                # print("3dout only use common weights")
            elif not (self.out_channels-self.im_kernel_num):
                _,_, C_in,K_D, K_H, K_W = self.weights_imbalance.shape
                expanded_weights_im = self.weights_imbalance.unsqueeze(0).expand(N, -1, -1, -1, -1, -1,-1)

                expanded_conv_attention = conv_attention.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6).expand(-1, -1,self.im_kernel_num, C_in,K_D, K_H, K_W)
                weight_im = expanded_weights_im * expanded_conv_attention
                weight = weight_im.sum(dim=1)
                # print("3dout only use imbalance weights")
            else:
                # print("\033[91m" +"3dout use common and imbalance weights"+ "\033[0m")
                _,_, C_in,K_D, K_H, K_W = self.weights_imbalance.shape
                expanded_weights_im = self.weights_imbalance.unsqueeze(0).expand(N, -1, -1, -1, -1, -1,-1)

                expanded_conv_attention = conv_attention.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6).expand(-1, -1,self.im_kernel_num, C_in,K_D, K_H, K_W)
                weight_im = expanded_weights_im * expanded_conv_attention
                weight_im = weight_im.sum(dim=1)
                # weight = expanded_weights
                # ipdb.set_trace()
                weight_common = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1, -1)
                weight = torch.cat((weight_common, weight_im), dim=1)
        x_old = x
        x = x.view(1, -1, D, H, W)
        N, C_out, C_in,K_D, K_H, K_W = weight.shape
        weight = weight.reshape(N*C_out, C_in,K_D, K_H, K_W).contiguous()
        # aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size, self.kernel_size)
        if self.bias :
            if task_indx>0:
                if not self.im_kernel_num:
                    aggregate_bias = self.bias_common.unsqueeze(0).expand(N,-1).reshape(-1)
                elif not (self.out_channels-self.im_kernel_num):
                    aggregate_bias = self.bias_multi[task_indx-1].unsqueeze(0).expand(N,-1).reshape(-1)
                else:
                    aggregate_bias = torch.cat((self.bias_common,self.bias_multi[task_indx-1]), dim=0).unsqueeze(0).expand(N, -1).reshape(-1)
            else:
                if not self.im_kernel_num:
                    aggregate_bias = self.bias_common.unsqueeze(0).expand(N,-1).reshape(-1)
                elif not (self.out_channels-self.im_kernel_num):
                    aggregate_bias = torch.mm(conv_attention, self.bias_im).reshape(-1)
                else:
                    aggregate_bias =torch.cat((self.bias_common.unsqueeze(0).expand(N,-1),
                                               self.im_kernel_ratio*torch.mm(conv_attention, self.bias_im)),dim=1).reshape(-1) #N,C_out_im
            output = F.conv3d(x, weight=weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,groups=N)
        else:
            output = F.conv3d(x, weight=weight, bias=None, stride=self.stride, padding=self.padding, groups=N)

        output = output.view(N, self.out_channels, output.size(-3), output.size(-2), output.size(-1))


        # # 添加padding
        # input_padded = torch.nn.functional.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        # D_, H_, W_ = input_padded.shape[2:]
        # new_stride = (
        #     C* D_ * H_ * W_,
        #     D_ * H_ * W_,
        #     self.stride[0] * W_ * H_,
        #     self.stride[1]*W_,
        #     self.stride[2],
        #     W_*H_,
        #     W_,
        #     1
        # )
        # # input_strided = input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1, (H_- K_H) // self.stride[1] + 1,(W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
        # #                      new_stride)
        # #
        #
        #
        #
        #
        # # 执行卷积操作
        # output = torch.einsum('bidhwjkl,boijkl->bodhw', input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1,
        #                                                                          (H_- K_H) // self.stride[1] + 1,
        #                                                                          (W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
        #                                                                         new_stride), weight)
        # ipdb.set_trace()

        # # 添加偏置项（如果有）
        # if self.bias:
        #     if task_indx>0:
        #         output += self.bias_multi[task_indx-1][None, :, None, None, None]
        #     else:
        #         bias_im = self.bias_im.unsqueeze(0).expand(N, -1,-1)*conv_attention.unsqueeze(2).expand(-1,-1,self.out_channels)
        #         output += bias_im.mean(dim=1).squeeze(1)[:,:,None,None,None]


        #
        # if self.bias:
        #     output = F.conv3d(x, self.weights_common, bias=self.bias_common, stride=self.stride, padding=self.padding)
        # else:
        #     output = F.conv3d(x, self.weights_common, bias=None, stride=self.stride, padding=self.padding)

        # if self.bias:
        #     if not self.im_kernel_num:
        #         output1 = F.conv3d(x_old,self.weights_common , bias=self.bias_common, stride=self.stride, padding=self.padding)
        #     elif not (self.out_channels-self.im_kernel_num):
        #         output1 = torch.tensor([]).cuda()
        #         for i in range(N):
        #             weight_im_i = conv_attention[i].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand( -1,self.im_kernel_num, C_in,K_D, K_H, K_W)*self.weights_imbalance
        #             weight_im_i = weight_im_i.sum(dim=0)
        #             bias_im_i = torch.mm(conv_attention, self.bias_im)[i]
        #             output1_i = F.conv3d(x_old[i].unsqueeze(0),weight_im_i , bias=bias_im_i, stride=self.stride, padding=self.padding)
        #             output1 = torch.cat((output1,output1_i),dim=0)
        #
        #         # output1 = F.conv3d(x_old,self.weights_common , bias=self.bias_multi[task_indx-1], stride=self.stride, padding=self.padding)
        #     # ipdb.set_trace()
        #     if torch.allclose(output, output1, atol=1e-5, rtol=1e-5):
        #         print("in MIAoutConv Custom convolution 3d matches  e-5 PyTorch built-in convolution.weight shape:{}".format(self.weights_imbalance.shape))
        #         pass
        #     else:
        #     # print("Custom convolution does not match PyTorch built-in convolution.")
        #     #print in red
        #         print("\033[91m" + "in MIAoutConv  Custom convolution 3d does not  match e-5 PyTorch built-in convolution. weight shape:{}".format(self.weights_imbalance.shape) + "\033[0m")
        #


        # if self.bias:
        #     out_3d_1 = F.conv3d(x, weight_, bias=self.bias_multi[task_indx-1], stride=self.stride, padding=self.padding)
        # else:
        #     out_3d_1 = F.conv3d(x, weight_, stride=self.stride, padding=self.padding)
        # N, C, D, H, W = x.shape
        # expanded_weights = weight_.unsqueeze(0).expand(N, -1, -1, -1, -1, -1)
        # weight = expanded_weights
        # N, C_out, _, K_D, K_H, K_W = weight.shape
        #
        # # 添加padding
        # input_padded = torch.nn.functional.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        # D_, H_, W_ = input_padded.shape[2:]
        # new_stride = (
        #     C* D_ * H_ * W_,
        #     D_ * H_ * W_,
        #     self.stride[0] * W_ * H_,
        #     self.stride[1]*W_,
        #     self.stride[2],
        #     W_*H_,
        #     W_,
        #     1
        # )
        # input_strided = input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1, (H_- K_H) // self.stride[1] + 1,(W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
        #                                         new_stride)
        #
        # out_3d_2 = torch.einsum('bidhwjkl,boijkl->bodhw', input_strided, weight)
        # if self.bias:
        #     out_3d_2 += self.bias_multi[task_indx-1][None, :, None, None, None]
        # if torch.allclose(out_3d_1, out_3d_2):
        #     print("2nd Custom convolution matches PyTorch built-in convolution.")
        #
        # else:
        #     print("\033[91m" + "2nd Custom convolution does not match PyTorch built-in convolution." + "\033[0m")






        return output

class MIAInConv3D(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,im_class_num=1,tasks_num =2,im_kernel_ratio=0.5, stride=1, padding=0, bias=True):
        #MIAOutConv3D means Multitask-available Convolution 3D with Imbalance Aware in Output channel
        super(MIAInConv3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = to_3tuple(kernel_size)
        if isinstance(stride, int):
            stride = to_3tuple(stride)
        if isinstance(padding, int):
            padding = to_3tuple(padding)
        self.fan_out = out_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.im_kernel_ratio = im_kernel_ratio
        self.im_kernel_num = int(in_channels*im_kernel_ratio)
        # print("in MIAInConv3D im_kernel_num:{}".format(self.im_kernel_num))
        # print("in MIAInConv3D inchannels:{}".format(in_channels))
        # print("in MIAInConv3D im_kernel_ratio:{}".format(im_kernel_ratio))
        if (in_channels-self.im_kernel_num)>0:
            self.weights_common = nn.Parameter(torch.randn(out_channels, in_channels-self.im_kernel_num,
                                                       self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
            nn.init.kaiming_normal_(self.weights_common, mode='fan_out', nonlinearity='relu')
            # self.weights_common = nn.Parameter(torch.ones(out_channels, in_channels-self.im_kernel_num,
            #                                               self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
        if self.im_kernel_num>0:
            self.weights_imbalance = nn.Parameter(torch.randn(1+im_class_num,out_channels, self.im_kernel_num,
                                                          self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
            for i in range(1+im_class_num):
                nn.init.kaiming_normal_(self.weights_imbalance[i], mode='fan_out', nonlinearity='relu')
        if tasks_num>1:
            if self.im_kernel_num>0:
                self.weights_multi = nn.Parameter(torch.randn(tasks_num-1,out_channels, self.im_kernel_num,
                                                          self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
                for i in range(tasks_num-1):
                    nn.init.kaiming_normal_(self.weights_multi[i], mode='fan_out', nonlinearity='relu')
                # self.weights_multi = nn.Parameter(torch.zeros(tasks_num-1,out_channels, self.im_kernel_num,
                #                                               self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
        self.bias = bias
        if bias:
            self.bias_im = nn.Parameter(torch.zeros(im_class_num+1,out_channels))
            self.bias_multi = nn.Parameter(torch.zeros(tasks_num-1,out_channels))
            self.bias_common = nn.Parameter(torch.zeros(out_channels))



        # self.bias = bias
    def forward(self, x, conv_attention=None,task_indx=1):
        # conv_attention: N, C



        N, C, D, H, W = x.shape
        if task_indx>0:
            if not self.im_kernel_num:
                weight = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
            elif not (self.in_channels-self.im_kernel_num):
                weight =self.weights_multi[task_indx-1].unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
            else:
                weight_multi = self.weights_multi[task_indx-1].unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
                weight_common = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
                # weight = torch.cat(( weight_multi,weight_common), dim=2)
                weight = torch.cat((weight_common, weight_multi), dim=2)
                weight_ =torch.cat((self.weights_common, self.weights_multi[task_indx-1]), dim=1)
        else:
            _ , C_atten= conv_attention.shape
            if not self.im_kernel_num:
                weight = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
                # print("3din only use common weights")
            elif not (self.in_channels-self.im_kernel_num):
                _,C_out,_,K_D, K_H, K_W = self.weights_imbalance.shape
                expanded_weights_im = self.weights_imbalance.unsqueeze(0).expand(N, -1, -1, -1, -1, -1,-1)

                expanded_conv_attention = conv_attention.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6).expand(-1,-1,C_out,self.im_kernel_num,K_D, K_H, K_W)
                weight_im = expanded_weights_im * expanded_conv_attention
                weight = weight_im.sum(dim=1)
                # print("3din only use imbalance weights")
            else:
                # print("\033[91m" +"3din use both common and imbalance weights"+ "\033[0m")

                _,C_out,_,K_D, K_H, K_W = self.weights_imbalance.shape
                expanded_weights_im = self.weights_imbalance.unsqueeze(0).expand(N, -1, -1, -1, -1, -1,-1)

                expanded_conv_attention = conv_attention.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6).expand(-1,-1,C_out,self.im_kernel_num,K_D, K_H, K_W)
                weight_im = expanded_weights_im * expanded_conv_attention
                weight_im = weight_im.sum(dim=1)
                # weight = expanded_weights
                # ipdb.set_trace()
                weight_common = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1, -1)
                weight = torch.cat((weight_common, weight_im), dim=2)
                # weight = torch.cat(( weight_im,weight_common), dim=2)

        N, C_out, C_in,K_D, K_H, K_W = weight.shape
        x = x.view(1, -1, D, H, W)

        weight = weight.reshape(N*C_out, C_in,K_D, K_H, K_W).contiguous()
        # aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size, self.kernel_size)
        if self.bias:
            if task_indx>0:
                aggregate_bias = (1-self.im_kernel_ratio)*self.bias_common + self.im_kernel_ratio*self.bias_multi[task_indx-1]
                aggregate_bias = aggregate_bias.unsqueeze(0).expand(N,-1).reshape(-1)
            else:
                aggregate_bias = (1-self.im_kernel_ratio)*self.bias_common.unsqueeze(0).expand(N,-1) + self.im_kernel_ratio*torch.mm(conv_attention, self.bias_im)
                aggregate_bias = aggregate_bias.reshape(-1)
                 #N,C_out_im
            output = F.conv3d(x, weight=weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,groups=N)
        else:
            output = F.conv3d(x, weight=weight, bias=None, stride=self.stride, padding=self.padding, groups=N)

        output = output.view(N, self.out_channels, output.size(-3), output.size(-2), output.size(-1))

        # # 添加padding
        # input_padded = torch.nn.functional.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0])).to(torch.float64)
        # D_, H_, W_ = input_padded.shape[2:]
        # new_stride = (
        #     C* D_ * H_ * W_,
        #     D_ * H_ * W_,
        #     self.stride[0] * W_ * H_,
        #     self.stride[1]*W_,
        #     self.stride[2],
        #     W_*H_,
        #     W_,
        #     1
        # )
        # # input_strided = input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1, (H_- K_H) // self.stride[1] + 1,(W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
        # #                      new_stride)
        # #
        #
        #
        #
        #
        # # # 执行卷积操作
        # # output = torch.einsum('bidhwjkl,boijkl->bodhw', input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1,
        # #                                                                          (H_- K_H) // self.stride[1] + 1,
        # #                                                                          (W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
        # #                                                                         new_stride), weight).to(torch.float32)
        # # # ipdb.set_trace()
        # #
        # # # 添加偏置项（如果有）
        # # if self.bias:
        # #     if task_indx>0:
        # #         output += self.bias_multi[task_indx-1][None, :, None, None, None]
        # #     else:
        # #         bias_im = self.bias_im.unsqueeze(0).expand(N, -1,-1)*conv_attention.unsqueeze(2).expand(-1,-1,self.out_channels)
        # #         output += bias_im.mean(dim=1).squeeze(1)[:,:,None,None,None]
        # if self.bias:
        #     output = F.conv3d(x, self.weights_common, bias=self.bias_common, stride=self.stride, padding=self.padding)
        # else:
        #     output = F.conv3d(x, self.weights_common, bias=None, stride=self.stride, padding=self.padding)

        # if self.bias:
        #     if task_indx>0:
        #         output1 = F.conv3d(x, weight_, bias=self.bias_multi[task_indx-1], stride=self.stride, padding=self.padding)
        #         if torch.allclose(output, output1, atol=1e-3, rtol=1e-3):
        #             # print("in MIAoutConv Custom convolution 3d matches PyTorch built-in convolution.weight shape:{}".format(weight_.shape))
        #             pass
        #         else:
        #         # print("Custom convolution does not match PyTorch built-in convolution.")
        #         #print in red
        #             print("\033[91m" + "in MIAoutConv  Custom convolution 3d does not match PyTorch built-in convolution. weight shape:{}".format(weight_.shape) + "\033[0m")



        # if self.bias:
        #     out_3d_1 = F.conv3d(x, weight_, bias=self.bias_multi[task_indx-1], stride=self.stride, padding=self.padding)
        # else:
        #     out_3d_1 = F.conv3d(x, weight_, stride=self.stride, padding=self.padding)
        # N, C, D, H, W = x.shape
        # expanded_weights = weight_.unsqueeze(0).expand(N, -1, -1, -1, -1, -1)
        # weight = expanded_weights
        # N, C_out, _, K_D, K_H, K_W = weight.shape
        #
        # # 添加padding
        # input_padded = torch.nn.functional.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        # D_, H_, W_ = input_padded.shape[2:]
        # new_stride = (
        #     C* D_ * H_ * W_,
        #     D_ * H_ * W_,
        #     self.stride[0] * W_ * H_,
        #     self.stride[1]*W_,
        #     self.stride[2],
        #     W_*H_,
        #     W_,
        #     1
        # )
        # input_strided = input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1, (H_- K_H) // self.stride[1] + 1,(W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
        #                                         new_stride)
        #
        # out_3d_2 = torch.einsum('bidhwjkl,boijkl->bodhw', input_strided, weight)
        # if self.bias:
        #     out_3d_2 += self.bias_multi[task_indx-1][None, :, None, None, None]
        # if torch.allclose(out_3d_1, out_3d_2):
        #     print("2nd Custom convolution matches PyTorch built-in convolution.")
        #
        # else:
        #     print("\033[91m" + "2nd Custom convolution does not match PyTorch built-in convolution." + "\033[0m")
        return output

class MIAOutConv2D(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,im_class_num=1,tasks_num =2, im_kernel_ratio=0.5, stride=1, padding=0, bias=True):
        super(MIAOutConv2D, self).__init__()
        #MIAOutConv2D means Multitask-available Convolution 2D with Imbalance Aware in Output channel

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = to_2tuple(kernel_size)
        if isinstance(stride, int):
            stride = to_2tuple(stride)
        if isinstance(padding, int):
            padding = to_2tuple(padding)
        self.fan_out = out_channels*kernel_size[0]*kernel_size[1]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.im_kernel_ratio = im_kernel_ratio
        self.im_kernel_num = int(out_channels*im_kernel_ratio)
        if (out_channels-self.im_kernel_num)>0:
            self.weights_common = nn.Parameter(torch.randn(out_channels-self.im_kernel_num, in_channels,
                                                       self.kernel_size[0], self.kernel_size[1]))
            nn.init.kaiming_normal_(self.weights_common, mode='fan_out', nonlinearity='relu')

        if self.im_kernel_num>0:
            self.weights_imbalance = nn.Parameter(torch.randn(1+im_class_num,self.im_kernel_num, in_channels,
                                                          self.kernel_size[0], self.kernel_size[1]))
            for i in range(1+im_class_num):
                nn.init.kaiming_normal_(self.weights_imbalance[i], mode='fan_out', nonlinearity='relu')
        if tasks_num>1:
            if self.im_kernel_num>0:
                self.weights_multi = nn.Parameter(torch.randn(tasks_num-1,self.im_kernel_num, in_channels,
                                                          self.kernel_size[0], self.kernel_size[1]))
                for i in range(tasks_num-1):
                    nn.init.kaiming_normal_(self.weights_multi[i], mode='fan_out', nonlinearity='relu')

        self.bias = bias
        if bias:
            if self.im_kernel_num>0:
                self.bias_im = nn.Parameter(torch.zeros(im_class_num+1,self.im_kernel_num))
                self.bias_multi = nn.Parameter(torch.zeros(tasks_num-1,self.im_kernel_num))
            if (out_channels-self.im_kernel_num)>0:
                self.bias_common = nn.Parameter(torch.zeros(out_channels-self.im_kernel_num))


        # self.bias = bias
    def forward(self, x, conv_attention=None,task_indx =1):
        # conv_attention: N, C
        N, C,  H, W = x.shape
        if task_indx>0:
            if not self.im_kernel_num:
                weight = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1)
            elif not (self.out_channels-self.im_kernel_num):
                weight =self.weights_multi[task_indx-1].unsqueeze(0).expand(N, -1, -1, -1, -1)
            else:
                weight_multi = self.weights_multi[task_indx-1].unsqueeze(0).expand(N, -1, -1, -1, -1)
                weight_common = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1)
                weight = torch.cat((weight_common, weight_multi), dim=1)
                weight_ = torch.cat((self.weights_common, self.weights_multi[task_indx-1]), dim=0)
        else:

            _ , C_atten= conv_attention.shape

            if not self.im_kernel_num:
                weight = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1)
                # print("2Dout only use common weights")
            elif not (self.out_channels-self.im_kernel_num):
                _,_, C_in, K_H, K_W = self.weights_imbalance.shape
                expanded_weights_im = self.weights_imbalance.unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
                expanded_conv_attention = conv_attention.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(-1, -1,self.im_kernel_num, C_in, K_H, K_W)
                weight_im = expanded_weights_im * expanded_conv_attention
                weight = weight_im.sum(dim=1)
                # print("2Dout only use imbalance weights")
            else:
                # print("\033[91m" +"2Dout use both common and imbalance weights"+ "\033[0m")
                _,_, C_in, K_H, K_W = self.weights_imbalance.shape
                expanded_weights_im = self.weights_imbalance.unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
                expanded_conv_attention = conv_attention.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(-1, -1,self.im_kernel_num, C_in, K_H, K_W)
                weight_im = expanded_weights_im * expanded_conv_attention
                weight_im = weight_im.sum(dim=1)
                # weight = expanded_weights
                # ipdb.set_trace()
                weight_common = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1)
                weight = torch.cat((weight_common, weight_im), dim=1)





        # ipdb.set_trace()
        _, C_out,C_in, K_H, K_W = weight.shape
        x = x.view(1, -1,  H, W)
        weight = weight.reshape(N*C_out, C_in, K_H, K_W).contiguous()
        # aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size, self.kernel_size)
        if self.bias:
            if task_indx>0:
                if not self.im_kernel_num:
                    aggregate_bias = self.bias_common.unsqueeze(0).expand(N,-1).reshape(-1)
                elif not (self.out_channels-self.im_kernel_num):
                    aggregate_bias = self.bias_multi[task_indx-1].unsqueeze(0).expand(N,-1).reshape(-1)
                else:
                    aggregate_bias = torch.cat((self.bias_common,self.bias_multi[task_indx-1]), dim=0).unsqueeze(0).expand(N, -1).reshape(-1)
            else:
                if not self.im_kernel_num:
                    aggregate_bias = self.bias_common.unsqueeze(0).expand(N,-1).reshape(-1)
                elif not (self.out_channels-self.im_kernel_num):
                    aggregate_bias = torch.mm(conv_attention, self.bias_im).reshape(-1)
                else:
                    aggregate_bias =torch.cat((self.bias_common.unsqueeze(0).expand(N,-1),
                                               self.im_kernel_ratio*torch.mm(conv_attention, self.bias_im)),dim=1).reshape(-1) #N,C_out_im
            output = F.conv2d(x, weight=weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,groups=N)
        else:
            output = F.conv2d(x, weight=weight, bias=None, stride=self.stride, padding=self.padding, groups=N)

        output = output.view(N, self.out_channels,  output.size(-2), output.size(-1))

        # # 添加padding
        # input_padded = torch.nn.functional.pad(x, ( self.padding[1], self.padding[1], self.padding[0], self.padding[0])).to(torch.float64)
        # H_, W_ = input_padded.shape[2:]
        # new_stride = (
        #     C * H_ * W_,
        #     H_ * W_,
        #     self.stride[0]*W_,
        #     self.stride[1],
        #     W_,
        #     1
        # )
        # # input_strided = input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1, (H_- K_H) // self.stride[1] + 1,(W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
        # #                      new_stride)
        # #
        #
        #
        # #
        # # # 执行卷积操作
        # # output = torch.einsum('bihwkl,boikl->bohw', input_padded.as_strided((N, C,(H_- K_H) // self.stride[0] + 1,
        # #                                                                          (W_-K_W)//self.stride[1]+1,K_H, K_W),new_stride), weight).to(torch.float32)
        # # # ipdb.set_trace()
        # #
        # # # 添加偏置项（如果有）
        # # if self.bias:
        # #     if task_indx>0:
        # #         output += self.bias_multi[task_indx-1][None, :, None, None]
        # #     else:
        # #         bias_im = self.bias_im.unsqueeze(0).expand(N, -1,-1)*conv_attention.unsqueeze(2).expand(-1,-1,self.out_channels)
        # #         output += bias_im.mean(dim=1).squeeze(1)[:, :, None, None]
        #
        #
        # if self.bias:
        #     output = F.conv2d(x, self.weights_common, bias=self.bias_common, stride=self.stride, padding=self.padding)
        # else:
        #     output = F.conv2d(x, self.weights_common, bias=None, stride=self.stride, padding=self.padding)
        # if self.bias:
        #     if task_indx>0:
        #         output1 = F.conv2d(x, weight_, bias=self.bias_multi[task_indx-1], stride=self.stride, padding=self.padding)
        #         if torch.allclose(output, output1, atol=5e-4, rtol=5e-4):
        #             # print("in MIAoutConv Custom convolution 2d matches PyTorch built-in convolution.weight shape:{}".format(weight_.shape))
        #             pass
        #         else:
        #             # print("Custom convolution does not match PyTorch built-in convolution.")
        #             #print in red
        #             print("\033[91m" + "in MIAoutConv  Custom convolution 2d does not match PyTorch built-in convolution. weight shape:{}".format(weight_.shape) + "\033[0m")
        #         # ipdb.set_trace()

        return output

class MIAInConv2D(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,im_class_num=1,tasks_num =2,im_kernel_ratio=0.5, stride=1, padding=0, bias=True):
        super(MIAInConv2D, self).__init__()
        #IAInConv2D means Convolution 2D with Imbalance Aware in Input channel

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = to_2tuple(kernel_size)
        if isinstance(stride, int):
            stride = to_2tuple(stride)
        if isinstance(padding, int):
            padding = to_2tuple(padding)
        self.fan_out = out_channels * kernel_size[0] * kernel_size[1]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.im_kernel_ratio = im_kernel_ratio
        self.im_kernel_num = int(in_channels*im_kernel_ratio)
        if (in_channels-self.im_kernel_num)>0:
            self.weights_common = nn.Parameter(torch.randn(out_channels, in_channels-self.im_kernel_num,
                                                       self.kernel_size[0], self.kernel_size[1]))
            nn.init.kaiming_normal_(self.weights_common, mode='fan_out', nonlinearity='relu')
        if self.im_kernel_num>0:
            self.weights_imbalance = nn.Parameter(torch.randn(1+im_class_num,out_channels, self.im_kernel_num,
                                                          self.kernel_size[0], self.kernel_size[1]))
            for i in range(1+im_class_num):
                nn.init.kaiming_normal_(self.weights_imbalance[i], mode='fan_out', nonlinearity='relu')
        if tasks_num>1:
            if self.im_kernel_num>0:
                self.weights_multi = nn.Parameter(torch.randn(tasks_num-1, out_channels,self.im_kernel_num,
                                                          self.kernel_size[0], self.kernel_size[1]))
                for i in range(tasks_num-1):
                    nn.init.kaiming_normal_(self.weights_multi[i], mode='fan_out', nonlinearity='relu')
        self.bias = bias
        if bias:
            self.bias_im = nn.Parameter(torch.zeros(im_class_num+1,out_channels))
            self.bias_multi = nn.Parameter(torch.zeros(tasks_num-1,out_channels))
            self.bias_common = nn.Parameter(torch.zeros(out_channels))

        # self.bias = bias
    def forward(self, x, conv_attention=None,task_indx=1):
        # conv_attention: N, C
        # ipdb.set_trace()


        N, C,  H, W = x.shape
        if task_indx>0:
            if not self.im_kernel_num:
                weight = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1)
            elif not (self.in_channels-self.im_kernel_num):
                weight =self.weights_multi[task_indx-1].unsqueeze(0).expand(N, -1, -1, -1, -1)
            else:
                weight_multi = self.weights_multi[task_indx-1].unsqueeze(0).expand(N, -1, -1, -1, -1)
                weight_common = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1)
                weight = torch.cat((weight_common, weight_multi), dim=2)
                weight_ = torch.cat((self.weights_common, self.weights_multi[task_indx-1]), dim=1)
        else:
            _ , C_atten= conv_attention.shape
            if not self.im_kernel_num:
                weight = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1)
                # print("2Din only use common weights")
            elif not (self.in_channels-self.im_kernel_num):
                _,C_out, C_in, K_H, K_W = self.weights_imbalance.shape
                expanded_weights_im = self.weights_imbalance.unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
                expanded_conv_attention = conv_attention.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(-1,-1,C_out,self.im_kernel_num, K_H, K_W)
                weight_im = expanded_weights_im * expanded_conv_attention
                weight = weight_im.sum(dim=1)
                # print( "2Din only use Imbalance weights" )
            else:
                # print("\033[91m" +"2Din  use mix weights" + "\033[0m")
                _,C_out, C_in, K_H, K_W = self.weights_imbalance.shape
                expanded_weights_im = self.weights_imbalance.unsqueeze(0).expand(N, -1, -1, -1, -1,-1)
                expanded_conv_attention = conv_attention.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(-1,-1,C_out,self.im_kernel_num, K_H, K_W)
                weight_im = expanded_weights_im * expanded_conv_attention
                weight_im = weight_im.sum(dim=1)
                # weight = expanded_weights
                # ipdb.set_trace()
                weight_common = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1)
                weight = torch.cat((weight_common, weight_im), dim=2)
        # ipdb.set_trace()

        _, C_out,C_in, K_H, K_W = weight.shape
        x = x.view(1, -1,  H, W)
        weight = weight.reshape(N*C_out, C_in, K_H, K_W).contiguous()
        # aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size, self.kernel_size)
        if self.bias :
            if task_indx>0:
                aggregate_bias = (1-self.im_kernel_ratio)*self.bias_common + self.im_kernel_ratio*self.bias_multi[task_indx-1]
                aggregate_bias = aggregate_bias.unsqueeze(0).expand(N,-1).reshape(-1)
            else:
                aggregate_bias = (1-self.im_kernel_ratio)*self.bias_common.unsqueeze(0).expand(N,-1) + self.im_kernel_ratio*torch.mm(conv_attention, self.bias_im)
                aggregate_bias = aggregate_bias.reshape(-1)
                #N,C_out_im
            output = F.conv2d(x, weight=weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,groups=N)
        else:
            output = F.conv2d(x, weight=weight, bias=None, stride=self.stride, padding=self.padding, groups=N)

        output = output.view(N, self.out_channels,  output.size(-2), output.size(-1))


    # weight = weight.to(torch.float64)
        #
        # # 添加padding
        # input_padded = torch.nn.functional.pad(x, ( self.padding[1], self.padding[1], self.padding[0], self.padding[0])).to(torch.float64)
        # H_, W_ = input_padded.shape[2:]
        # new_stride = (
        #     C * H_ * W_,
        #     H_ * W_,
        #     self.stride[0]*W_,
        #     self.stride[1],
        #     W_,
        #     1
        # )
        # # input_strided = input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1, (H_- K_H) // self.stride[1] + 1,(W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
        # #                      new_stride)
        # #
        #
        #
        #
        # #
        # # # 执行卷积操作
        # # output = torch.einsum('bihwkl,boikl->bohw', input_padded.as_strided((N, C,(H_- K_H) // self.stride[0] + 1,
        # #                                                                      (W_-K_W)//self.stride[1]+1,K_H, K_W),new_stride), weight).to(torch.float32)
        # # # ipdb.set_trace()
        # #
        # # # 添加偏置项（如果有）
        # # if self.bias:
        # #     if task_indx>0:
        # #         output += self.bias_multi[task_indx-1][None, :, None, None]
        # #     else:
        # #         bias_im = self.bias_im.unsqueeze(0).expand(N, -1,-1)*conv_attention.unsqueeze(2).expand(-1,-1,self.out_channels)
        # #         output += bias_im.mean(dim=1).squeeze(1)[:,:,None,None]
        # if self.bias:
        #     output = F.conv2d(x, self.weights_common, bias=self.bias_common, stride=self.stride, padding=self.padding)
        # else:
        #     output = F.conv2d(x, self.weights_common, bias=None, stride=self.stride, padding=self.padding)

        return output



class MIAOutFC(nn.Module):
    def __init__(self,in_channels, out_channels,im_class_num=1,tasks_num =2,im_kernel_ratio=0.5, bias=True):
        super(MIAOutFC, self).__init__()
        ##MIAOutFC means Multitask-available Fully Connected layer with Imbalance Aware in Output channel

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.im_kernel_num = int(out_channels*im_kernel_ratio)
        self.weights_common = nn.Parameter(torch.randn(out_channels-self.im_kernel_num, in_channels))
        self.weights_imbalance = nn.Parameter(torch.randn(int(self.im_kernel_num*(1+im_class_num)), in_channels))
        if tasks_num>1:
            self.weights_multi = nn.Parameter(torch.randn(tasks_num-1,self.im_kernel_num, in_channels))
        self.bias = bias
        if bias:
            self.bias_im = nn.Parameter(torch.randn(im_class_num+1,out_channels))
            self.bias_multi = nn.Parameter(torch.randn(tasks_num-1,out_channels))

        # self.bias = bias
    def forward(self, x, fc_attention=None,task_indx =1):
        # conv_attention: N, C
        N, C = x.shape
        if task_indx>0:
            weight_multi = self.weights_multi[task_indx-1].unsqueeze(0).expand(N, -1, -1)
            weight_common = self.weights_common.unsqueeze(0).expand(N, -1, -1)
            weight = torch.cat((weight_common, weight_multi), dim=1)
        else:
            _ , C_atten= fc_attention.shape
            _, C_in = self.weights_imbalance.shape
            expanded_weights_im = self.weights_imbalance.unsqueeze(0).expand(N, -1, -1, -1, -1).reshape(N, C_atten,self.im_kernel_num,C_in)

            expanded_attention = fc_attention.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(-1, -1,self.im_kernel_num, C_in)
            weight_im = expanded_weights_im * expanded_attention
            weight_im = weight_im.mean(dim=1)
            weight_common = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1)
            weight = torch.cat((weight_common, weight_im), dim=1)
        out = torch.einsum('bi,boi->bo', x, weight)
        if self.bias:
            if task_indx>0:
                out += self.bias_multi[task_indx-1][None, :]
            else:
                bias_im = self.bias_im.unsqueeze(0).expand(N, -1,-1)*fc_attention.unsqueeze(2).expand(-1,-1,self.out_channels)
                out += bias_im.mean(dim=1).squeeze(1)
class IAInConv2D(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,im_class_num=1,im_kernel_ratio=0.5, stride=1, padding=0, bias=True):
        super(IAInConv2D, self).__init__()
        #IAInConv2D means Convolution 2D with Imbalance Aware in Input channel

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = to_2tuple(kernel_size)
        if isinstance(stride, int):
            stride = to_2tuple(stride)
        if isinstance(padding, int):
            padding = to_2tuple(padding)
        self.fan_out = out_channels * kernel_size[0] * kernel_size[1]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.im_kernel_num = int(in_channels*im_kernel_ratio)
        self.weights_common = nn.Parameter(torch.randn(out_channels, in_channels-self.im_kernel_num,
                                                       self.kernel_size[0], self.kernel_size[1]).normal_(0, math.sqrt(2. / self.fan_out)))
        self.weights_imbalance = nn.Parameter(torch.randn(out_channels, int(self.im_kernel_num*(1+im_class_num)),
                                                          self.kernel_size[0], self.kernel_size[1]).normal_(0, math.sqrt(2. / self.fan_out)))
        self.bias = bias
        if bias:
            self.bias_im = nn.Parameter(torch.zeros(im_class_num+1,out_channels))

        # self.bias = bias
    def forward(self, x, conv_attention):
        # conv_attention: N, C


        N, C,  H, W = x.shape
        _ , C_atten= conv_attention.shape
        C_out, C_in, K_H, K_W = self.weights_imbalance.shape
        expanded_weights_im = self.weights_imbalance.unsqueeze(0).expand(N, -1, -1, -1, -1).reshape(N,C_out,C_atten,self.im_kernel_num, K_H, K_W)
        expanded_conv_attention = conv_attention.unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(-1,C_out,-1,self.im_kernel_num, K_H, K_W)
        weight_im = expanded_weights_im * expanded_conv_attention
        weight_im = weight_im.mean(dim=2)
        # weight = expanded_weights
        # ipdb.set_trace()
        weight_common = self.weights_common.unsqueeze(0).expand(N, -1, -1, -1, -1)
        weight = torch.cat((weight_common, weight_im), dim=2).to(torch.float64)
        # ipdb.set_trace()
        _, C_out, _, _, _ = weight.shape

        # 添加padding
        input_padded = torch.nn.functional.pad(x, ( self.padding[1], self.padding[1], self.padding[0], self.padding[0])).to(torch.float64)
        H_, W_ = input_padded.shape[2:]
        new_stride = (
            C * H_ * W_,
            H_ * W_,
            self.stride[0]*W_,
            self.stride[1],
            W_,
            1
        )
        # input_strided = input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1, (H_- K_H) // self.stride[1] + 1,(W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
        #                      new_stride)
        #
        # 执行卷积操作
        output = torch.einsum('bihwkl,boikl->bohw', input_padded.as_strided((N, C,(H_- K_H) // self.stride[0] + 1,
                                                                             (W_-K_W)//self.stride[1]+1,K_H, K_W),new_stride), weight).to(torch.float32)
        # ipdb.set_trace()

        # 添加偏置项（如果有）
        if self.bias:
            bias_im = self.bias_im.unsqueeze(0).expand(N, -1,-1)*conv_attention.unsqueeze(2).expand(-1,-1,self.out_channels)
            output += bias_im.mean(dim=1).squeeze(1)[:,:,None,None]

        return output






class Einsum_check(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=3, stride=2, padding=1, bias=True):
        super(Einsum_check, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(stride, int):
            stride = to_3tuple(stride)
        if isinstance(padding, int):
            padding = to_3tuple(padding)
        if isinstance(kernel_size, int):
            kernel_size = to_3tuple(kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.weights_3d = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]))
        self.weights_3d = nn.Parameter(torch.randn(128, 64, 3, 4,5))
        # self.weights_3d = nn.Parameter(torch.randn(3, 1, 3, 3, 3))
        self.weights_2d = nn.Parameter(torch.randn(128, 64, 3,3))
        if bias:
            self.bias_3d = nn.Parameter(torch.zeros(128))
            self.bias_2d = nn.Parameter(torch.zeros(128))

        self.check = False
        # self.bias = bias
    def forward(self, x):
        #unfinish
        # ipdb.set_trace()
        # input = torch.randn(2,1,3,5,5)
        # input = torch.randn(8,1,5,,10).cuda()
        # input = torch.randn(8,1,5,10,10).cuda()
        x_ = torch.randn(8,64,15,96,96).cuda()
        out3d_f = F.conv3d(x_, self.weights_3d, self.bias_3d, self.stride, self.padding)
        n, c,d_in, h_in, w_in = x_.shape
        d, c, k, j,l = self.weights_3d.shape
        # x_pad = torch.zeros(n, c, h_in+2*self.padding[0], w_in+2*self.padding[1]).cuda   # 对输入进行补零操作
        x_pad = torch.nn.functional.pad(x_, ( self.padding[2], self.padding[2],self.padding[1], self.padding[1], self.padding[0], self.padding[0]))

        x_pad = x_pad.unfold(2, k, self.stride[0])
        x_pad = x_pad.unfold(3, j, self.stride[1])
        x_pad = x_pad.unfold(4, l, self.stride[2])
        # 按照滑动窗展开
        out3d = torch.einsum(                          # 按照滑动窗相乘，
            'ncdhwkjl,ockjl->nodhw',                    # 并将所有输入通道卷积结果累加
            x_pad, self.weights_3d)
        out3d = out3d + self.bias_3d.view(1, -1, 1, 1,1)          # 添加偏置值

        if torch.allclose(out3d, out3d_f,rtol=1e-3,atol=1e-3):
            print("In Einsum_check Custom convolution 3d matches PyTorch built-in convolution.")
            self.check = True

        else:
            print("\033[91m" + "In Einsum_check Custom convolution 3d does not match PyTorch built-in convolution." + "\033[0m")
            self.check = False




        x = torch.randn(8,64,128,96).cuda()
        out_f = F.conv2d(x, self.weights_2d, self.bias_2d, self.stride[0], self.padding[0])
        n, c, h_in, w_in = x.shape
        d, c, k, j = self.weights_2d.shape
        # x_pad = torch.zeros(n, c, h_in+2*self.padding[0], w_in+2*self.padding[1]).cuda   # 对输入进行补零操作
        x_pad = torch.nn.functional.pad(x, ( self.padding[0], self.padding[0], self.padding[0], self.padding[0]))

        x_pad = x_pad.unfold(2, k, self.stride[0])
        x_pad = x_pad.unfold(3, j, self.stride[0])        # 按照滑动窗展开
        out = torch.einsum(                          # 按照滑动窗相乘，
            'nchwkj,dckj->ndhw',                    # 并将所有输入通道卷积结果累加
            x_pad, self.weights_2d)
        out = out + self.bias_2d.view(1, -1, 1, 1)          # 添加偏置值

        if torch.allclose(out, out_f,rtol=1e-4,atol=1e-4):
            print("In Einsum_check Custom convolution 2d matches PyTorch built-in convolution.")
            self.check = True

        else:
            print("\033[91m" + "In Einsum_check Custom convolution 2d does not match PyTorch built-in convolution." + "\033[0m")
            self.check = False


        return out




        # out_3d_1 = F.conv3d(x, self.weights_3d,  stride=self.stride, padding=self.padding)
        #
        # # out_fc_1 = F.linear(x[:,0,0,:,:].reshape(x.shape[0],-1), self.weights_2d.reshape(self.weights_2d.shape[0],-1).t(), bias=self.bias_2d)
        # N, C, D, H, W = x.shape
        # expanded_weights = self.weights_3d.unsqueeze(0).expand(N, -1, -1, -1, -1, -1)
        # weight = expanded_weights
        # N, C_out, _, K_D, K_H, K_W = weight.shape
        #
        # # 添加padding
        # input_padded = torch.nn.functional.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        # D_, H_, W_ = input_padded.shape[2:]
        #
        # new_stride = (
        #     C* D_ * H_ * W_,
        #     D_ * H_ * W_,
        #     self.stride[0] * W_ * H_,
        #     self.stride[1]*W_,
        #     self.stride[2],
        #     W_*H_,
        #     W_,
        #     1
        # )
        # input_strided = input_padded.as_strided((N, C, (D_ - K_D) // self.stride[0] + 1, (H_- K_H) // self.stride[1] + 1,(W_-K_W)//self.stride[2]+1, K_D, K_H, K_W),
        #                                         new_stride)
        # # 反序
        #
        # out_3d_2 = torch.einsum('bidhwjkl,boijkl->bodhw', input_strided, weight)
        # out_3d_3 = torch.einsum('bidhwjkl,oijkl->bodhw', input_strided, self.weights_3d)
        #
        # # if self.bias_3d is not None:
        # #     out_3d_2 += self.bias_3d[None, :, None, None, None]
        # #     out_3d_3 += self.bias_3d[None, :, None, None, None]
        # # ipdb.set_trace()
        # if torch.allclose(out_3d_1, out_3d_2,rtol=1e-3):
        #     print("In Einsum_check Custom convolution 2 matches PyTorch built-in convolution.")
        #     self.check = True
        #
        # else:
        #     print("\033[91m" + "In Einsum_check Custom convolution 2 does not match PyTorch built-in convolution." + "\033[0m")
        #     self.check = False
        # if torch.allclose(out_3d_1, out_3d_3,rtol=1e-3):
        #     print("In Einsum_check Custom convolution 3 matches PyTorch built-in convolution.")
        #     self.check = True
        # else:
        #     print("\033[91m" + "In Einsum_check Custom convolution 3 does not match PyTorch built-in convolution." + "\033[0m")
        #     self.check = False
        # ipdb.set_trace()
        #
        #
        # return self.check
class ClassficationHead(nn.Module):
    def __init__(self, in_channels,hidden_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = x.flatten(start_dim=1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        # ipdb.set_trace()

        return out

class AttentionConvNet(nn.Module):

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
        # self.encoder1 = ResBlock3D(input_channels, 64, stride=2, p=0.5)
        # self.encoder1 = AttentionConvBlock3d(input_channels, 64, [15, img_size, img_size],stride=1, stride_attention=[[2,2,2],[2,2,2]],stride_attenconv=2)
        self.encoder1 = AttentionConvBlock3d_4stride(input_channels, 64, [15, img_size, img_size],stride=1, stride_attention=[[1,2,2],[1,2,2],[2,2,2],[2,2,2]],stride_attenconv=2)
        # self.encoder2 = ResBlock3D(64, 128, stride=2, p=0.5)
        # self.encoder2 = AttentionConvBlock3d(64, 128, [7, img_size//2, img_size//2],stride=1, stride_attention=[[1,2,2],[2,2,2]],stride_attenconv=2)
        self.encoder2 = AttentionConvBlock3d_4stride(64, 128, [7, img_size//2, img_size//2],stride=1, stride_attention=[[1,2,2],[1,2,2],[1,2,2],[2,2,2]],stride_attenconv=2)
        # self.encoder3 = ResBlock3D(128, 256, stride=2, p=0.5)
        # self.encoder3 = AttentionConvBlock3d(128, 256, [3, img_size//4, img_size//4],stride=1, stride_attention=[[1,2,2],[1,2,2]],stride_attenconv=2)
        self.encoder3 = AttentionConvBlock3d_4stride(128, 256, [3, img_size//4, img_size//4],stride=1, stride_attention=[[1,1,1],[1,2,2],[1,2,2],[1,2,2]],stride_attenconv=2)
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
        # ipdb.set_trace()
        idx2 = out.shape[2]//2
        t2 = out[:,:,idx2,:,:]
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out = self.encoder3(out)
        # ipdb.set_trace()
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



class MIAResBlock3D(nn.Module):
    """ residual block """
    def __init__(self, in_channels, out_channels, stride=1, p=0.5, downsample=None,im_class_num=1,task_num=2,im_kernel_ratio=0.5):
        super().__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(in_channels)
        padding = 1 if stride == 1 else (0, 1, 1)
        self.conv1 = MIAOutConv3D(in_channels, out_channels,3,im_class_num=im_class_num,tasks_num=task_num,
                                  im_kernel_ratio=im_kernel_ratio,stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = MIAInConv3D(out_channels, out_channels,3,im_class_num=im_class_num,tasks_num=task_num,
                                  im_kernel_ratio=im_kernel_ratio,stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout3d(p=p)

        if stride != 1 or in_channels != out_channels:
            self.downsample = MIAOutConv3D(in_channels, out_channels,
                          3,im_class_num=im_class_num,tasks_num=task_num,
                             im_kernel_ratio=im_kernel_ratio,stride=stride, bias=False, padding=padding)
            self.downsample_1 = MIAInConv3D(out_channels, out_channels,
                             3,im_class_num=im_class_num,tasks_num=task_num,
                             im_kernel_ratio=im_kernel_ratio,stride=1, bias=False, padding=1)

            self.bn_down = nn.BatchNorm3d(out_channels)

    def forward(self, x,attention=0,task_indx=1):
        residual = x
        # print("input residual size: {}".format(residual.size()))
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out,attention,task_indx)
        # out = self.conv2(out,attention,task_indx)
        # ipdb.set_trace()

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out,attention,task_indx)
        out = self.dp(out)
        if self.downsample is not None:
            residual = self.downsample(residual,attention,task_indx)
            # ipdb.set_trace()
            residual = self.downsample_1(residual,attention,task_indx)

            residual = self.bn_down(residual)
            # print("output residual size: {}".format(residual.size()))
        # print("output size: {}".format(out.size()))
        # ipdb.set_trace()
        out += residual

        return out

class MIAResBlock2D(nn.Module):
    """ residual block """
    def __init__(self, in_channels, out_channels, stride=1, p=0.5, downsample=None,im_class_num=1,task_num=2,im_kernel_ratio=0.5):
        super().__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_channels)
        padding = 1 if stride == 1 else ( 1, 1)
        self.conv1 = MIAOutConv2D(in_channels, out_channels,3,im_class_num=im_class_num,tasks_num=task_num,
                                  im_kernel_ratio=im_kernel_ratio,stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = MIAInConv2D(out_channels, out_channels,3,im_class_num=im_class_num,tasks_num=task_num,
                                  im_kernel_ratio=im_kernel_ratio,stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout2d(p=p)

        if stride != 1 or in_channels != out_channels:
            self.downsample = MIAOutConv2D(in_channels, out_channels,
                          3,im_class_num=im_class_num,tasks_num=task_num,
                             im_kernel_ratio=im_kernel_ratio,stride=stride, bias=False, padding=padding)
            self.downsample_1 = MIAInConv2D(out_channels, out_channels,
                             3,im_class_num=im_class_num,tasks_num=task_num,
                             im_kernel_ratio=im_kernel_ratio,stride=1, bias=False, padding=1)
            self.bn_down = nn.BatchNorm2d(out_channels)


    def forward(self, x,attention=0,task_indx=1):
        residual = x
        # print("input residual size: {}".format(residual.size()))
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out,attention,task_indx)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out,attention,task_indx)
        out = self.dp(out)
        if self.downsample is not None:
            residual = self.downsample(residual,attention,task_indx)
            residual = self.downsample_1(residual,attention,task_indx)
            residual = self.bn_down(residual)
            # print("output residual size: {}".format(residual.size()))
        # print("output size: {}".format(out.size()))
        out += residual

        return out
class MIAUnext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self,  num_classes, num_imbalance=1,input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
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
        # self.encoder1 = ResBlock3D(input_channels, 64, stride=2, p=0.5)
        # self.einscheck2 = Einsum_check(64, 128,stride=[1,2,2],padding=[0,1,1])
        # self.encoder2 = ResBlock3D(64, 128, stride=2, p=0.5)
        # self.encoder3 = ResBlock3D(128, 256, stride=2, p=0.5)
        # self.einscheck1 = Einsum_check(input_channels, 64,kernel_size=[3,6,6], stride=[1,2,2],padding=[0,1,1])
        self.encoder1 = MIAResBlock3D(input_channels, 64, stride=2, p=0.5,im_kernel_ratio=0.7)
        # # self.encoder1 = AttentionConvBlock3d(input_channels, 64, [15, img_size, img_size],stride=1, stride_attention=[[2,2,2],[2,2,2]],stride_attenconv=2)
        # # self.encoder1 = AttentionConvBlock3d_4stride(input_channels, 64, [15, img_size, img_size],stride=1, stride_attention=[[1,2,2],[1,2,2],[2,2,2],[2,2,2]],stride_attenconv=2)
        self.encoder2 = MIAResBlock3D(64, 128, stride=2, p=0.5,im_kernel_ratio=0.7)
        # # self.encoder2 = AttentionConvBlock3d(64, 128, [7, img_size//2, img_size//2],stride=1, stride_attention=[[1,2,2],[2,2,2]],stride_attenconv=2)
        # # self.encoder2 = AttentionConvBlock3d_4stride(64, 128, [7, img_size//2, img_size//2],stride=1, stride_attention=[[1,2,2],[1,2,2],[1,2,2],[2,2,2]],stride_attenconv=2)
        self.encoder3 = MIAResBlock3D(128, 256, stride=2, p=0.5,im_kernel_ratio=0.7)
        # self.encoder3 = AttentionConvBlock3d(128, 256, [3, img_size//4, img_size//4],stride=1, stride_attention=[[1,2,2],[1,2,2]],stride_attenconv=2)
        # self.encoder3 = AttentionConvBlock3d_4stride(128, 256, [3, img_size//4, img_size//4],stride=1, stride_attention=[[1,1,1],[1,2,2],[1,2,2],[1,2,2]],stride_attenconv=2)
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

        # self.decoder1 = nn.Conv2d(512, 384, 3, stride=1,padding=1)
        # self.decoder2 =   nn.Conv2d(384, 256, 3, stride=1, padding=1)
        # self.decoder3 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        # self.decoder4 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoder5 =   nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.decoder1 =   MIAOutConv2D(512, 384, 3, stride=1,padding=1,im_kernel_ratio=0.7)
        self.decoder2 =   MIAOutConv2D(384, 256, 3, stride=1, padding=1,im_kernel_ratio=0.7)
        self.decoder3 =   MIAOutConv2D(256, 128, 3, stride=1, padding=1,im_kernel_ratio=0.7)
        self.decoder4 =   MIAOutConv2D(128, 64, 3, stride=1, padding=1,im_kernel_ratio=0.7)
        self.decoder5 =   MIAOutConv2D(64, 64, 3, stride=1, padding=1,im_kernel_ratio=0.7)
        # self.decoder1 = nn.Conv2d(512, 384, 3, stride=1,padding=1)
        # self.decoder2 =   nn.Conv2d(384, 256, 3, stride=1, padding=1)
        # self.decoder3 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        # self.decoder4 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoder5 =   nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(384)
        self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(128)
        self.dbn4 = nn.BatchNorm2d(64)

        # self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = IAInConv2D(64, num_classes, kernel_size=1,im_kernel_ratio=0.7)
        self.soft = nn.Softmax(dim =1)
        self.classhead = ClassficationHead(embed_dims[2]*9,embed_dims[2]//4, num_imbalance+1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fl =nn.Sequential( nn.Linear(4*24*24, 32),
                                nn.Linear(32, 2))

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        # out = self.encoder1(x)
        # _ = self.einscheck1(x)
        # ipdb.set_trace()
        out = self.encoder1(x,task_indx = 1)
        # ipdb.set_trace()

        #get middle tensor
        idx1 = out.shape[2]//2
        t1 = out[:,:,idx1,:,:]

        ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        # _=self.einscheck2(out)
        # out = self.encoder2(out)
        out = self.encoder2(out,task_indx = 1)
        # ipdb.set_trace()
        idx2 = out.shape[2]//2
        t2 = out[:,:,idx2,:,:]
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        # out = self.encoder3(out)
        out = self.encoder3(out,task_indx = 1)
        # ipdb.set_trace()
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
        # ipdb.set_trace()
        out = self.norm4(out)
        # ipdb.set_trace()
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        # ipdb.set_trace()

        # out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out,task_indx = 1)),scale_factor=(2,2),mode ='bilinear'))

        out = torch.add(out,t4)
        # print("stage4:",out.shape)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out,task_indx = 1)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out,task_indx = 1)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        # out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out,task_indx = 1)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        # out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.decoder5(out,task_indx = 1),scale_factor=(2,2),mode ='bilinear'))

        # classification head
        out = self.layer1(out)
        # print("out_layer2:", out.shape)
        out = self.layer2(out)
        # out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        # ipdb.set_trace()
        # print("out_layerfl:", out.shape)
        imbalance_attention = self.soft(self.fl(out))
        # return imbalance_attention
        # ipdb.set_trace()

        B_1 = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        out_1 = self.encoder1(x,imbalance_attention,0)
        # ipdb.set_trace()

        #get middle tensor
        idx1_1 = out_1.shape[2]//2
        t1_1 = out_1[:,:,idx1_1,:,:]

        ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out_1 = self.encoder2(out_1,imbalance_attention,0)
        # ipdb.set_trace()
        idx2_1 = out_1.shape[2]//2
        t2_1 = out_1[:,:,idx2_1,:,:]
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out_1 = self.encoder3(out_1,imbalance_attention,0)
        # ipdb.set_trace()
        idx3_1 = out_1.shape[2]//2
        t3_1 = out_1[:,:,idx3_1,:,:]
        out_1 = t3_1

        # ipdb.set_trace()


        ### Tokenized MLP Stage
        ### Stage 4

        out_1,H_1,W_1 = self.patch_embed3(out_1)
        for i, blk in enumerate(self.block1):
            out_1 = blk(out_1, H_1, W_1)
        out_1 = self.norm3(out_1)
        out_1 = out_1.reshape(B_1, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        t4_1 = out_1

        ### Bottleneck

        out_1 ,H_1,W_1= self.patch_embed4(out_1)
        for i, blk in enumerate(self.block2):
            out_1 = blk(out_1, H_1, W_1)
        # ipdb.set_trace()
        out_1 = self.norm4(out_1)
        # ipdb.set_trace()
        out_1 = out_1.reshape(B_1, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        # ipdb.set_trace()

        out_1 = F.relu(F.interpolate(self.dbn1(self.decoder1(out_1,imbalance_attention,0)),scale_factor=(2,2),mode ='bilinear'))

        out_1 = torch.add(out_1,t4_1)
        # print("stage4:",out.shape)
        _,_,H_1,W_1 = out_1.shape
        out_1 = out_1.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out_1 = blk(out_1, H_1, W_1)

        ### Stage 3

        out_1 = self.dnorm3(out_1)
        out_1 = out_1.reshape(B_1, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        out_1 = F.relu(F.interpolate(self.dbn2(self.decoder2(out_1,imbalance_attention,0)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t3_1)
        _,_,H_1,W_1 = out_1.shape
        out_1 = out_1.flatten(2).transpose(1,2)

        for i, blk in enumerate(self.dblock2):
            out_1 = blk(out_1, H_1, W_1)

        out_1 = self.dnorm4(out_1)
        out_1 = out_1.reshape(B_1, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()

        out_1 = F.relu(F.interpolate(self.dbn3(self.decoder3(out_1,imbalance_attention,0)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t2_1)
        out_1 = F.relu(F.interpolate(self.dbn4(self.decoder4(out_1,imbalance_attention,0)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t1_1)
        out_1 = F.relu(F.interpolate(self.decoder5(out_1,imbalance_attention,0),scale_factor=(2,2),mode ='bilinear'))



        return self.final(out_1,imbalance_attention),out_1,imbalance_attention


class MIAUnext_test(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self,  num_classes, num_imbalance=1,input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
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
        # self.encoder1 = ResBlock3D(input_channels, 64, stride=2, p=0.5)
        # self.einscheck2 = Einsum_check(64, 128,stride=[1,2,2],padding=[0,1,1])
        # self.encoder2 = ResBlock3D(64, 128, stride=2, p=0.5)
        # self.encoder3 = ResBlock3D(128, 256, stride=2, p=0.5)
        # self.einscheck1 = Einsum_check(input_channels, 64,kernel_size=[3,6,6], stride=[1,2,2],padding=[0,1,1])
        self.encoder1 = MIAResBlock3D(input_channels, 64, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # # self.encoder1 = AttentionConvBlock3d(input_channels, 64, [15, img_size, img_size],stride=1, stride_attention=[[2,2,2],[2,2,2]],stride_attenconv=2)
        # # self.encoder1 = AttentionConvBlock3d_4stride(input_channels, 64, [15, img_size, img_size],stride=1, stride_attention=[[1,2,2],[1,2,2],[2,2,2],[2,2,2]],stride_attenconv=2)
        self.encoder2 = MIAResBlock3D(64, 128, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # # self.encoder2 = AttentionConvBlock3d(64, 128, [7, img_size//2, img_size//2],stride=1, stride_attention=[[1,2,2],[2,2,2]],stride_attenconv=2)
        # # self.encoder2 = AttentionConvBlock3d_4stride(64, 128, [7, img_size//2, img_size//2],stride=1, stride_attention=[[1,2,2],[1,2,2],[1,2,2],[2,2,2]],stride_attenconv=2)
        self.encoder3 = MIAResBlock3D(128, 256, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # self.encoder3 = AttentionConvBlock3d(128, 256, [3, img_size//4, img_size//4],stride=1, stride_attention=[[1,2,2],[1,2,2]],stride_attenconv=2)
        # self.encoder3 = AttentionConvBlock3d_4stride(128, 256, [3, img_size//4, img_size//4],stride=1, stride_attention=[[1,1,1],[1,2,2],[1,2,2],[1,2,2]],stride_attenconv=2)

        self.decoder1 =   MIAResBlock2D(256, 128, stride=1, im_kernel_ratio=cfg.im_kernel_ratio)
        self.decoder2 =   MIAResBlock2D(128, 64, stride=1, im_kernel_ratio=cfg.im_kernel_ratio)
        self.decoder3 =   MIAResBlock2D(64, 64, stride=1, im_kernel_ratio=cfg.im_kernel_ratio)
        # self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(128)
        self.dbn4 = nn.BatchNorm2d(64)


        # self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_1 =MIAOutConv2D(64, 64, kernel_size=1,im_kernel_ratio=cfg.im_kernel_ratio)
        self.final_2 =MIAInConv2D(64, num_classes, kernel_size=1,im_kernel_ratio=cfg.im_kernel_ratio)
        self.soft = nn.Softmax(dim =1)
        self.classhead = ClassficationHead(embed_dims[2]*9,embed_dims[2]//4, num_imbalance+1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fl =nn.Sequential( nn.Linear(4*24*24, 32),
                                nn.Linear(32, 2))

    def forward(self, x,imbalance_attention):

        # return imbalance_attention
        # ipdb.set_trace()

        B_1 = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        out_1 = self.encoder1(x,imbalance_attention,0)
        # ipdb.set_trace()

        #get middle tensor
        idx1_1 = out_1.shape[2]//2
        t1_1 = out_1[:,:,idx1_1,:,:]

        ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out_1 = self.encoder2(out_1,imbalance_attention,0)
        # ipdb.set_trace()
        idx2_1 = out_1.shape[2]//2
        t2_1 = out_1[:,:,idx2_1,:,:]
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out_1 = self.encoder3(out_1,imbalance_attention,0)
        # ipdb.set_trace()
        idx3_1 = out_1.shape[2]//2
        t3_1 = out_1[:,:,idx3_1,:,:]
        out_1 = t3_1

        # ipdb.set_trace()



        out_1 = F.relu(F.interpolate(self.dbn3(self.decoder1(out_1,imbalance_attention,0)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t2_1)
        out_1 = F.relu(F.interpolate(self.dbn4(self.decoder2(out_1,imbalance_attention,0)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t1_1)
        out_1 = F.relu(F.interpolate(self.decoder3(out_1,imbalance_attention,0),scale_factor=(2,2),mode ='bilinear'))
        cnn_feature = out_1
        final_1 = self.final_1(out_1,imbalance_attention,0)
        out = self.final_2(final_1,imbalance_attention,0)



        return out,cnn_feature,imbalance_attention

class HybridResunet_test(nn.Module):
    def __init__(self,  num_classes, num_imbalance=1,input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
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
        self.encoder = ResBlock3D(input_channels, 64, stride=1, p=0.5) #15 96->15 96
        self.encoder1 = ResBlock3D(64, 64, stride=2, p=0.5) # 15 96->7 ,48
        # self.einscheck2 = Einsum_check(64, 128,stride=[1,2,2],padding=[0,1,1])
        self.encoder2 = ResBlock3D(64, 128, stride=2, p=0.5)    # 7 48->3 24
        self.bottleneck = ResBlock3D(128, 256, stride=2, p=0.5) # 3 24->1 12
        self.convbottle = nn.Conv3d(256, 256, 1, stride=1, padding=0) #1 12->1 12

        # self.einscheck1 = Einsum_check(input_channels, 64,kernel_size=[3,6,6], stride=[1,2,2],padding=[0,1,1])
        # self.encoder1 = MIAResBlock3D(input_channels, 64, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # # self.encoder1 = AttentionConvBlock3d(input_channels, 64, [15, img_size, img_size],stride=1, stride_attention=[[2,2,2],[2,2,2]],stride_attenconv=2)
        # # self.encoder1 = AttentionConvBlock3d_4stride(input_channels, 64, [15, img_size, img_size],stride=1, stride_attention=[[1,2,2],[1,2,2],[2,2,2],[2,2,2]],stride_attenconv=2)
        # self.encoder2 = MIAResBlock3D(64, 128, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # # self.encoder2 = AttentionConvBlock3d(64, 128, [7, img_size//2, img_size//2],stride=1, stride_attention=[[1,2,2],[2,2,2]],stride_attenconv=2)
        # # self.encoder2 = AttentionConvBlock3d_4stride(64, 128, [7, img_size//2, img_size//2],stride=1, stride_attention=[[1,2,2],[1,2,2],[1,2,2],[2,2,2]],stride_attenconv=2)
        # self.encoder3 = MIAResBlock3D(128, 256, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # self.encoder3 = AttentionConvBlock3d(128, 256, [3, img_size//4, img_size//4],stride=1, stride_attention=[[1,2,2],[1,2,2]],stride_attenconv=2)
        # self.encoder3 = AttentionConvBlock3d_4stride(128, 256, [3, img_size//4, img_size//4],stride=1, stride_attention=[[1,1,1],[1,2,2],[1,2,2],[1,2,2]],stride_attenconv=2)

        self.decoder1 =   ResBlock2D(256, 128, stride=1,p=0.5 ) # 256 1 12-> 128 1 24
        self.decoder2 =   ResBlock2D(128, 64, stride=1, p =0.5) #128 1 24->64 1 48
        self.decoder3 =   ResBlock2D(64, 64, stride=1,p=0.5 ) # 64 1 48->64  1 96
        self.decoder4 =   ResBlock2D(64, 64, stride=1,p=0.5 ) #64  1 96-> 64 1 96
        # self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(128)
        self.dbn4 = nn.BatchNorm2d(64)
        self.dbn5 = nn.BatchNorm2d(64)


        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        # self.final_1 =MIAOutConv2D(64, 64, kernel_size=1,im_kernel_ratio=cfg.im_kernel_ratio)
        # self.final_2 =MIAInConv2D(64, num_classes, kernel_size=1,im_kernel_ratio=cfg.im_kernel_ratio)
        self.soft = nn.Softmax(dim =1)


    def forward(self, x,imbalance_attention):

        # return imbalance_attention
        # ipdb.set_trace()

        B_1 = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        out_1 = self.encoder(x)
        idx1_1 = out_1.shape[2]//2
        t0_1 = out_1[:,:,idx1_1,:,:]

        out_1 = self.encoder1(out_1)
        # ipdb.set_trace()

        #get middle tensor
        idx1_1 = out_1.shape[2]//2
        t1_1 = out_1[:,:,idx1_1,:,:]

        ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out_1 = self.encoder2(out_1)
        # ipdb.set_trace()
        idx2_1 = out_1.shape[2]//2
        t2_1 = out_1[:,:,idx2_1,:,:]
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out_1 = self.bottleneck(out_1)
        out_1 = self.convbottle(out_1)
        idx3_1 = out_1.shape[2]//2
        t3_1 = out_1[:,:,idx3_1,:,:]

        # ipdb.set_trace()



        out_1 = F.relu(F.interpolate(self.dbn3(self.decoder1(t3_1)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t2_1)
        out_1 = F.relu(F.interpolate(self.dbn4(self.decoder2(out_1)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t1_1)
        out_1 = F.relu(F.interpolate(self.dbn5( self.decoder3(out_1)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t0_1)
        out_1 = self.decoder4(out_1)

        cnn_feature = out_1
        out = self.final(out_1)

        # final_1 = self.final_1(out_1,imbalance_attention,0)
        # out = self.final_2(final_1,imbalance_attention,0)




        return out,cnn_feature,imbalance_attention
class MIAHybridResUnet_test(nn.Module):
    def __init__(self,  num_classes, num_imbalance=1,input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
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
        # self.encoder = ResBlock3D(input_channels, 64, stride=1, p=0.5) #15 96->15 96
        self.encoder = MIAResBlock3D(input_channels, 64, stride=1, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # self.encoder1 = ResBlock3D(64, 64, stride=2, p=0.5) # 15 96->7 ,48
        self.encoder1 = MIAResBlock3D(64, 64, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # self.einscheck2 = Einsum_check(64, 128,stride=[1,2,2],padding=[0,1,1])
        # self.encoder2 = ResBlock3D(64, 128, stride=2, p=0.5)    # 7 48->3 24
        self.encoder2 = MIAResBlock3D(64, 128, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # self.bottleneck = ResBlock3D(128, 256, stride=2, p=0.5) # 3 24->1 12
        self.bottleneck = MIAResBlock3D(128, 256, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # self.convbottle = nn.Conv3d(256, 256, 1, stride=1, padding=0) #1 12->1 12
        self.convbottle = MIAOutConv3D(256, 256, 3, stride=1,im_kernel_ratio=cfg.im_kernel_ratio, padding=1) #1 12->1 12

        # self.einscheck1 = Einsum_check(input_channels, 64,kernel_size=[3,6,6], stride=[1,2,2],padding=[0,1,1])
        # self.encoder1 = MIAResBlock3D(input_channels, 64, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # # self.encoder1 = AttentionConvBlock3d(input_channels, 64, [15, img_size, img_size],stride=1, stride_attention=[[2,2,2],[2,2,2]],stride_attenconv=2)
        # # self.encoder1 = AttentionConvBlock3d_4stride(input_channels, 64, [15, img_size, img_size],stride=1, stride_attention=[[1,2,2],[1,2,2],[2,2,2],[2,2,2]],stride_attenconv=2)
        # self.encoder2 = MIAResBlock3D(64, 128, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # # self.encoder2 = AttentionConvBlock3d(64, 128, [7, img_size//2, img_size//2],stride=1, stride_attention=[[1,2,2],[2,2,2]],stride_attenconv=2)
        # # self.encoder2 = AttentionConvBlock3d_4stride(64, 128, [7, img_size//2, img_size//2],stride=1, stride_attention=[[1,2,2],[1,2,2],[1,2,2],[2,2,2]],stride_attenconv=2)
        # self.encoder3 = MIAResBlock3D(128, 256, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        # self.encoder3 = AttentionConvBlock3d(128, 256, [3, img_size//4, img_size//4],stride=1, stride_attention=[[1,2,2],[1,2,2]],stride_attenconv=2)
        # self.encoder3 = AttentionConvBlock3d_4stride(128, 256, [3, img_size//4, img_size//4],stride=1, stride_attention=[[1,1,1],[1,2,2],[1,2,2],[1,2,2]],stride_attenconv=2)


        # self.decoder1 =   ResBlock2D(256, 128, stride=1,p=0.5 ) # 256 1 12-> 128 1 24
        self.decoder1 = MIAResBlock2D(256, 128, stride=1,p=0.5,im_kernel_ratio=cfg.im_kernel_ratio ) # 256 1 12-> 128 1 24
        # self.decoder2 =   ResBlock2D(128, 64, stride=1, p =0.5) #128 1 24->64 1 48
        self.decoder2 = MIAResBlock2D(128, 64, stride=1, p =0.5,im_kernel_ratio=cfg.im_kernel_ratio) #128 1 24->64 1 48
        # self.decoder3 =   ResBlock2D(64, 64, stride=1,p=0.5 ) # 64 1 48->64  1 96
        self.decoder3 = MIAResBlock2D(64, 64, stride=1,p=0.5,im_kernel_ratio=cfg.im_kernel_ratio ) # 64 1 48->64  1 96
        # self.decoder4 =   ResBlock2D(64, 64, stride=1,p=0.5 ) #64  1 96-> 64 1 96
        self.decoder4 = MIAResBlock2D(64, 64, stride=1,p=0.5,im_kernel_ratio=cfg.im_kernel_ratio ) #64  1 96-> 64 1 96
        # self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(128)
        self.dbn4 = nn.BatchNorm2d(64)
        self.dbn5 = nn.BatchNorm2d(64)


        # self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        # self.final_1 =MIAOutConv2D(64, 64, kernel_size=1,im_kernel_ratio=cfg.im_kernel_ratio)
        self.final =MIAInConv2D(64, num_classes, kernel_size=1,im_kernel_ratio=cfg.im_kernel_ratio)
        self.soft = nn.Softmax(dim =1)


    def forward(self, x,imbalance_attention):

        # return imbalance_attention
        # ipdb.set_trace()
        # print("im_kernel_ratio:",cfg.im_kernel_ratio)

        B_1 = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        out_1 = self.encoder(x,imbalance_attention,0)
        idx1_1 = out_1.shape[2]//2
        t0_1 = out_1[:,:,idx1_1,:,:]

        out_1 = self.encoder1(out_1,imbalance_attention,0)
        # ipdb.set_trace()

        #get middle tensor
        idx1_1 = out_1.shape[2]//2
        t1_1 = out_1[:,:,idx1_1,:,:]

        ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out_1 = self.encoder2(out_1,imbalance_attention,0)
        # ipdb.set_trace()
        idx2_1 = out_1.shape[2]//2
        t2_1 = out_1[:,:,idx2_1,:,:]
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out_1 = self.bottleneck(out_1,imbalance_attention,0)
        out_1 = self.convbottle(out_1,imbalance_attention,0)
        idx3_1 = out_1.shape[2]//2
        t3_1 = out_1[:,:,idx3_1,:,:]

        # ipdb.set_trace()



        out_1 = F.relu(F.interpolate(self.dbn3(self.decoder1(t3_1,imbalance_attention,0)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t2_1)
        out_1 = F.relu(F.interpolate(self.dbn4(self.decoder2(out_1,imbalance_attention,0)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t1_1)
        out_1 = F.relu(F.interpolate(self.dbn5( self.decoder3(out_1,imbalance_attention,0)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t0_1)
        out_1 = self.decoder4(out_1,imbalance_attention,0)

        cnn_feature = out_1
        out = self.final(out_1,imbalance_attention,0)

        # final_1 = self.final_1(out_1,imbalance_attention,0)
        # out = self.final_2(final_1,imbalance_attention,0)




        return out,cnn_feature,imbalance_attention

class MIAHybridResUnet(nn.Module):
    def __init__(self,  num_classes, num_imbalance=1,input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.encoder = MIAResBlock3D(input_channels, 64, stride=1, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        self.encoder1 = MIAResBlock3D(64, 64, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        self.encoder2 = MIAResBlock3D(64, 128, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        self.bottleneck = MIAResBlock3D(128, 256, stride=2, p=0.5,im_kernel_ratio=cfg.im_kernel_ratio)
        self.convbottle = MIAOutConv3D(256, 256, 3, stride=1,im_kernel_ratio=cfg.im_kernel_ratio, padding=1) #1 12->1 12
        self.decoder1 = MIAResBlock2D(256, 128, stride=1,p=0.5,im_kernel_ratio=cfg.im_kernel_ratio ) # 256 1 12-> 128 1 24
        self.decoder2 = MIAResBlock2D(128, 64, stride=1, p =0.5,im_kernel_ratio=cfg.im_kernel_ratio) #128 1 24->64 1 48
        self.decoder3 = MIAResBlock2D(64, 64, stride=1,p=0.5,im_kernel_ratio=cfg.im_kernel_ratio ) # 64 1 48->64  1 96
        self.decoder4 = MIAResBlock2D(64, 64, stride=1,p=0.5,im_kernel_ratio=cfg.im_kernel_ratio ) #64  1 96-> 64 1 96
        self.dbn3 = nn.BatchNorm2d(128)
        self.dbn4 = nn.BatchNorm2d(64)
        self.dbn5 = nn.BatchNorm2d(64)

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fl =nn.Sequential( nn.Linear(4*24*24, 32),
                                nn.Linear(32, 2))



        self.final_1 =MIAOutConv2D(64, 64, kernel_size=1,im_kernel_ratio=cfg.im_kernel_ratio)
        self.final_2 =MIAInConv2D(64, num_classes, kernel_size=1,im_kernel_ratio=cfg.im_kernel_ratio)
        self.soft = nn.Softmax(dim =1)



    def forward(self, x):

        # return imbalance_attention
        # ipdb.set_trace()
        # print("im_kernel_ratio:",cfg.im_kernel_ratio)
        out = self.encoder(x,task_indx=1)
        idx1 = out.shape[2]//2
        t0 = out[:,:,idx1,:,:]

        out = self.encoder1(out,task_indx=1)
        # ipdb.set_trace()

        #get middle tensor
        idx1 = out.shape[2]//2
        t1 = out[:,:,idx1,:,:]

        ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out = self.encoder2(out,task_indx=1)
        # ipdb.set_trace()
        idx2 = out.shape[2]//2
        t2 = out[:,:,idx2,:,:]
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out = self.bottleneck(out,task_indx=1)
        out = self.convbottle(out,task_indx=1)
        idx3 = out.shape[2]//2
        t3 = out[:,:,idx3,:,:]

        # ipdb.set_trace()



        out = F.relu(F.interpolate(self.dbn3(self.decoder1(t3,task_indx=1)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder2(out,task_indx=1)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.dbn5( self.decoder3(out,task_indx=1)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t0)
        out = self.decoder4(out,task_indx=1)

        out_cls = self.layer1(out)
        out_cls = self.layer2(out_cls)
        imbalance_attention = self.fl(out_cls.view(out_cls.size(0), -1))
        imbalance_attention = self.soft(imbalance_attention)
        imbalance_attention_ =imbalance_attention.detach()
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out_1 = self.encoder(x,imbalance_attention_,0)
        idx1_1 = out_1.shape[2]//2
        t0_1 = out_1[:,:,idx1_1,:,:]

        out_1 = self.encoder1(out_1,imbalance_attention_,0)
        # ipdb.set_trace()

        #get middle tensor
        idx1_1 = out_1.shape[2]//2
        t1_1 = out_1[:,:,idx1_1,:,:]

        ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out_1 = self.encoder2(out_1,imbalance_attention_,0)
        # ipdb.set_trace()
        idx2_1 = out_1.shape[2]//2
        t2_1 = out_1[:,:,idx2_1,:,:]
        ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out_1 = self.bottleneck(out_1,imbalance_attention_,0)
        out_1 = self.convbottle(out_1,imbalance_attention_,0)
        idx3_1 = out_1.shape[2]//2
        t3_1 = out_1[:,:,idx3_1,:,:]

        # ipdb.set_trace()



        out_1 = F.relu(F.interpolate(self.dbn3(self.decoder1(t3_1,imbalance_attention_,0)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t2_1)
        out_1 = F.relu(F.interpolate(self.dbn4(self.decoder2(out_1,imbalance_attention_,0)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t1_1)
        out_1 = F.relu(F.interpolate(self.dbn5( self.decoder3(out_1,imbalance_attention_,0)),scale_factor=(2,2),mode ='bilinear'))
        out_1 = torch.add(out_1,t0_1)
        out_1 = self.decoder4(out_1,imbalance_attention_,0)

        cnn_feature = out_1
        out_1 = self.final_1(out_1,imbalance_attention_,0)
        output = self.final_2(out_1,imbalance_attention_,0)
        #
        # final_1 = self.final_1(out_1,imbalance_attention,0)
        # out = self.final_2(final_1,imbalance_attention,0)




        # return imbalance_attention
        return output,cnn_feature,imbalance_attention


def Attentionconvnet():
    return AttentionConvNet(num_classes=3, input_channels=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                   num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                   attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                   depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
def MIAunext():
    return MIAUnext(num_classes=3, input_channels=1,num_imbalance=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                               num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                               attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                               depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])
def MIAunext_test():
    return MIAUnext_test(num_classes=3, input_channels=1,num_imbalance=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                    num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])

def Hybridresunet_test():
    return HybridResunet_test(num_classes=3, input_channels=1,num_imbalance=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                    num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])

def MIAHybridresreunet_test():
    return MIAHybridResUnet_test(num_classes=3, input_channels=1,num_imbalance=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                    num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])

def MIAHybrid():
    return MIAHybridResUnet(num_classes=3, input_channels=1,num_imbalance=1, deep_supervision=False,img_size=96, patch_size=16, in_chans=3,  embed_dims=[ 256, 384, 512],
                    num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])