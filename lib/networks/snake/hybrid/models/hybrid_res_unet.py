# coding = utf-8

""" Hybrid Res-UNet architecture with regularization of number of predicted boundary pixels
    the contract path is 3D while the expansion path is 2D.
    For input, slices before and after current slice are concatenated as a volume.
    For output, annotation of current slice is compared with the prediction (single slice)
"""

import torch
from torch import nn
from .utils import _initialize_weights_2d, _initialize_weights_3d
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# 3D convolution
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

# 2D convolution
def conv_33(in_channels, out_channels, stride=1):
    # since BN is used, bias is not necessary
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv_11(in_channels, out_channels, stride=1):
    # since BN is used, bias is not necessary
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=False)

class ResBlock2D(nn.Module):
    """ 2D residual block """
    def __init__(self, in_channels, out_channels, stride=1, p=0.5, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv_33(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv_33(out_channels, out_channels, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout2d(p=p)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        # print("x_shape:",x.shape)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dp(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dp(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual

        return out

class UpConv(nn.Module):
    """ up convolution """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,
                                            stride=2, padding=0)

    def forward(self, skip, x):
        """ skip is 3D volume and x is 2D slice, central slice of skip is concatenated with x """
        # print("skip shape: {}".format(skip.shape))
        # print("x shape: {}".format(x.shape))
        central_inx = skip.size(2) // 2
        skip_slice = skip[:, :, central_inx]
        # print("x:",x.shape)
        # print("skip_slice:",skip_slice.shape)

        out = self.transconv(x)
        # print("out:", out.shape)
        out = torch.cat([skip_slice, out], 1)
        # print("out_cat:", out.shape)
        # x: torch.Size([16, 256, 12, 12])
        # skip_slice: torch.Size([16, 128, 24, 24])
        # out: torch.Size([16, 128, 24, 24])
        # out_cat: torch.Size([16, 256, 24, 24])

        return out
class UpConv_2D(nn.Module):
    """ up convolution """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,
                                            stride=2, padding=0)

    def forward(self, skip, x):
        """ skip is 3D volume and x is 2D slice, central slice of skip is concatenated with x """
        # central_inx = skip.size(2) // 2
        # skip_slice = skip[:, :, central_inx]
        # print("x:",x.shape)
        # print("skip_slice:",skip_slice.shape)

        out = self.transconv(x)
        # print("out:", out.shape)
        out = torch.cat([skip, out], 1)
        # print("out_cat:", out.shape)
        # x: torch.Size([16, 256, 12, 12])
        # skip_slice: torch.Size([16, 128, 24, 24])
        # out: torch.Size([16, 128, 24, 24])
        # out_cat: torch.Size([16, 256, 24, 24])

        return out
class UpConv_skip(nn.Module):
    """ up convolution """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,
                                            stride=2, padding=0)

    def forward(self, skip, x):
        """ skip is 3D volume and x is 2D slice, central slice of skip is concatenated with x """
        central_inx = skip.size(2) // 2
        skip_slice = skip[:, :, central_inx]
        # print("x:",x.shape)
        # print("skip_slice:",skip_slice.shape)

        out = self.transconv(x)
        # print("out:", out.shape)
        out = torch.cat([skip_slice, out], 1)
        # print("out_cat:", out.shape)
        # x: torch.Size([16, 256, 12, 12])
        # skip_slice: torch.Size([16, 128, 24, 24])
        # out: torch.Size([16, 128, 24, 24])
        # out_cat: torch.Size([16, 256, 24, 24])

        return out
class Resblock(nn.Module):
    """ Res UNet class """
    def __init__(self, in_channels=64, out_channels=[64,3], n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32], bottleneck = 512, p=0.5):
        super().__init__()
        self.Block = ResBlock2D(in_channels, out_channels[0], stride=1, p=p)
        # self.Block_1 = ResBlock2D(in_channels, in_channels, stride=1, p=p)
        # self.Block_2 = ResBlock2D(in_channels, out_channels[0], stride=1, p=p)
        self.fl = nn.Conv2d(out_channels[0], out_channels[-1], kernel_size=1)

        # initialize weights
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.Block(x)
        # out = self.Block_1(x)
        # out = self.Block_2(x)
        cnn_feature = out
        out = self.fl(out)
        # print(out.size())

        return out,cnn_feature
class Resblock_seg(nn.Module):
    """ Res UNet class """
    def __init__(self, in_channels=64, out_channels=[64,3], n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32], bottleneck = 512, p=0.5):
        super().__init__()
        self.Block = ResBlock2D(in_channels, out_channels[0], stride=1, p=p)
        # self.Block_1 = ResBlock2D(in_channels, in_channels, stride=1, p=p)
        # self.Block_2 = ResBlock2D(in_channels, out_channels[0], stride=1, p=p)
        self.fl_0 = nn.Conv2d(out_channels[0], out_channels[-1], kernel_size=1)
        self.fl_1 = nn.Conv2d(out_channels[0], out_channels[-1], kernel_size=1)

        # initialize weights
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.Block(x)
        # out = self.Block_1(x)
        # out = self.Block_2(x)
        cnn_feature = out
        out_0 = self.fl_0(out)
        out_1 = self.fl_1(out)
        # print(out.size())

        return out_0,out_1,cnn_feature

class ResUNet_Cls(nn.Module):
    """ Res UNet class """
    def __init__(self, in_channels=1, out_channels=[10,2], n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32], bottleneck = 512, p=0.5):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.n_slices = n_slices
        self.input_size = input_size

        self.conv1 = nn.Conv3d(in_channels, self.down_blocks[0], 3, padding=1)

        # contract path
        self.BlocksDown = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx-1]
                self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=2, p=p))

        # bottleneck block
        # make sure there is only single one slice in current layer
        self.bottleneck  = ResBlock3D(self.down_blocks[-1], bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11 = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices//scale, 1, 1))

        # expansive path
        self.BlocksUp = nn.ModuleList([])
        self.TransUpBlocks = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
            output_channel = self.up_blocks[b_inx]
            self.TransUpBlocks.append(UpConv(input_channel, output_channel))
            self.BlocksUp.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))

        # final convolution layer
        # self.fl = nn.Conv2d(self.up_blocks[-1], out_channels, kernel_size=1)

        self.fl_seg = nn.Conv2d(self.up_blocks[-1], out_channels[-1]-1, kernel_size=1)

        # self.layer1 = ResBlock2D(up_blocks[-1], int(up_blocks[-1] / 2), stride=2, p=p)
        #
        # self.layer2 = ResBlock2D(int(up_blocks[-1] / 2), int(up_blocks[-1] / 4), stride=2, p=p)
        # self.layer3 = ResBlock2D(int(up_blocks[-1] / 4), int(up_blocks[-1] / 4), stride=2, p=p)
        # self.fl = nn.Sequential(nn.Linear(8 * 12 * 12, out_channels[0]),
        #                         nn.Linear(out_channels[0], out_channels[1]))
        # self.sigmoid = nn.Sigmoid()

        # initialize weights
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        # print(out.size())
        skip_connections = []
        for down_block in self.BlocksDown:
            out = down_block(out)
            # print("out_shape:",out.shape)
            skip_connections.append(out)
            # print(out.size())
        # sys.exit()

        out = self.bottleneck(out)
        # if out.size(2) > 1:
        out = self.conv_n11(out) # fuse several slices in the bottleneck layer
        # print("self.up_blocks:",self.up_blocks)
        for b_inx in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            if b_inx == 0:
                out = self.TransUpBlocks[b_inx](skip, out[:, :, 0])
            else:
                out = self.TransUpBlocks[b_inx](skip, out)
            # print("out_shape:",out.shape)
            out= self.BlocksUp[b_inx](out)


            if out.shape[1]==64 and out.shape[2]==96:
                out_cnn=out
        # input.shape: torch.Size([8, 1, 15, 96, 96])
        # out.shape: torch.Size([8, 32, 15, 96, 96])
        # out.shape: torch.Size([8, 64, 7, 48, 48])
        # out.shape: torch.Size([8, 128, 3, 24, 24])
        # mid_out.shape: torch.Size([8, 256, 1, 12, 12])
        # out.shape: torch.Size([8, 128, 24, 24])
        # out.shape: torch.Size([8, 64, 48, 48])
        # out.shape: torch.Size([8, 32, 96, 96])
        # F_out.shape: torch.Size([8, 3, 96, 96])
        # print("out_layer1:",out.shape)
        out_seg = self.fl_seg(out)
        out = self.layer1(out)
        # print("out_layer2:", out.shape)
        out = self.layer2(out)
        # out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        # print("out_layerfl:", out.shape)
        out = self.fl(out)
        # out = self.sigmoid(out)
        # print("out_layerfl_aft:", out.shape)
        return out,out_seg
class SkipResUNet_Cls(nn.Module):
    """ Res UNet class """
    def __init__(self, in_channels=1, out_channels=5, n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32],up_blocks_sub=[ 128, 64, 32], bottleneck = 512, p=0.5):
        super().__init__()
        self.down_blocks = down_blocks
        self.down_blocks_sub = down_blocks
        self.up_blocks_sub = up_blocks_sub
        self.up_blocks = up_blocks
        self.n_slices = n_slices
        self.input_size = input_size

        self.conv1 = nn.Conv3d(in_channels, self.down_blocks[0], 3, padding=1)
        self.conv1_sub = nn.Conv3d(in_channels, self.down_blocks_sub[0], 3, padding=1)

        # contract path
        self.BlocksDown = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown.append(ResBlock3D(input_channel*2, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx-1]
                self.BlocksDown.append(ResBlock3D(input_channel*2, output_channel, stride=2, p=p))
        self.BlocksDown_sub = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown_sub.append(ResBlock3D(input_channel, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx - 1]
                self.BlocksDown_sub.append(ResBlock3D(input_channel, output_channel, stride=2, p=p))


        # bottleneck block
        # make sure there is only single one slice in current layer
        self.bottleneck  = ResBlock3D(self.down_blocks[-1]*2, bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11 = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices//scale, 1, 1))
        self.bottleneck_sub = ResBlock3D(self.down_blocks[-1], bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11_sub = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices // scale, 1, 1))

        # expansive path
        self.BlocksUp = nn.ModuleList([])
        self.TransUpBlocks = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
            output_channel = self.up_blocks[b_inx]
            self.TransUpBlocks.append(UpConv(input_channel, output_channel))
            self.BlocksUp.append(ResBlock2D(input_channel*2, output_channel, stride=1, p=p))

        # final convolution layer

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=2, stride=1, padding=0),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2))
        self.fl =nn.Sequential( nn.Linear(8*22*22, out_channels[0]),
                            nn.Linear(out_channels[0], out_channels[1]))
        self.BlocksUp_sub = nn.ModuleList([])
        self.TransUpBlocks_sub = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks_sub):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks_sub[b_inx - 1]
            output_channel = self.up_blocks_sub[b_inx]
            self.TransUpBlocks_sub.append(UpConv(input_channel, output_channel))
            self.BlocksUp_sub.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))

        # final convolution layer
        self.fl_sub = nn.Conv2d(self.up_blocks[-1], out_channels[-1], kernel_size=1)

        # initialize weights
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out_sub = self.conv1_sub(x)
        # print(out.size())
        skip_connections = []
        skip_connections_sub = []
        skip_sub=[]
        skip_sub.append(out_sub)
        for down_block in self.BlocksDown_sub:

            out_sub = down_block(out_sub)
            # print("out_shape:",out.shape)
            skip_connections_sub.append(out_sub)
            skip_sub.append(out_sub)
            # print("out_sub_down:",out_sub.size())

        for b_inx in range(len(self.down_blocks)):
            out = torch.cat([skip_sub[b_inx], out], 1)
            # print("out_down_connect:", out.size())
            out = self.BlocksDown[b_inx](out)
            skip_connections.append(out)
            # print("out_down:", out.size())

        # sys.exit()


        out_sub = self.bottleneck_sub(out_sub)
        # print("out_sub_bottle:", out_sub.size())
        # if out.size(2) > 1:
        out_sub = self.conv_n11_sub(out_sub) # fuse several slices in the bottleneck layer
        # print("out_sub_conv_n11:", out_sub.size())
        out = torch.cat([skip_sub[-1], out], 1)
        # print("out_bottle_before:", out.size())
        out = self.bottleneck(out)
        # print("out_bottle:", out.size())
        out = self.conv_n11(out)
        # print("out_conv_n11:", out.size())

        # print("self.up_blocks:",self.up_blocks)
        skip_connection_up_sub=[]
        for b_inx in range(len(self.up_blocks_sub)):
            skip = skip_connections_sub.pop()
            # print("in skip_connection_up_sub:",skip.shape)
            if b_inx == 0:
                out_sub = self.TransUpBlocks_sub[b_inx](skip, out_sub[:, :, 0])
                skip_connection_up_sub.append(out_sub)
            else:
                out_sub = self.TransUpBlocks_sub[b_inx](skip, out_sub)
                skip_connection_up_sub.append(out_sub)
            # print("outsub_shape:",out_sub.shape)
            out_sub = self.BlocksUp_sub[b_inx](out_sub)


            if out_sub.shape[1]==32 and out_sub.shape[2]==96:
                out_cnn_sub=out_sub
        # input.shape: torch.Size([8, 1, 15, 96, 96])
        # out.shape: torch.Size([8, 32, 15, 96, 96])
        # out.shape: torch.Size([8, 64, 7, 48, 48])
        # out.shape: torch.Size([8, 128, 3, 24, 24])
        # mid_out.shape: torch.Size([8, 256, 1, 12, 12])
        # out.shape: torch.Size([8, 128, 24, 24])
        # out.shape: torch.Size([8, 64, 48, 48])
        # out.shape: torch.Size([8, 32, 96, 96])
        # F_out.shape: torch.Size([8, 3, 96, 96])
        output_sub = self.fl_sub(out_sub)
        for b_inx in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            if b_inx == 0:
                out = self.TransUpBlocks[b_inx](skip, out[:, :, 0])
            else:
                out = self.TransUpBlocks[b_inx](skip, out)
            out = torch.cat([skip_connection_up_sub[b_inx], out], 1)
            # print("out_shape:",out.shape)
            out = self.BlocksUp[b_inx](out)


            if out.shape[1]==32 and out.shape[2]==96:
                out_cnn=out
        # output = self.fl(out)
        out = self.layer1(out)
        # print("out_layer2:", out.shape)
        out = self.layer2(out)
        # out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        # print("out_layerfl:", out.shape)
        output = self.fl(out)


        return output, out_cnn,output_sub,out_cnn_sub
class ResUNet_out_2d(nn.Module):
    """ Res UNet class """
    def __init__(self, in_channels=1, out_channels=5, n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32], bottleneck = 512, p=0.5):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.n_slices = n_slices
        self.input_size = input_size

        self.conv1 = nn.Conv2d(in_channels, self.down_blocks[0], 3, padding=1)

        # contract path
        self.BlocksDown = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx-1]
                self.BlocksDown.append(ResBlock2D(input_channel, output_channel, stride=2, p=p))

        # bottleneck block
        # make sure there is only single one slice in current layer
        self.bottleneck  = ResBlock2D(self.down_blocks[-1], bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11 = nn.Conv2d(bottleneck, bottleneck, kernel_size=( 1, 1))

        # expansive path
        self.BlocksUp = nn.ModuleList([])
        self.TransUpBlocks = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
            output_channel = self.up_blocks[b_inx]
            self.TransUpBlocks.append(UpConv_2D(input_channel, output_channel))
            self.BlocksUp.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))

        # final convolution layer
        # self.fl = nn.Conv2d(self.up_blocks[-1], out_channels, kernel_size=1)

        # initialize weights
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        # print(out.size())
        skip_connections = []
        for down_block in self.BlocksDown:
            out = down_block(out)
            # print("downout_shape:",out.shape)
            skip_connections.append(out)
            # print(out.size())
        # sys.exit()

        out = self.bottleneck(out)
        # if out.size(2) > 1:
        out = self.conv_n11(out) # fuse several slices in the bottleneck layer
        # print("conv_n11:",out.shape)
        # print("self.up_blocks:",self.up_blocks)
        for b_inx in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            # print("skip:",skip.shape)
            if b_inx == 0:
                out = self.TransUpBlocks[b_inx](skip, out)
            else:
                out = self.TransUpBlocks[b_inx](skip, out)
            # print("up_out_shape:",out.shape)
            out = self.BlocksUp[b_inx](out)


            # if out.shape[1]==64 and out.shape[2]==96:
            #     out_cnn=out
        # input.shape: torch.Size([8, 1, 15, 96, 96])
        # out.shape: torch.Size([8, 32, 15, 96, 96])
        # out.shape: torch.Size([8, 64, 7, 48, 48])
        # out.shape: torch.Size([8, 128, 3, 24, 24])
        # mid_out.shape: torch.Size([8, 256, 1, 12, 12])
        # out.shape: torch.Size([8, 128, 24, 24])
        # out.shape: torch.Size([8, 64, 48, 48])
        # out.shape: torch.Size([8, 32, 96, 96])
        # F_out.shape: torch.Size([8, 3, 96, 96])
        # output = self.fl(out)
        return out
class ResUNet_64out(nn.Module):
    """ Res UNet class """
    def __init__(self, in_channels=1, out_channels=5, n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32], bottleneck = 512, p=0.5):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.n_slices = n_slices
        self.input_size = input_size

        self.conv1 = nn.Conv3d(in_channels, self.down_blocks[0], 3, padding=1)

        # contract path
        self.BlocksDown = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx-1]
                self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=2, p=p))

        # bottleneck block
        # make sure there is only single one slice in current layer
        self.bottleneck  = ResBlock3D(self.down_blocks[-1], bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11 = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices//scale, 1, 1))

        # expansive path
        self.BlocksUp = nn.ModuleList([])
        self.TransUpBlocks = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
            output_channel = self.up_blocks[b_inx]
            self.TransUpBlocks.append(UpConv(input_channel, output_channel))
            self.BlocksUp.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))

        # final convolution layer
        # self.fl = nn.Conv2d(self.up_blocks[-1], out_channels, kernel_size=1)

        # initialize weights
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        # print(out.size())
        skip_connections = []
        for down_block in self.BlocksDown:
            out = down_block(out)
            # print("out_shape:",out.shape)
            skip_connections.append(out)
            # print(out.size())
        # sys.exit()

        out = self.bottleneck(out)
        # if out.size(2) > 1:
        out = self.conv_n11(out) # fuse several slices in the bottleneck layer
        # print("self.up_blocks:",self.up_blocks)
        for b_inx in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            if b_inx == 0:
                out = self.TransUpBlocks[b_inx](skip, out[:, :, 0])
            else:
                out = self.TransUpBlocks[b_inx](skip, out)
            # print("out_shape:",out.shape)
            out = self.BlocksUp[b_inx](out)


            if out.shape[1]==64 and out.shape[2]==96:
                out_cnn=out
        # input.shape: torch.Size([8, 1, 15, 96, 96])
        # out.shape: torch.Size([8, 32, 15, 96, 96])
        # out.shape: torch.Size([8, 64, 7, 48, 48])
        # out.shape: torch.Size([8, 128, 3, 24, 24])
        # mid_out.shape: torch.Size([8, 256, 1, 12, 12])
        # out.shape: torch.Size([8, 128, 24, 24])
        # out.shape: torch.Size([8, 64, 48, 48])
        # out.shape: torch.Size([8, 32, 96, 96])
        # F_out.shape: torch.Size([8, 3, 96, 96])
        # output = self.fl(out)
        return out,out_cnn
# class ResUNet(nn.Module):
#     """ Res UNet class """
#     def __init__(self, in_channels=1, out_channels=5, n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
#                  up_blocks = [256, 128, 64, 32], bottleneck = 512, p=0.5):
#         super().__init__()
#         self.down_blocks = down_blocks
#         self.up_blocks = up_blocks
#         self.n_slices = n_slices
#         self.input_size = input_size
#
#         self.conv1 = nn.Conv3d(in_channels, self.down_blocks[0], 3, padding=1)
#
#         # contract path
#         self.BlocksDown = nn.ModuleList([])
#         for b_inx, down_block in enumerate(self.down_blocks):
#             output_channel = self.down_blocks[b_inx]
#             if b_inx == 0:
#                 input_channel = self.down_blocks[0]
#                 self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=1, p=p))
#             else:
#                 input_channel = self.down_blocks[b_inx-1]
#                 self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=2, p=p))
#
#         # bottleneck block
#         # make sure there is only single one slice in current layer
#         self.bottleneck  = ResBlock3D(self.down_blocks[-1], bottleneck, stride=2, p=p)
#         scale = 2 ** len(down_blocks)
#         self.conv_n11 = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices//scale, 1, 1))
#
#         # expansive path
#         self.BlocksUp = nn.ModuleList([])
#         self.TransUpBlocks = nn.ModuleList([])
#         for b_inx, up_block in enumerate(self.up_blocks):
#             input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
#             output_channel = self.up_blocks[b_inx]
#             self.TransUpBlocks.append(UpConv(input_channel, output_channel))
#             self.BlocksUp.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))
#
#         # final convolution layer
#         self.fl = nn.Conv2d(self.up_blocks[-1], out_channels, kernel_size=1)
#
#         # initialize weights
#         _initialize_weights_3d(self)
#         _initialize_weights_2d(self)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         # print(out.size())
#         skip_connections = []
#         for down_block in self.BlocksDown:
#             out = down_block(out)
#             print("out_shape:",out.shape)
#             skip_connections.append(out)
#             print(out.size())
#         # sys.exit()
#         # sys.exit()
#
#         out = self.bottleneck(out)
#         print("mid_out.shape:",out.shape)
#         # if out.size(2) > 1:
#         out = self.conv_n11(out) # fuse several slices in the bottleneck layer\
#         print("out.shape:",out.shape)
#         # print("self.up_blocks:",self.up_blocks)
#         for b_inx in range(len(self.up_blocks)):
#             skip = skip_connections.pop()
#             print("skip.shape:",skip.shape)
#             if b_inx == 0:
#                 out = self.TransUpBlocks[b_inx](skip, out[:, :, 0])
#             else:
#                 out = self.TransUpBlocks[b_inx](skip, out)
#             print("out1_shape:",out.shape)
#             out = self.BlocksUp[b_inx](out)
#             print("out2_shape:",out.shape)
#
#
#             if out.shape[1]==32 and out.shape[2]==96:
#                 out_cnn=out
#         # input.shape: torch.Size([8, 1, 15, 96, 96])
#         # out.shape: torch.Size([8, 32, 15, 96, 96])
#         # out.shape: torch.Size([8, 64, 7, 48, 48])
#         # out.shape: torch.Size([8, 128, 3, 24, 24])
#         # mid_out.shape: torch.Size([8, 256, 1, 12, 12])
#         # out.shape: torch.Size([8, 128, 24, 24])
#         # out.shape: torch.Size([8, 64, 48, 48])
#         # out.shape: torch.Size([8, 32, 96, 96])
#         # F_out.shape: torch.Size([8, 3, 96, 96])
#         output = self.fl(out)
#         return output, out_cnn

class ResUNet(nn.Module):
    """ Res UNet class """
    def __init__(self, in_channels=1, out_channels=5, n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32], bottleneck = 512, p=0.5):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.n_slices = n_slices
        self.input_size = input_size

        self.conv1 = nn.Conv3d(in_channels, self.down_blocks[0], 3, padding=1)

        # contract path
        self.BlocksDown = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx-1]
                self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=2, p=p))

        # bottleneck block
        # make sure there is only single one slice in current layer
        self.bottleneck  = ResBlock3D(self.down_blocks[-1], bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11 = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices//scale, 1, 1))

        # expansive path
        self.BlocksUp = nn.ModuleList([])
        self.TransUpBlocks = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
            output_channel = self.up_blocks[b_inx]
            self.TransUpBlocks.append(UpConv(input_channel, output_channel))
            self.BlocksUp.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))

        # final convolution layer
        self.fl = nn.Conv2d(self.up_blocks[-1], out_channels, kernel_size=1)

        # initialize weights
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print("input.shape:", x.shape)
        # [batch_size,1,15,96,96]
        out = self.conv1(x)
        # print(out.size())
        skip_connections = []
        for down_block in self.BlocksDown:
            out = down_block(out)
            # print("outdown.shape:", out.shape)

            skip_connections.append(out)
            # print(out.size())
        # sys.exit()
        out = self.bottleneck(out)

        # if out.size(2) > 1:
        out = self.conv_n11(out) # fuse several slices in the bottleneck layer
        # print("mid_out.shape:", out.shape)
        # [batch_size,256,1,12,12]
        for b_inx in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            # print("skip.shape:", skip.shape)
            # print("out.shape:", out.shape)
            if b_inx == 0:

                out = self.TransUpBlocks[b_inx](skip, out[:, :, 0])

            else:
                out = self.TransUpBlocks[b_inx](skip, out)
            # print("out1_shape:", out.shape)


            out = self.BlocksUp[b_inx](out)
            # print("out.shape:", out.shape)
            # [batchsize,64,48,48]...
            if out.shape[1]==64 and out.shape[2]==96:
                    out_cnn=out
            # if out.shape[1]==32 and out.shape[2]==96:
            #     out_cnn=out



        output = self.fl(out)
        # [batch_size,3,96,96]
        # print("F_out.shape:", output.shape)
        # sys.exit()
        return output,out_cnn
class ResUNet_Double(nn.Module):
    """ Res UNet class """
    def __init__(self, in_channels=1, out_channels=5, n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32],up_blocks_seg=[256, 128, 64, 32], bottleneck = 512, p=0.5):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks_seg = up_blocks_seg
        self.up_blocks = up_blocks
        self.n_slices = n_slices
        self.input_size = input_size

        self.conv1 = nn.Conv3d(in_channels, self.down_blocks[0], 3, padding=1)

        # contract path
        self.BlocksDown = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx-1]
                self.BlocksDown.append(ResBlock3D(input_channel, output_channel, stride=2, p=p))


        # bottleneck block
        # make sure there is only single one slice in current layer
        self.bottleneck  = ResBlock3D(self.down_blocks[-1], bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11 = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices//scale, 1, 1))

        # expansive path
        self.BlocksUp = nn.ModuleList([])
        self.TransUpBlocks = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
            output_channel = self.up_blocks[b_inx]
            self.TransUpBlocks.append(UpConv(input_channel, output_channel))
            self.BlocksUp.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))

        # final convolution layer
        self.fl = nn.Conv2d(self.up_blocks[-1], out_channels, kernel_size=1)
        self.BlocksUp_seg = nn.ModuleList([])
        self.TransUpBlocks_seg = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks_seg):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks_seg[b_inx - 1]
            output_channel = self.up_blocks_seg[b_inx]
            self.TransUpBlocks_seg.append(UpConv(input_channel, output_channel))
            self.BlocksUp_seg.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))

        # final convolution layer
        self.fl_seg = nn.Conv2d(self.up_blocks[-1], out_channels, kernel_size=1)

        # initialize weights
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        # print(out.size())
        skip_connections = []
        skip_connections_seg = []
        for down_block in self.BlocksDown:
            out = down_block(out)
            # print("out_shape:",out.shape)
            skip_connections.append(out)
            skip_connections_seg.append(out)
            # print(out.size())
        # sys.exit()

        out = self.bottleneck(out)
        # if out.size(2) > 1:
        out = self.conv_n11(out) # fuse several slices in the bottleneck layer
        out_seg = out
        # print("out_shape:", out.shape)
        # print("self.up_blocks:",self.up_blocks)
        for b_inx in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            if b_inx == 0:
                # print("out1_shape:", out.shape)
                out = self.TransUpBlocks[b_inx](skip, out[:, :, 0])
                # print("out2_shape:", out.shape)
            else:
                out = self.TransUpBlocks[b_inx](skip, out)
                # print("out2_shape:",out.shape)
            out = self.BlocksUp[b_inx](out)
            # print("out3_shape:", out.shape)


            if out.shape[1]==32 and out.shape[2]==96:
                out_cnn=out
        # input.shape: torch.Size([8, 1, 15, 96, 96])
        # out.shape: torch.Size([8, 32, 15, 96, 96])
        # out.shape: torch.Size([8, 64, 7, 48, 48])
        # out.shape: torch.Size([8, 128, 3, 24, 24])
        # mid_out.shape: torch.Size([8, 256, 1, 12, 12])
        # out.shape: torch.Size([8, 128, 24, 24])
        # out.shape: torch.Size([8, 64, 48, 48])
        # out.shape: torch.Size([8, 32, 96, 96])
        # F_out.shape: torch.Size([8, 3, 96, 96])
        output = self.fl(out)
        for b_inx in range(len(self.up_blocks_seg)):
            skip = skip_connections_seg.pop()
            if b_inx == 0:
                out_seg = self.TransUpBlocks_seg[b_inx](skip, out_seg[:, :, 0])
            else:
                out_seg = self.TransUpBlocks_seg[b_inx](skip, out_seg)
            # print("out_shape:",out.shape)
            out_seg = self.BlocksUp_seg[b_inx](out_seg)


            if out_seg.shape[1]==32 and out_seg.shape[2]==96:
                out_cnn_seg=out_seg
        output_seg = self.fl_seg(out_seg)
        return output, out_cnn,output_seg,out_cnn_seg

class ResUNet_Double_Skip(nn.Module):
    """ Res UNet class """
    def __init__(self, in_channels=1, out_channels=5,out_channels_sub=5, n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32],up_blocks_sub=[256, 128, 64, 32], bottleneck = 512, p=0.5):
        super().__init__()
        self.down_blocks = down_blocks
        self.down_blocks_sub = down_blocks
        self.up_blocks_sub = up_blocks_sub
        self.up_blocks = up_blocks
        self.n_slices = n_slices
        self.input_size = input_size

        self.conv1 = nn.Conv3d(in_channels, self.down_blocks[0], 3, padding=1)
        self.conv1_sub = nn.Conv3d(in_channels, self.down_blocks_sub[0], 3, padding=1)

        # contract path
        self.BlocksDown = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown.append(ResBlock3D(input_channel*2, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx-1]
                self.BlocksDown.append(ResBlock3D(input_channel*2, output_channel, stride=2, p=p))
        self.BlocksDown_sub = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown_sub.append(ResBlock3D(input_channel, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx - 1]
                self.BlocksDown_sub.append(ResBlock3D(input_channel, output_channel, stride=2, p=p))


        # bottleneck block
        # make sure there is only single one slice in current layer
        self.bottleneck  = ResBlock3D(self.down_blocks[-1]*2, bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11 = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices//scale, 1, 1))
        self.bottleneck_sub = ResBlock3D(self.down_blocks[-1], bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11_sub = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices // scale, 1, 1))

        # expansive path
        self.BlocksUp = nn.ModuleList([])
        self.TransUpBlocks = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
            output_channel = self.up_blocks[b_inx]
            self.TransUpBlocks.append(UpConv(input_channel, output_channel))
            self.BlocksUp.append(ResBlock2D(input_channel*2, output_channel, stride=1, p=p))

        # final convolution layer
        self.fl = nn.Conv2d(self.up_blocks[-1], out_channels, kernel_size=1)
        self.BlocksUp_sub = nn.ModuleList([])
        self.TransUpBlocks_sub = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks_sub):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks_sub[b_inx - 1]
            output_channel = self.up_blocks_sub[b_inx]
            self.TransUpBlocks_sub.append(UpConv(input_channel, output_channel))
            self.BlocksUp_sub.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))

        # final convolution layer
        self.fl_sub = nn.Conv2d(self.up_blocks[-1], out_channels, kernel_size=1)

        # initialize weights
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out_sub = self.conv1_sub(x)
        # print(out.size())
        skip_connections = []
        skip_connections_sub = []
        skip_sub=[]
        skip_sub.append(out_sub)
        for down_block in self.BlocksDown_sub:

            out_sub = down_block(out_sub)
            # print("out_shape:",out.shape)
            skip_connections_sub.append(out_sub)
            skip_sub.append(out_sub)
            # print("out_sub_down:",out_sub.size())

        for b_inx in range(len(self.down_blocks)):
            out = torch.cat([skip_sub[b_inx], out], 1)
            # print("out_down_connect:", out.size())
            out = self.BlocksDown[b_inx](out)
            skip_connections.append(out)
            # print("out_down:", out.size())

        # sys.exit()


        out_sub = self.bottleneck_sub(out_sub)
        # print("out_sub_bottle:", out_sub.size())
        # if out.size(2) > 1:
        out_sub = self.conv_n11_sub(out_sub) # fuse several slices in the bottleneck layer
        # print("out_sub_conv_n11:", out_sub.size())
        out = torch.cat([skip_sub[-1], out], 1)
        # print("out_bottle_before:", out.size())
        out = self.bottleneck(out)
        # print("out_bottle:", out.size())
        out = self.conv_n11(out)
        # print("out_conv_n11:", out.size())

        # print("self.up_blocks:",self.up_blocks)
        skip_connection_up_sub=[]
        for b_inx in range(len(self.up_blocks_sub)):
            skip = skip_connections_sub.pop()
            if b_inx == 0:
                out_sub = self.TransUpBlocks_sub[b_inx](skip, out_sub[:, :, 0])
                skip_connection_up_sub.append(out_sub)
            else:
                out_sub = self.TransUpBlocks_sub[b_inx](skip, out_sub)
                skip_connection_up_sub.append(out_sub)
            # print("out_shape:",out.shape)
            out_sub = self.BlocksUp_sub[b_inx](out_sub)


            if out_sub.shape[1]==32 and out_sub.shape[2]==96:
                out_cnn_sub=out_sub
        # input.shape: torch.Size([8, 1, 15, 96, 96])
        # out.shape: torch.Size([8, 32, 15, 96, 96])
        # out.shape: torch.Size([8, 64, 7, 48, 48])
        # out.shape: torch.Size([8, 128, 3, 24, 24])
        # mid_out.shape: torch.Size([8, 256, 1, 12, 12])
        # out.shape: torch.Size([8, 128, 24, 24])
        # out.shape: torch.Size([8, 64, 48, 48])
        # out.shape: torch.Size([8, 32, 96, 96])
        # F_out.shape: torch.Size([8, 3, 96, 96])
        output_sub = self.fl_sub(out_sub)
        for b_inx in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            if b_inx == 0:
                out = self.TransUpBlocks[b_inx](skip, out[:, :, 0])
            else:
                out = self.TransUpBlocks[b_inx](skip, out)
            out = torch.cat([skip_connection_up_sub[b_inx], out], 1)
            # print("out_shape:",out.shape)
            out = self.BlocksUp[b_inx](out)


            if out.shape[1]==32 and out.shape[2]==96:
                out_cnn=out
        output = self.fl(out)


        return output, out_cnn,output_sub,out_cnn_sub


class ResUNet_Double_Skip_2out(nn.Module):
    """ Res UNet class """
    def __init__(self, in_channels=1, out_channels=5, n_slices=31, input_size=96, down_blocks=[32, 64, 128, 256],
                 up_blocks = [256, 128, 64, 32],up_blocks_sub=[256, 128, 64, 32], bottleneck = 512, p=0.5):
        super().__init__()
        self.down_blocks = down_blocks
        self.down_blocks_sub = down_blocks
        self.up_blocks_sub = up_blocks_sub
        self.up_blocks = up_blocks
        self.n_slices = n_slices
        self.input_size = input_size

        self.conv1 = nn.Conv3d(in_channels, self.down_blocks[0], 3, padding=1)
        self.conv1_sub = nn.Conv3d(in_channels, self.down_blocks_sub[0], 3, padding=1)

        # contract path
        self.BlocksDown = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown.append(ResBlock3D(input_channel*2, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx-1]
                self.BlocksDown.append(ResBlock3D(input_channel*2, output_channel, stride=2, p=p))
        self.BlocksDown_sub = nn.ModuleList([])
        for b_inx, down_block in enumerate(self.down_blocks):
            output_channel = self.down_blocks[b_inx]
            if b_inx == 0:
                input_channel = self.down_blocks[0]
                self.BlocksDown_sub.append(ResBlock3D(input_channel, output_channel, stride=1, p=p))
            else:
                input_channel = self.down_blocks[b_inx - 1]
                self.BlocksDown_sub.append(ResBlock3D(input_channel, output_channel, stride=2, p=p))


        # bottleneck block
        # make sure there is only single one slice in current layer
        self.bottleneck  = ResBlock3D(self.down_blocks[-1]*2, bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11 = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices//scale, 1, 1))
        self.bottleneck_sub = ResBlock3D(self.down_blocks[-1], bottleneck, stride=2, p=p)
        scale = 2 ** len(down_blocks)
        self.conv_n11_sub = nn.Conv3d(bottleneck, bottleneck, kernel_size=(n_slices // scale, 1, 1))

        # expansive path
        self.BlocksUp = nn.ModuleList([])
        self.TransUpBlocks = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks[b_inx-1]
            output_channel = self.up_blocks[b_inx]
            self.TransUpBlocks.append(UpConv(input_channel, output_channel))
            self.BlocksUp.append(ResBlock2D(input_channel*2, output_channel, stride=1, p=p))

        # final convolution layer
        self.fl = nn.Conv2d(self.up_blocks[-1], out_channels, kernel_size=1)
        self.BlocksUp_sub = nn.ModuleList([])
        self.TransUpBlocks_sub = nn.ModuleList([])
        for b_inx, up_block in enumerate(self.up_blocks_sub):
            input_channel = bottleneck if b_inx == 0 else self.up_blocks_sub[b_inx - 1]
            output_channel = self.up_blocks_sub[b_inx]
            self.TransUpBlocks_sub.append(UpConv(input_channel, output_channel))
            self.BlocksUp_sub.append(ResBlock2D(input_channel, output_channel, stride=1, p=p))

        # final convolution layer
        self.fl_sub_0 = nn.Conv2d(self.up_blocks[-1], out_channels-1, kernel_size=1)
        self.fl_sub_1 = nn.Conv2d(self.up_blocks[-1], out_channels-1, kernel_size=1)

        # initialize weights
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out_sub = self.conv1_sub(x)
        # print(out.size())
        skip_connections = []
        skip_connections_sub = []
        skip_sub=[]
        skip_sub.append(out_sub)
        for down_block in self.BlocksDown_sub:

            out_sub = down_block(out_sub)
            # print("out_shape:",out.shape)
            skip_connections_sub.append(out_sub)
            skip_sub.append(out_sub)
            # print("out_sub_down:",out_sub.size())

        for b_inx in range(len(self.down_blocks)):
            out = torch.cat([skip_sub[b_inx], out], 1)
            # print("out_down_connect:", out.size())
            out = self.BlocksDown[b_inx](out)
            skip_connections.append(out)
            # print("out_down:", out.size())

        # sys.exit()


        out_sub = self.bottleneck_sub(out_sub)
        # print("out_sub_bottle:", out_sub.size())
        # if out.size(2) > 1:
        out_sub = self.conv_n11_sub(out_sub) # fuse several slices in the bottleneck layer
        # print("out_sub_conv_n11:", out_sub.size())
        out = torch.cat([skip_sub[-1], out], 1)
        # print("out_bottle_before:", out.size())
        out = self.bottleneck(out)
        # print("out_bottle:", out.size())
        out = self.conv_n11(out)
        # print("out_conv_n11:", out.size())

        # print("self.up_blocks:",self.up_blocks)
        skip_connection_up_sub=[]
        for b_inx in range(len(self.up_blocks_sub)):
            skip = skip_connections_sub.pop()
            if b_inx == 0:
                out_sub = self.TransUpBlocks_sub[b_inx](skip, out_sub[:, :, 0])
                skip_connection_up_sub.append(out_sub)
            else:
                out_sub = self.TransUpBlocks_sub[b_inx](skip, out_sub)
                skip_connection_up_sub.append(out_sub)
            # print("out_shape:",out.shape)
            out_sub = self.BlocksUp_sub[b_inx](out_sub)


            if out_sub.shape[1]==32 and out_sub.shape[2]==96:
                out_cnn_sub=out_sub
        # input.shape: torch.Size([8, 1, 15, 96, 96])
        # out.shape: torch.Size([8, 32, 15, 96, 96])
        # out.shape: torch.Size([8, 64, 7, 48, 48])
        # out.shape: torch.Size([8, 128, 3, 24, 24])
        # mid_out.shape: torch.Size([8, 256, 1, 12, 12])
        # out.shape: torch.Size([8, 128, 24, 24])
        # out.shape: torch.Size([8, 64, 48, 48])
        # out.shape: torch.Size([8, 32, 96, 96])
        # F_out.shape: torch.Size([8, 3, 96, 96])
        output_sub_0 = self.fl_sub_0(out_sub)
        output_sub_1 = self.fl_sub_1(out_sub)
        for b_inx in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            if b_inx == 0:
                out = self.TransUpBlocks[b_inx](skip, out[:, :, 0])
            else:
                out = self.TransUpBlocks[b_inx](skip, out)
            out = torch.cat([skip_connection_up_sub[b_inx], out], 1)
            # print("out_shape:",out.shape)
            out = self.BlocksUp[b_inx](out)


            if out.shape[1]==32 and out.shape[2]==96:
                out_cnn=out
        output = self.fl(out)


        return output, out_cnn,output_sub_0,output_sub_1,out_cnn_sub

class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=15):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)
class Attention_head(nn.Module):
    def __init__(self,in_channels=[1,15],out_channels=[7,12,32,64],stride =1,p=0.5):
        super().__init__()
        self.conv3_1 = conv_33(out_channels[0],out_channels[1],stride)
        self.conv3_2 = conv_33(out_channels[1],out_channels[2],stride)
        self.BN_1 = nn.BatchNorm2d(out_channels[1])
        self.BN_2 = nn.BatchNorm2d(out_channels[2])
        self.conv1 = conv_11(out_channels[2],out_channels[3],stride)
        # self.relu = nn.functional.leaky_relu()
        # self.sigmoid = nn.functional.sigmoid()
    def forward(self, x):

        out = self.conv3_1(x)
        out = self.BN_1(out)
        out = nn.functional.leaky_relu(out)
        out = self.conv3_1(out)
        out = self.BN_1(out)
        out = nn.functional.leaky_relu(out)
        out = self.conv1(out)
        out = nn.functional.sigmoid(out)

        return out
class Segment_head(nn.Module):
    def __init__(self,in_channels=[1,15],out_channels=[7,12,32,64],stride =1,p=0.5):
        super().__init__()
        self.conv3_1 = conv_33(out_channels[0],out_channels[1],stride)
        self.conv3_2 = conv_33(out_channels[1],out_channels[2],stride)
        self.BN_1 = nn.BatchNorm2d(out_channels[1])
        self.BN_2 = nn.BatchNorm2d(out_channels[2])
        self.conv1 = conv_11(out_channels[2],out_channels[3],stride)
        # self.relu = nn.functional.leaky_relu()
        # self.sigmoid = nn.functional.sigmoid()
    def forward(self, x):

        out = self.conv3_1(x)
        out = self.BN_1(out)
        out = nn.functional.leaky_relu(out)
        out = self.conv3_1(out)
        out = self.BN_1(out)
        out = nn.functional.leaky_relu(out)
        out = self.conv1(out)
        # out = self.sigmoid(out)

        return out

class DenseNet_Branch(nn.Module):
    def __init__(self,in_channels=[1,15],out_channels=[3,5,7,12,32,64],stride =1,p=0.5):
        super().__init__()
        self.dense_1 =  ResBlock2D(in_channels[0],out_channels[0],stride=1,p=p)
        self.dense_2 =  ResBlock2D(out_channels[0],out_channels[1],stride=1,p=p)
        self.dense_3 =  ResBlock2D(out_channels[1],out_channels[2],stride=1,p=p)
        self.conv3_1 = conv_33(out_channels[2],out_channels[3],stride)
        self.conv3_2 = conv_33(out_channels[3],out_channels[4],stride)
        self.BN_1 = nn.BatchNorm2d(out_channels[3])
        self.BN_2 = nn.BatchNorm2d(out_channels[4])
        self.conv1 = conv_11(out_channels[4],out_channels[5],stride)
        # self.relu = nn.functional.leaky_relu()

    def forward(self, x):
        out1 = self.dense_1(x)
        out2 = self.dense_2(out1)
        out3 = self.dense_3(out2)
        out4 = self.conv3_1(out3)
        out5 = self.BN_1(out4)
        out6 = nn.functional.leaky_relu(out5)
        out7 = self.conv3_2(out6)
        out8 = self.BN_2(out7)
        out9 = nn.functional.leaky_relu(out8)
        out = self.conv1(out9)

        return out



class Time_Series(nn.Module):
    """ Res UNet class """
    def __init__(self,  filter_size=5, num_features=32,input_size=96,shape=[96,96], in_channels=[1,15], out_channels=[3,5,7,12,32,64,3],
                 down_blocks=[32, 64, 128],bottleneck = 256,up_blocks = [128, 64, 32], stride =1, p=0.5,n_slices=15):
        super().__init__()
        self.Encoder = ResUNet_out_2d(in_channels=1, out_channels=3, n_slices=n_slices, input_size=input_size,
                                     down_blocks=down_blocks, up_blocks = up_blocks, bottleneck = bottleneck, p=p)
        self.Clstm = CLSTM_cell(shape,up_blocks[-1],filter_size,num_features)
        self.Decoder = Resblock(in_channels=up_blocks[-1], out_channels=[up_blocks[-1],1], p=p)
        self.Attention = Attention_head(in_channels = [up_blocks[-1]],out_channels=[up_blocks[-1],up_blocks[-1],up_blocks[-1],out_channels[-2]] )
        self.Segment = Segment_head(in_channels = [up_blocks[-1]],out_channels=[up_blocks[-1],up_blocks[-1],up_blocks[-1],out_channels[-2]])
        self.Dense_branch = DenseNet_Branch(in_channels=in_channels,out_channels=out_channels,stride =stride,p=p)
        self.Block = Resblock(in_channels=out_channels[-2], out_channels=[out_channels[-2],out_channels[-1]], p=p)

        _initialize_weights_3d(self)
        _initialize_weights_2d(self)


    def forward(self, x):
        input_feature = []
        decode_outs=[]
        for i in range(x.size(1)):
            input =x[:,i,:,:]
            input = torch.unsqueeze(input,1)
            # print("x_shape:",x.shape)
            # input = x.transpose(1,0,2,3)
            # print("input_shape1:",input.shape)
            # input=input[i]
            # print("input_shape2:",input.shape)
            # input = input.transpose[1,0,2,3]
            # print("input_shape3:",input.shape) input_shape3: torch.Size([batch, 1, 96, 96])
            out_1_encode = self.Encoder(input)#shape: input [batch,1,96,96] out [batch,32,96,96]
            input_feature.append(out_1_encode)
        input_feature = torch.stack(input_feature).cuda()
        # print("input_feature:",input_feature.shape) input_feature: torch.Size([15, batch, 32, 96, 96])
        out_series,out_clstm = self.Clstm(input_feature)

        # print("out_clstm:",out_clstm[0].shape) out_clstm: torch.Size([batch, 32, 96, 96])
        for i in range(out_series.size(1)):
            input=out_series[:,i,:,:]
            # print("input decode shape:",input.shape) input decode shape: torch.Size([15, 32, 96, 96])

            out_1_decode,_ = self.Decoder(input)#shape: input [batch,32,96,96] out [batch,3,96,96]
            decode_outs.append(out_1_decode)
        decode_outs = torch.stack(decode_outs).cuda()
        # print("output decode shape:",decode_outs.shape) output decode shape: torch.Size([11, 15, 1, 96, 96])
        out_attention = self.Attention(out_clstm[0])#[batch,64,96,96]
        out_seg = self.Segment(out_clstm[0])#[batch,64,96,96]

        # print("first_brach finish")


        input =x[:,7,:,:]
        input = torch.unsqueeze(input,1)
        out_2 = self.Dense_branch(input)#[batch,64,96,96]
        out_mul_1 = torch.mul(out_attention,out_seg)
        out_mul_2 = torch.mul(out_attention,out_2)
        out_feature = torch.add(out_mul_1,out_mul_2)
        # print("out_feature_shape:",out_feature.shape)
        out,_ = self.Block(out_feature)
        # print("out.shape:",out.shape)





        return out,out_feature,decode_outs

class ResTime_Series(nn.Module):
    """ Res UNet class """
    def __init__(self,  filter_size=5, num_features=32,input_size=96,shape=[96,96], in_channels=[1,15], out_channels=[3,5,7,12,32,64,3],
                 down_blocks=[32, 64, 128],bottleneck = 256,up_blocks = [128, 64, 32], down_blocks_2=[64, 128, 256],bottleneck_2 = 512,up_blocks_2 = [256,128, 64],stride =1, p=0.5,n_slices=15):
        super().__init__()
        self.Encoder = ResUNet_out_2d(in_channels=1, out_channels=3, n_slices=n_slices, input_size=input_size,
                                      down_blocks=down_blocks, up_blocks = up_blocks, bottleneck = bottleneck, p=p)
        self.Clstm = CLSTM_cell(shape,up_blocks[-1],filter_size,num_features)
        self.Decoder = Resblock(in_channels=up_blocks[-1], out_channels=[up_blocks[-1],1], p=p)
        self.Attention = Attention_head(in_channels = [up_blocks[-1]],out_channels=[up_blocks[-1],up_blocks[-1],up_blocks[-1],out_channels[-2]] )
        self.Segment = Segment_head(in_channels = [up_blocks[-1]],out_channels=[up_blocks[-1],up_blocks[-1],up_blocks[-1],out_channels[-2]])
        self.Dense_branch = ResUNet_out_2d(in_channels=1, out_channels=3, n_slices=n_slices, input_size=input_size,
                                           down_blocks=down_blocks_2, up_blocks = up_blocks_2, bottleneck = bottleneck_2, p=p)
        self.Block = Resblock(in_channels=out_channels[-2], out_channels=[out_channels[-2],out_channels[-1]], p=p)

        _initialize_weights_3d(self)
        _initialize_weights_2d(self)


    def forward(self, x):
        input_feature = []
        decode_outs=[]
        for i in range(x.size(1)):
            input =x[:,i,:,:]
            input = torch.unsqueeze(input,1)
            # print("x_shape:",x.shape)
            # input = x.transpose(1,0,2,3)
            # print("input_shape1:",input.shape)
            # input=input[i]
            # print("input_shape2:",input.shape)
            # input = input.transpose[1,0,2,3]
            # print("input_shape3:",input.shape)
            out_1_encode = self.Encoder(input)#shape: input [batch,1,96,96] out [batch,32,96,96]
            input_feature.append(out_1_encode)
        input_feature = torch.stack(input_feature).cuda()
        # print("input_feature:",input_feature.shape)
        out_series,out_clstm = self.Clstm(input_feature)

        # print("out_clstm:",out_clstm[0].shape)
        for i in range(out_series.size(1)):
            input=out_series[:,i,:,:]
            out_1_decode,_ = self.Decoder(input)#shape: input [batch,32,96,96] out [batch,3,96,96]
            decode_outs.append(out_1_decode)
        decode_outs = torch.stack(decode_outs).cuda()
        out_attention = self.Attention(out_clstm[0])#[batch,64,96,96]
        out_seg = self.Segment(out_clstm[0])#[batch,64,96,96]

        # print("first_brach finish")


        input =x[:,7,:,:]
        input = torch.unsqueeze(input,1)
        out_2 = self.Dense_branch(input)#[batch,64,96,96]
        out_mul_1 = torch.mul(out_attention,out_seg)
        out_mul_2 = torch.mul(out_attention,out_2)
        out_feature = torch.add(out_mul_1,out_mul_2)
        # print("out_feature_shape:",out_feature.shape)
        out,_ = self.Block(out_feature)
        # print("out.shape:",out.shape)





        return out,out_feature,decode_outs
class ConvLSTMClassifier(nn.Module):
    def __init__(self,  filter_size=5, num_features=32,input_size=96,shape=[96,96], in_channels=[1,15], out_channels=[3,5,7,12,32,64,3],
                 down_blocks=[32, 64, 128],bottleneck = 256,up_blocks = [128, 64, 32], down_blocks_2=[64, 128, 256],bottleneck_2 = 512,up_blocks_2 = [256,128, 64],stride =1, p=0.5,n_slices=15):
        super().__init__()
        self.Encoder = ResUNet_out_2d(in_channels=1, out_channels=3, n_slices=n_slices, input_size=input_size,
                                      down_blocks=down_blocks, up_blocks = up_blocks, bottleneck = bottleneck, p=p)
        self.Clstm = CLSTM_cell(shape,up_blocks[-1],filter_size,num_features)
        self.Decoder = Resblock(in_channels=up_blocks[-1], out_channels=[up_blocks[-1],1], p=p)
        self.Attention = Attention_head(in_channels = [up_blocks[-1]],out_channels=[up_blocks[-1],up_blocks[-1],up_blocks[-1],out_channels[-3]] )
        self.Segment = Segment_head(in_channels = [up_blocks[-1]],out_channels=[up_blocks[-1],up_blocks[-1],up_blocks[-1],out_channels[-3]])
        self.Linear = nn.Linear(in_features=up_blocks[-1], out_features=2)
        _initialize_weights_3d(self)
        _initialize_weights_2d(self)
    def forward(self, x):
        input_feature = []
        for i in range(x.size(1)):
            input =x[:,i,:,:]
            input = torch.unsqueeze(input,1)
            # print("x_shape:",x.shape)
            # input = x.transpose(1,0,2,3)
            # print("input_shape1:",input.shape)
            # input=input[i]
            # print("input_shape2:",input.shape)
            # input = input.transpose[1,0,2,3]
            # print("input_shape3:",input.shape)
            out_1_encode = self.Encoder(input)#shape: input [batch,1,96,96] out [batch,32,96,96]
            input_feature.append(out_1_encode)
        input_feature = torch.stack(input_feature).cuda()
        # print("input_feature:",input_feature.shape)
        out_series,out_clstm = self.Clstm(input_feature)
        out_attention = self.Attention(out_clstm[0])#[batch,64,96,96]
        out_seg = self.Segment(out_clstm[0])#[batch,64,96,96]
        out_mul_1 = torch.mul(out_attention,out_seg)







def ResUNet28(in_channels, out_channels, n_slices=63, input_size=96, p=0.0):
    return ResUNet(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[32, 64, 128, 256, 512], up_blocks = [512, 256, 128, 64, 32], bottleneck = 1024, p=p)
def Time_series( filter_size=5, num_features=32,shape=[96,96], input_size=96,in_channels=[1,15], out_channels=[3,5,7,12,32,64,3],
                down_blocks=[32, 64, 128],bottleneck = 256,up_blocks = [128, 64, 32], stride =1, p=0.5,n_slices=15):
    return Time_Series(filter_size=filter_size, num_features=num_features,shape=[96,96], in_channels=[1,15], out_channels=[3,5,7,12,32,64,3],
                       down_blocks=[32, 64, 128],bottleneck = 256,up_blocks = [128, 64, 32], stride =1, p=0.5)
def ResTime_series( filter_size=5, num_features=32,shape=[96,96], input_size=96,in_channels=[1,15], out_channels=[3,5,7,12,32,64,3],
                 down_blocks=[32, 64, 128],bottleneck = 256,up_blocks = [128, 64, 32],down_blocks_2=[64, 128, 256],bottleneck_2 = 512,up_blocks_2 = [256,128, 64], stride =1, p=0.5,n_slices=15):
    return ResTime_Series(filter_size=filter_size, num_features=num_features,shape=[96,96], in_channels=[1,15], out_channels=[3,5,7,12,32,64,3],
                       down_blocks=[32, 64, 128],bottleneck = 256,up_blocks = [128, 64, 32],down_blocks_2=[64, 128, 256],bottleneck_2 = 512,up_blocks_2 = [256,128, 64], stride =1, p=0.5)
def ResUNet23(in_channels, out_channels, n_slices=31, input_size=96, p=0.0):
    return ResUNet(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[32, 64, 128, 256], up_blocks = [256, 128, 64, 32], bottleneck = 512, p=p)
def ResUNet18_5(in_channels=1, out_channels=5, n_slices=15, input_size=96, p=0.0):
    return ResUNet(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[64, 128, 256], up_blocks = [256, 128, 64], bottleneck = 512, p=p)

def ResUNet18(in_channels=1, out_channels=3, n_slices=15, input_size=96, p=0.0):
    return ResUNet(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[64, 128, 256], up_blocks = [256, 128, 64], bottleneck = 512, p=p)
def ResUNet18_64out(in_channels=1, out_channels=3, n_slices=15, input_size=96, p=0.0):
    return ResUNet_64out(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[64, 128, 256], up_blocks = [256, 128, 64], bottleneck = 512, p=p)
def ResUNet18_tiny(in_channels=1, out_channels=3, n_slices=15, input_size=96, p=0.0):
    return ResUNet(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[32,64,128], up_blocks = [128,64,32], bottleneck = 256, p=p)
def ResUNet18_Double(in_channels=1, out_channels=3, n_slices=15, input_size=96, p=0.0):
    return ResUNet_Double(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[32,64,128], up_blocks = [ 128, 64,32],up_blocks_seg = [128, 64,32], bottleneck = 256, p=p)
def ResUNet18_Double_Skip(in_channels=1, out_channels=3,out_channels_sub = 3, n_slices=15, input_size=96, p=0.0):
    return ResUNet_Double_Skip(in_channels=in_channels, out_channels=out_channels,out_channels_sub = out_channels_sub, n_slices=n_slices, input_size=input_size,
                   down_blocks=[32,64,128], up_blocks = [ 128, 64,32],up_blocks_sub = [128, 64,32], bottleneck = 256, p=p)
def ResUNet18_Double_Skip_2out(in_channels=1, out_channels=3, n_slices=15, input_size=96, p=0.0):
    return ResUNet_Double_Skip_2out(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                               down_blocks=[32,64,128], up_blocks = [ 128, 64,32],up_blocks_sub = [128, 64,32], bottleneck = 256, p=p)
def ResUNet18_Cls(in_channels=1, out_channels=[10,2], n_slices=15, input_size=96, p=0.0):
    return ResUNet_Cls(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[32,64,128], up_blocks = [ 128, 64,32], bottleneck = 256, p=p)
def ResUNet18_3Cls(in_channels=1, out_channels=[10,3], n_slices=15, input_size=96, p=0.0):
    return ResUNet_Cls(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[32,64,128], up_blocks = [ 128, 64,32], bottleneck = 256, p=p)
def SkipResUNet18_3Cls(in_channels=1, out_channels=[10,3], n_slices=15, input_size=96, p=0.0):
    return SkipResUNet_Cls(in_channels=in_channels, out_channels=out_channels, n_slices=n_slices, input_size=input_size,
                   down_blocks=[32,64,128], up_blocks = [ 128, 64,32],up_blocks_sub = [128, 64,32], bottleneck = 256, p=p)
def ResUNet18_block(in_channels=64, out_channels=[64,3], p=0.0):
    return Resblock(in_channels=in_channels, out_channels=out_channels, p=p)
def ResUNet_block_32_3(in_channels=32, out_channels=[32,3], p=0.0):
    return Resblock(in_channels=in_channels, out_channels=out_channels, p=p)
def ResUNet_block_32_5(in_channels=32, out_channels=[32,5], p=0.0):
    return Resblock(in_channels=in_channels, out_channels=out_channels, p=p)
def ResUNet_block_32_2(in_channels=32, out_channels=[32,2], p=0.0):
    return Resblock(in_channels=in_channels, out_channels=out_channels, p=p)
def ResUNet_block_seg_32_2(in_channels=32, out_channels=[32,2], p=0.0):
    return Resblock_seg(in_channels=in_channels, out_channels=out_channels, p=p)

if __name__ == "__main__":
    in_channels = 1
    out_channels = 3
    n_slices = 15
    input_size = 96
    unet = ResUNet18(in_channels, out_channels, n_slices=n_slices, input_size=input_size)
    print(unet)
    x = torch.FloatTensor(6, in_channels, n_slices, input_size, input_size)  # the smallest patch size is 12 * 12
    y = unet(x)