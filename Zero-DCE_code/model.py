import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

'''ZDCE original implementation'''


class enhance_net_nopool(nn.Module):

    def __init__(self, in_channels=3):
        super(enhance_net_nopool, self, ).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(in_channels, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f*2, 24, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(
            2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        xr, xg1, xb, xg2 = torch.split(x, 1, dim=1)  # RGGB
        rgb = torch.cat([xr, (xg1+xg2)/2, xb], axis=1)

        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = rgb
        x = x + r1*(torch.pow(x, 2)-x)
        x = x + r2*(torch.pow(x, 2)-x)
        x = x + r3*(torch.pow(x, 2)-x)
        enhance_image_1 = x + r4*(torch.pow(x, 2)-x)
        x = enhance_image_1 + r5 * \
            (torch.pow(enhance_image_1, 2)-enhance_image_1)
        x = x + r6*(torch.pow(x, 2)-x)
        x = x + r7*(torch.pow(x, 2)-x)
        enhance_image = x + r8*(torch.pow(x, 2)-x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r


'''UNet https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py'''


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv(x))


class UNet(nn.Module):
    def __init__(self, in_channels=4, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 3
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.out_channels)

    @staticmethod
    def weights_init(m):
        pass  # XXX

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


'''My Curve'''


class ll_enhance_net_nopool(nn.Module):

    def __init__(self, in_channels=3):
        super(ll_enhance_net_nopool, self, ).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(in_channels, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f*2, 32, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(
            2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        xr, xg1, xb, xg2 = torch.split(x, 1, dim=1)  # RGGB
        rgb = torch.cat([xr, (xg1+xg2)/2, xb], axis=1)

        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = self.e_conv7(torch.cat([x1, x6], 1))
        R1, R2, R3, R4, R5, R6, R7, R8 = torch.split(x_r, 4, dim=1)
        b1, r1 = R1[:, 0:1], R1[:, 1:]
        b2, r2 = R2[:, 0:1], R2[:, 1:]
        b3, r3 = R3[:, 0:1], R3[:, 1:]
        b4, r4 = R4[:, 0:1], R4[:, 1:]
        b5, r5 = R5[:, 0:1], R5[:, 1:]
        b6, r6 = R6[:, 0:1], R6[:, 1:]
        b7, r7 = R7[:, 0:1], R7[:, 1:]
        b8, r8 = R8[:, 0:1], R8[:, 1:]
        
        b1, b2, b3, b4, b5, b6, b7, b8 = [F.tanh(v) for v in [b1, b2, b3, b4, b5, b6, b7, b8]]
        r1, r2, r3, r4, r5, r6, r7, r8 = [self.sigmoid(v) for v in [r1, r2, r3, r4, r5, r6, r7, r8]]

        x = rgb
        x = b1*(x + r1*(torch.pow(x, 2)-x)) + (1-b1)
        x = b2*(x + r2*(torch.pow(x, 2)-x)) + (1-b2)
        x = b3*(x + r3*(torch.pow(x, 2)-x)) + (1-b3)
        enhance_image_1 = b4 * (x + r4*(torch.pow(x, 2)-x)) + (1-b4)
        x = b5 * (enhance_image_1 + r5 *
                  (torch.pow(enhance_image_1, 2)-enhance_image_1)) + (1-b5)
        x = b6*(x + r6*(torch.pow(x, 2)-x)) + (1-b6)
        x = b7*(x + r7*(torch.pow(x, 2)-x)) + (1-b7)
        enhance_image = b8 * (x + r8*(torch.pow(x, 2)-x)) - (1-b8)
        r = torch.cat([R1, R2, R3, R4, R5, R6, R7, R8], 1)
        return enhance_image_1, enhance_image, r



'''My Curve + CC'''


class ll_cc_enhance_net_nopool(nn.Module):

    def __init__(self, in_channels=3):
        super(ll_cc_enhance_net_nopool, self, ).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(in_channels, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f*2, 32, 3, 1, 1, bias=True)
        
        self.ccm = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8,32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 3*3),
            nn.Sigmoid()
        )

        self.maxpool = nn.MaxPool2d(
            2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        xr, xg1, xb, xg2 = torch.split(x, 1, dim=1)  # RGGB
        rgb = torch.cat([xr, (xg1+xg2)/2, xb], axis=1)

        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = self.e_conv7(torch.cat([x1, x6], 1))
        R1, R2, R3, R4, R5, R6, R7, R8 = torch.split(x_r, 4, dim=1)
        b1, r1 = R1[:, 0:1], R1[:, 1:]
        b2, r2 = R2[:, 0:1], R2[:, 1:]
        b3, r3 = R3[:, 0:1], R3[:, 1:]
        b4, r4 = R4[:, 0:1], R4[:, 1:]
        b5, r5 = R5[:, 0:1], R5[:, 1:]
        b6, r6 = R6[:, 0:1], R6[:, 1:]
        b7, r7 = R7[:, 0:1], R7[:, 1:]
        b8, r8 = R8[:, 0:1], R8[:, 1:]
        
        b1, b2, b3, b4, b5, b6, b7, b8 = [F.tanh(v) for v in [b1, b2, b3, b4, b5, b6, b7, b8]]
        r1, r2, r3, r4, r5, r6, r7, r8 = [self.sigmoid(v) for v in [r1, r2, r3, r4, r5, r6, r7, r8]]

        x = rgb
        x = b1*(x + r1*(torch.pow(x, 2)-x)) + (1-b1)
        ccm = self.fc(self.ccm(F.interpolate(x, size=(256, 256), mode='bilinear')).view(x.size(0), -1)).view(-1,3,3)
        x = torch.einsum('bij,bjkm->bikm', ccm, x)
        x = b2*(x + r2*(torch.pow(x, 2)-x)) + (1-b2)
        x = b3*(x + r3*(torch.pow(x, 2)-x)) + (1-b3)
        enhance_image_1 = b4 * (x + r4*(torch.pow(x, 2)-x)) + (1-b4)
        x = b5 * (enhance_image_1 + r5 *
                  (torch.pow(enhance_image_1, 2)-enhance_image_1)) + (1-b5)
        x = b6*(x + r6*(torch.pow(x, 2)-x)) + (1-b6)
        x = b7*(x + r7*(torch.pow(x, 2)-x)) + (1-b7)
        enhance_image = b8 * (x + r8*(torch.pow(x, 2)-x)) - (1-b8)
        r = torch.cat([R1, R2, R3, R4, R5, R6, R7, R8], 1)
        return enhance_image_1, enhance_image, r

''' Sig UISP'''


class sig_enhance_net_nopool(nn.Module):

    def __init__(self, in_channels=3):
        super(sig_enhance_net_nopool, self, ).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.amplification = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8,32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
        )

        number_f = 32
        self.e_conv1 = nn.Conv2d(in_channels, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f*2, 24, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(
            2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        xr, xg1, xb, xg2 = torch.split(x, 1, dim=1)  # RGGB
        rgb = torch.cat([xr, (xg1+xg2)/2, xb], axis=1)

        amplification, bias = torch.split(self.amplification(F.interpolate(x, size=(256, 256), mode='bilinear')), 1, dim=1)

        x1 = self.relu(self.e_conv1(torch.clamp(x*amplification + bias, 0, 1)))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = rgb
        x = torch.clamp(x*amplification + bias, 0, 1)
        x = x + r1*(torch.pow(x, 2)-x)
        x = x + r2*(torch.pow(x, 2)-x)
        x = x + r3*(torch.pow(x, 2)-x)
        enhance_image_1 = x + r4*(torch.pow(x, 2)-x)
        x = enhance_image_1 + r5 * \
            (torch.pow(enhance_image_1, 2)-enhance_image_1)
        x = x + r6*(torch.pow(x, 2)-x)
        x = x + r7*(torch.pow(x, 2)-x)
        enhance_image = x + r8*(torch.pow(x, 2)-x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r

