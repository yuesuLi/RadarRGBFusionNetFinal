import os
import argparse
import datetime
import shutil

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



class LidarBEVBB(nn.Module):
    def __init__(self):
        super().__init__()

        self.inchannel = 64
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)

        self.head_hm = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.head_wh = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1),
        )
        self.head_ori = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1),
        )
        self.head_offset = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1),
        )


    def forward(self, LidarBEV):
        '''
        :param LidarBEV: (B, h=512, w=512, channels=3)
        :return: y: (B, )
        '''
        # batch_size = dynamic_HM.shape[0]

        LidarBEV = LidarBEV.permute(0, 3, 1, 2)  # (B, channels=3, h=512, w=512)


        f1 = self.inc(LidarBEV)  # (B, 64, 256, 256)
        # print('shape: ', f1.cpu().detach().numpy().shape)
        f2 = self.down1(f1)  # (B, 128, 128, 128)
        f3 = self.down2(f2)  # (B, 256, 64, 64)
        f4 = self.down3(f3)  # (B, 512, 32, 32)
        f5 = self.down4(f4)  # (B, 1024, 16, 16)
        f = self.up1(f5, f4)  # (B, 512, 32, 32)
        f = self.up2(f, f3)  # (B, 256, 64, 64)
        bb_feature = self.up3(f, f2)  # (B, 128, 128, 128)

        # print('d', d.shape)
        # print('s', s.shape)

        # HM_feature = torch.cat([d, s], dim=1)   # (B, 128*2, 128, 128)
        # print('HM_feature', HM_feature.shape)
        #
        hm_feature = self.head_hm(bb_feature)  # (B, 1, 128, 128)
        # wh_map = self.head_wh(bb_feature)    # (B, 2, 128, 128)
        # ori_map = self.head_ori(bb_feature)  # (B, 2, 128, 128)
        # offset_map = self.head_offset(f)    # (B, 2, 128, 128)
        # y = y.squeeze(1)    # (B, 128, 128)
        # Center_HM_height = y.shape[2]
        # Center_HM_width = y.shape[3]

        return bb_feature, hm_feature

# w and h keep constant, have two 3*3 conv
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# w / 2 and h /2
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# DoubleConv
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

