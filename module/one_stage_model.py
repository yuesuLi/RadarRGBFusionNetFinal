import os
import argparse
import datetime
import shutil

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



class one_stage_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.inchannel = 64
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        # self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        # self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        # self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        # self.fc = nn.Linear(512, num_classes)
        # self.conv77 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.conv55 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU()
        # )
        # self.conv33 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        #     # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
        #     # nn.BatchNorm2d(256),
        #     # nn.ReLU()
        # )




        self.inc = DoubleConv(2, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.onestage_head = nn.Sequential(
            # nn.Conv2d(512, 256, kernel_size=3, padding=1),
            # nn.Conv2d(256, 128, kernel_size=3, padding=1),
            # nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )


    # # 这个函数主要是用来，重复同一个残差块
    # def make_layer(self, block, channels, num_blocks, stride):
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.inchannel, channels, stride))
    #         self.inchannel = channels
    #     return nn.Sequential(*layers)

    def forward(self, dynamic_HM, static_HM):
        '''
        :param dynamic_HM: (B, h=256, w=256, channels=3)
        :param static_HM: (B, h=256, w=256, channels=3)
        :return: y: (B, )
        '''
        # batch_size = dynamic_HM.shape[0]

        dynamic_HM = dynamic_HM.permute(0, 3, 1, 2)  # (B, channels=3, h=256, w=256)
        static_HM = static_HM.permute(0, 3, 1, 2)  # (B, channels=3, h=256, w=256)

        fusion_heatmap = torch.cat([static_HM, dynamic_HM], dim=1)  # (B, 2, 128, 128)

        f1 = self.inc(fusion_heatmap)  # (B, 64, 128, 128)
        # print('shape: ', f1.cpu().detach().numpy().shape)
        f2 = self.down1(f1)  # (B, 128, 64, 64)
        # print('shape: ', f2.cpu().detach().numpy().shape)
        f3 = self.down2(f2)  # (B, 256, 32, 32)
        # print('shape: ', f3.cpu().detach().numpy().shape)
        f4 = self.down3(f3)  # (B, 512, 16, 16)
        # print('shape: ', f4.cpu().detach().numpy().shape)
        f = self.up1(f4, f3)  # (B, 256, 32, 32)
        # print('shape: ', f.cpu().detach().numpy().shape)
        f = self.up2(f, f2)  # (B, 128, 64, 64)
        # print('shape: ', f.cpu().detach().numpy().shape)
        f = self.up3(f, f1)  # (B, 64, 128, 128)
        # print('shape: ', f.cpu().detach().numpy().shape)

        # d1 = self.conv77(dynamic_HM)
        # d2 = self.conv55(d1)
        # d = self.conv33(d2)
        #
        # s1 = self.conv77(static_HM)
        # s2 = self.conv55(s1)
        # s = self.conv33(s2)

        # (B*range=B*256, doppler=64, elevation=32, azimuth=128)
        # x.resize_(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        # d1 = self.conv1(dynamic_HM)  # (B, 64, 256, 256)
        # d2 = self.layer1(d1)    # (B, 64, 256, 256)
        # d3 = self.layer2(d2)    # (B, 128, 128, 128)
        # d4 = self.layer3(d3)  # (B, 256, 64, 64)
        # d5 = self.layer4(d4)  # (B, 512, 32, 32)
        #
        # d1 = self.inc(fusion_heatmap)  # (B, 64, 256, 256)
        # d2 = self.down1(d1)  # (B, 128, 128, 128)
        # d3 = self.down2(d2)  # (B, 256, 64, 64)
        # d4 = self.down3(d3)  # (B, 512, 32, 32)
        # d = self.up1(d4, d3)  # (B, 256, 64, 64)
        # d = self.up2(d, d2)  # (B, 128, 128, 128)
        #
        # s1 = self.conv1(static_HM)  # (B, 64, 256, 256)
        # s2 = self.layer1(s1)  # (B, 64, 256, 256)
        # s3 = self.layer2(s2)  # (B, 128, 128, 128)
        # s4 = self.layer3(s3)  # (B, 256, 64, 64)
        # s5 = self.layer4(s4)  # (B, 512, 32, 32)
        #
        #
        # s1 = self.inc(static_HM)  # (B, 64, 256, 256)
        # s2 = self.down1(s1)  # (B, 128, 128, 128)
        # s3 = self.down2(s2)  # (B, 256, 64, 64)
        # s4 = self.down3(s3)  # (B, 512, 32, 32)
        # s = self.up1(s4, s3)  # (B, 256, 64, 64)
        # s = self.up2(s, s2)  # (B, 128, 128, 128)

        # print('d', d.shape)
        # print('s', s.shape)

        # HM_feature = torch.cat([d, s], dim=1)   # (B, 128*2, 128, 128)
        # print('HM_feature', HM_feature.shape)
        #
        y = self.onestage_head(f)  # (B, 1, 128, 128)
        # y = y.squeeze(1)    # (B, 128, 128)
        # Center_HM_height = y.shape[2]
        # Center_HM_width = y.shape[3]

        return y

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


# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.MaxPool2d(2)
            )
        else:
            self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        # self.fc = nn.Linear(512, num_classes)

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
