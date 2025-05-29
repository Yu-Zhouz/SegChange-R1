# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: feature_diff.py
@Time    : 2025/4/18 上午9:51
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 特征对比
@Usage   :
"""
import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class SE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输出 (B, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        weight = self.avg_pool(x)  # Squeeze
        weight = self.fc(weight)  # Excitation -> (B, C, 1, 1)
        return x * weight  # Reweight


class CBAM(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=7, padding=3, groups=channel // ratio),
            nn.BatchNorm2d(channel // ratio),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, 1, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * self.sigmoid(ca)
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * self.sigmoid(sa)
        return x


class FeatureDiffModule(nn.Module):
    def __init__(self, in_channels_list, diff_attention='SE', use_fusion=False):
        super().__init__()
        self.use_fusion = use_fusion

        # 输入是 feats1[i] 和 feats2[i] 的 concat + diff + product
        self.diff_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),  # 只保留 concat + diff/sim
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in in_channels_list
        ])

        if diff_attention == 'CBAM':
            self.attention = nn.ModuleList([
                CBAM(in_channels) for in_channels in in_channels_list  # better than original attention
            ])
        elif diff_attention == 'SE':
            self.attention = nn.ModuleList([
                SE(in_channels) for in_channels in in_channels_list
            ])
        else:
            raise ValueError('diff_attention must be CBAM or SE')

    def forward(self, feats1: List[torch.Tensor], feats2: List[torch.Tensor]) -> List[torch.Tensor]:
        diff_feats = []
        for i in range(len(feats1)):
            f1, f2 = feats1[i], feats2[i]
            diff_l1 = torch.abs(f1 - f2)
            sim = f1 * f2
            concat = torch.cat([f1, f2, diff_l1, sim], dim=1)
            diff = self.diff_conv[i](concat)
            diff = self.attention[i](diff)
            diff_feats.append(diff)

        return diff_feats


# 测试
if __name__ == '__main__':
    model = FeatureDiffModule(in_channels_list=[256, 256, 256, 256]).to('cuda')
    feats1 = [torch.randn(2, 256, 128, 128).to('cuda'),
              torch.randn(2, 256, 64, 64).to('cuda'),
              torch.randn(2, 256, 32, 32).to('cuda'),
              torch.randn(2, 256, 16, 16).to('cuda')]
    feats2 = [torch.randn(2, 256, 128, 128).to('cuda'),
              torch.randn(2, 256, 64, 64).to('cuda'),
              torch.randn(2, 256, 32, 32).to('cuda'),
              torch.randn(2, 256, 16, 16).to('cuda')]
    import time

    start = time.time()
    diff_feats = model(feats1, feats2)
    last = time.time()
    for i in range(len(diff_feats)):
        print(diff_feats[i].shape)

    from thop import profile

    flops, params = profile(model, inputs=(feats1, feats2))
    print(f"encoder FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M, time: {(last - start):.2f} s")
