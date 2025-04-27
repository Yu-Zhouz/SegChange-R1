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


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.scale = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        attention = self.conv(x)
        attention = self.softmax(attention)
        attention = self.scale(attention)
        return x * attention


class FeatureDiffModule(nn.Module):
    def __init__(self, in_channels_list, use_attention=True, use_fusion=False):
        super().__init__()
        self.use_attention = use_attention
        self.use_fusion = use_fusion

        self.diff_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in in_channels_list
        ])

        if self.use_attention:
            self.attention = nn.ModuleList([
                AttentionModule(in_channels) for in_channels in in_channels_list
            ])

        if self.use_fusion:
            self.fusion_conv = nn.Conv2d(sum(in_channels_list), in_channels_list[-1], kernel_size=1)

    def forward(self, feats1, feats2):
        diff_feats = []
        for i in range(len(feats1)):
            concat = torch.cat([feats1[i], feats2[i]], dim=1)
            diff = self.diff_conv[i](concat)

            if self.use_attention:
                diff = self.attention[i](diff)

            diff_feats.append(diff)

        if self.use_fusion:
            # 在特征融合时，将所有尺度的特征图 resize 到最小尺度
            min_h, min_w = diff_feats[-1].shape[2:]
            fused_feats = [F.interpolate(feat, size=(min_h, min_w), mode='bilinear', align_corners=True) for feat in diff_feats]
            fused_feats = torch.cat(fused_feats, dim=1)
            fused_feats = self.fusion_conv(fused_feats)
            diff_feats.append(fused_feats)

        return diff_feats  # 返回多尺度特征差异和融合后的特征


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
    diff_feats = model(feats1, feats2)
    for i in range(len(diff_feats)):
        print(diff_feats[i].shape)

    from thop import profile
    flops, params = profile(model, inputs=(feats1, feats2))
    print(f"diff FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")