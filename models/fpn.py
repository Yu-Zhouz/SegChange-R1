# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: fpn.py
@Time    : 2025/5/6 上午9:35
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
import torch
import torch.nn as nn
from models import TokenConnector


class FPNFeatureFuser(nn.Module):
    def __init__(self, in_channels, fpn_out_channels=256, feature_strides=[4, 8, 16, 32], use_token_connector=False):
        super().__init__()
        assert len(in_channels) == len(feature_strides), "通道数和stride数量必须一致"
        self.use_token_connector = use_token_connector

        # 上采样头：每个层级独立恢复到原图大小
        self.scale_heads = nn.ModuleList()
        for in_channel, stride in zip(in_channels, feature_strides):
            self.scale_heads.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, fpn_out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, fpn_out_channels),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
                )
            )

        # 最终融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_out_channels * len(in_channels), fpn_out_channels, kernel_size=1),
            nn.GroupNorm(32, fpn_out_channels),
            nn.ReLU()
        )

        # TokenConnector 放在最后
        if self.use_token_connector:
            self.token_connector = TokenConnector(fpn_out_channels, fpn_out_channels // 2)

    def forward(self, features):
        """
        features: list of [B, C_i, H_i, W_i]
        """
        upsampled = []
        for i, (feat, head) in enumerate(zip(features, self.scale_heads)):
            x = head(feat)
            upsampled.append(x)

        # 拼接所有上采样后的特征
        merged = torch.cat(upsampled, dim=1)  # [B, 4*C, H, W]

        # 融合
        fused = self.fusion_conv(merged)  # [B, C, H, W]

        # TokenConnector
        if self.use_token_connector:
            fused = self.token_connector(fused)

        return fused



if __name__ == '__main__':
    multi_scale_feats = [torch.randn(2, 256, 128, 128).to('cuda'),
                         torch.randn(2, 256, 64, 64).to('cuda'),
                         torch.randn(2, 256, 32, 32).to('cuda'),
                         torch.randn(2, 256, 16, 16).to('cuda')]
    fpn_feature_fuser = FPNFeatureFuser(in_channels=[256, 256, 256, 256]).to('cuda')
    fpn_feats = fpn_feature_fuser(multi_scale_feats)
    print(fpn_feats.shape)

    from thop import profile
    flops, params = profile(fpn_feature_fuser, inputs=(multi_scale_feats,))
    print(f'flops: {flops / 1e9}G, params: {params / 1e6}M')