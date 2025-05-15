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
from models import TokenConnector, EGA


class FPNFeatureFuser(nn.Module):
    def __init__(self, in_channels, fpn_out_channels=256, feature_strides=[4, 8, 16, 32], use_token_connector=False, use_ega=True):
        super().__init__()
        assert len(in_channels) == len(feature_strides), "通道数和stride数量必须一致"
        self.use_token_connector = use_token_connector
        self.use_ega = use_ega  # 是否使用 EGA
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

        if self.use_ega:
            self.ega = EGA(channel=fpn_out_channels, size=3, sigma=1.0)

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

        if self.use_ega:
            fused = self.ega(fused)

        return fused


class LightweightFPN(nn.Module):
    def __init__(self, in_channels, fpn_out_channels=128, feature_strides=[4, 8, 16, 32], use_token_connector=False, use_ega=True):
        super().__init__()
        self.use_token_connector = use_token_connector
        self.use_ega = use_ega  # 是否使用 EGA
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
        self.fusion_conv = nn.Conv2d(fpn_out_channels * len(in_channels), fpn_out_channels, kernel_size=1)

        # TokenConnector 放在最后
        if self.use_token_connector:
            self.token_connector = TokenConnector(fpn_out_channels, fpn_out_channels // 2)

        if self.use_ega:
            self.ega = EGA(channel=fpn_out_channels, size=3, sigma=1.0)

    def forward(self, features):
        upsampled = [head(feat) for feat, head in zip(features, self.scale_heads)]
        merged = torch.cat(upsampled, dim=1)
        fused = self.fusion_conv(merged)

        # TokenConnector
        if self.use_token_connector:
            fused = self.token_connector(fused)

        if self.use_ega:
            fused = self.ega(fused)

        return fused



if __name__ == '__main__':
    multi_scale_feats = [torch.randn(2, 256, 128, 128).to('cuda:1'),
                         torch.randn(2, 256, 64, 64).to('cuda:1'),
                         torch.randn(2, 256, 32, 32).to('cuda:1'),
                         torch.randn(2, 256, 16, 16).to('cuda:1')]
    fpn_feature_fuser = FPNFeatureFuser(in_channels=[256, 256, 256, 256]).to('cuda:1')
    import time
    start = time.time()
    fpn_feats = fpn_feature_fuser(multi_scale_feats)
    last = time.time()
    print(fpn_feats.shape)

    from thop import profile
    flops, params = profile(fpn_feature_fuser, inputs=(multi_scale_feats,))
    print(f"encoder FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M, time: {(last - start):.2f} s")

    # 示例输入
    multi_scale_feats = [torch.randn(2, 256, 128, 128).to('cuda:1'),
                         torch.randn(2, 256, 64, 64).to('cuda:1'),
                         torch.randn(2, 256, 32, 32).to('cuda:1'),
                         torch.randn(2, 256, 16, 16).to('cuda:1')]

    # 创建轻量化FPN
    light_fpn = LightweightFPN(in_channels=[256, 256, 256, 256]).to('cuda:1')

    # 测试
    import time

    start = time.time()
    fpn_feats = light_fpn(multi_scale_feats)
    last = time.time()
    print(fpn_feats.shape)
    from thop import profile
    flops, params = profile(light_fpn, inputs=(multi_scale_feats,))
    print(f"encoder FLOPs: {flops / 1e9:.2f} G, Params:  {params / 1e6:.2f} M, time: {(last - start):.2f} s")