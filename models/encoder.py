# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: encoder.py
@Time    : 2025/4/18 上午9:50
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
import torch.nn as nn
from models import VisualEncoder, BEVTransformer, BEVLinearAttention

class DualInputVisualEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = VisualEncoder(cfg)
        # 添加1×1卷积层用于统一通道
        self.conv_list = nn.ModuleList([nn.Conv2d(in_dim, out_dim, kernel_size=1) for in_dim, out_dim in zip([128, 256, 512, 1024], cfg.model.out_dims)])
        # self.bev_transformers = nn.ModuleList([
        #     BEVTransformer(in_channels=out_dim, out_channels=out_dim) for out_dim in cfg.model.out_dims
        # ])
        self.bev_transformers = nn.ModuleList([
            BEVLinearAttention(in_channels=out_dim, out_channels=out_dim) for out_dim in cfg.model.out_dims
        ])

    def forward(self, image1, image2):
        # 提取两幅图像的多尺度特征
        feats1 = self.encoder(image1)
        feats2 = self.encoder(image2)
        # 统一通道
        feats1 = [self.conv_list[i](feat) for i, feat in enumerate(feats1)]
        feats2 = [self.conv_list[i](feat) for i, feat in enumerate(feats2)]
        # 对每个尺度的特征进行 BEV 转换
        multi_scale_bev_feats1 = []
        multi_scale_bev_feats2 = []
        for i in range(len(feats1)):
            feat1_bev = self.bev_transformers[i](feats1[i])
            feat2_bev = self.bev_transformers[i](feats2[i])
            multi_scale_bev_feats1.append(feat1_bev)
            multi_scale_bev_feats2.append(feat2_bev)
        return multi_scale_bev_feats1, multi_scale_bev_feats2


# 测试
if __name__ == '__main__':
    import torch
    from utils import load_config
    cfg = load_config('../configs/config.yaml')
    model = DualInputVisualEncoder(cfg).to('cuda')
    image1 = torch.randn(2, 3, 512, 512).to('cuda')
    image2 = torch.randn(2, 3, 512, 512).to('cuda')
    feats1, feats2 = model(image1, image2)
    for i in range(len(feats1)):
        print(feats1[i].shape, feats2[i].shape)

    from thop import profile
    flops, params = profile(model, inputs=(image1, image2))
    print(f"encoder FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")