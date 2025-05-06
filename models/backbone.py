# backbone.py
# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: backbone.py
@Time    : 2025/4/17 下午3:19
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 使用Swin Transformer作为视觉编码器提取多尺度特征
@Usage   : https://huggingface.co/microsoft/swin-base-patch4-window7-224/tree/main
"""
import torch.nn as nn
from timm import create_model


class VisualEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 使用Swin Transformer作为视觉编码器
        self.backbone = create_model(model_name=cfg.model.backbone_name, features_only=True, out_indices=[0, 1, 2, 3],
                                     pretrained=cfg.model.pretrained, img_size=cfg.model.img_size)
        self.out_dims = cfg.model.out_dims

    def forward(self, x):
        """
        x: [B,3,H,W]
        returns list of multi-scale features:
          feats[i] is [B, C_i, H/2^{i+2}, W/2^{i+2}]
        """
        feats = self.backbone(x)
        # 将每个特征图的维度从 [B, H, W, C] 转换为 [B, C, H, W]
        feats = [feat.permute(0, 3, 1, 2).contiguous() for feat in feats]
        return feats


# 测试
if __name__ == '__main__':
    import torch
    from utils import load_config

    cfg = load_config('../configs/config.yaml')
    model = VisualEncoder(cfg)  # 指定输入尺寸为 512x512
    x = torch.randn(2, 3, 512, 512)  # 示例输入：2张3 通道512x512的图像
    feats = model(x)
    for i, feat in enumerate(feats):
        print(f"Layer {i} feature shape: {feat.shape}")

    from thop import profile

    flops, params = profile(model, inputs=(x,))
    print(f"Backbone FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
