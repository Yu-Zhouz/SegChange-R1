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
from torchvision.models import resnet50


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


class ResNet50Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 使用ResNet50作为视觉编码器
        self.backbone = resnet50(pretrained=cfg.model.pretrained)
        # 获取ResNet50的各层输出维度
        self.out_dims = [64, 256, 512, 1024, 2048]

        # 定义各特征提取层
        self.layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

    def forward(self, x):
        """
        x: [B,3,H,W]
        returns list of multi-scale features:
          feats[i] is [B, C_i, H/2^{i}, W/2^{i}]
        """
        feats = []
        x = self.layer0(x)
        x = self.layer1(x)
        feats.append(x)  # C=256, 1/4分辨率
        x = self.layer2(x)
        feats.append(x)  # C=512, 1/8分辨率
        x = self.layer3(x)
        feats.append(x)  # C=1024, 1/16分辨率
        x = self.layer4(x)
        feats.append(x)  # C=2048, 1/32分辨率
        return feats


# 测试
if __name__ == '__main__':
    import torch
    from utils import load_config

    cfg = load_config('../configs/config.yaml')
    # 测试Swin Transformer
    model_swin = VisualEncoder(cfg)
    x = torch.randn(2, 3, 512, 512)  # 示例输入：2张3通道512x512的图像
    feats_swin = model_swin(x)
    print("Swin Transformer Features:")
    for i, feat in enumerate(feats_swin):
        print(f"Layer {i} feature shape: {feat.shape}")

    # 测试ResNet50
    model_resnet = ResNet50Encoder(cfg)
    feats_resnet = model_resnet(x)
    print("\nResNet50 Features:")
    for i, feat in enumerate(feats_resnet):
        print(f"Layer {i} feature shape: {feat.shape}")

    # 计算FLOPs和Params
    from thop import profile

    flops_swin, params_swin = profile(model_swin, inputs=(x,))
    print(f"\nSwin Transformer Backbone FLOPs: {flops_swin / 1e9:.2f} G, Params: {params_swin / 1e6:.2f} M")

    flops_resnet, params_resnet = profile(model_resnet, inputs=(x,))
    print(f"ResNet50 Backbone FLOPs: {flops_resnet / 1e9:.2f} G, Params: {params_resnet / 1e6:.2f} M")