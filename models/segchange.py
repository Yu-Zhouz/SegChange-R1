# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: segchange.py
@Time    : 2025/4/18 上午9:37
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 变化检测模型
@Usage   :
"""
from torch import nn
from models import MaskGenerator, DualInputVisualEncoder, FeatureDiffModule, build_textencoder
from models.losses import TotalLoss

class ChangeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg.device
        self.dual_encoder = DualInputVisualEncoder(self.cfg).to(self.device)
        self.feature_diff = FeatureDiffModule(in_channels_list=self.cfg.model.out_dims).to(self.device)
        self.mask_generator = MaskGenerator(
            in_channels=self.cfg.model.out_dims[-2],
            feature_strides=[4, 8, 16, 32],
            num_classes=1,
            lang_dim=2048 if cfg.model.model_name == 'microsoft/phi-1_5' else 768,
            n_heads=8
        ).to(self.device)
        TextEncoder = build_textencoder(cfg)
        self.llm = TextEncoder(model_name=cfg.model.model_name).to(self.device)

    def forward(self, image1, image2, prompts):
        # 确保输入数据在正确的设备上
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)

        # 提取多尺度 BEV 特征
        multi_scale_bev_feats1, multi_scale_bev_feats2 = self.dual_encoder(image1, image2)
        # 计算多尺度 BEV 特征差异
        multi_scale_diff_feats = self.feature_diff(multi_scale_bev_feats1, multi_scale_bev_feats2)
        # 处理文本描述
        description_embeddings, _ = self.llm(prompts)
        # 生成变化掩码
        mask = self.mask_generator(multi_scale_diff_feats, description_embeddings)
        return mask

    def to(self, device):
        self.device = device
        self.dual_encoder = self.dual_encoder.to(device)
        self.feature_diff = self.feature_diff.to(device)
        self.mask_generator = self.mask_generator.to(device)
        self.llm = self.llm.to(device)
        return self

def build_model(cfg, training=False):
    model = ChangeModel(cfg)
    if not training:
        return model

    # 创建损失函数
    losses = TotalLoss(weight_ce=cfg.loss.weight_ce, weight_dice=cfg.loss.weight_dice, weight_focal=cfg.loss.weight_focal, weight_bcl=cfg.loss.weight_bcl)

    return model, losses

# 测试
if __name__ == '__main__':
    import torch
    from utils import load_config
    cfg = load_config("../configs/config.yaml")
    model = ChangeModel(cfg)
    image1 = torch.randn(2, 3, 512, 512).to('cuda')
    image2 = torch.randn(2, 3, 512, 512).to('cuda')
    prompts = ["Detection of building changes", "This is another test prompt."]
    mask = model(image1, image2, prompts).to('cuda')
    print(mask.shape)
    print(mask)

    # 打印模型的参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")

    from thop import profile
    flops, params = profile(model, inputs=(image1, image2, prompts))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
