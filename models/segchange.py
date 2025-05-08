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
from models import MaskHead, DualInputVisualEncoder, FeatureDiffModule, TextEncoderBert, TextEncoderLLM, \
    FPNFeatureFuser
from models.losses import TotalLoss


class ChangeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg.device
        self.dual_encoder = DualInputVisualEncoder(cfg).to(self.device)
        self.feature_diff = FeatureDiffModule(in_channels_list=self.cfg.model.out_dims, diff_attention=self.cfg.model.diff_attention).to(self.device)
        self.fpn = FPNFeatureFuser(in_channels=self.cfg.model.out_dims, use_token_connector=self.cfg.model.use_token_connector).to(self.device)
        self.lang_dim = 2048 if cfg.model.text_encoder_name == 'microsoft/phi-1_5' else 768
        self.use_text_description = cfg.model.use_text_description
        self.mask_head = MaskHead(
            vis_dim=self.cfg.model.out_dims[-1] if not self.cfg.model.use_token_connector else self.cfg.model.out_dims[-1]//2,
            lang_dim=self.lang_dim,
            num_classes=self.cfg.model.num_classes,
            n_heads=8
        ).to(self.device)

        if self.use_text_description:
            if cfg.model.text_encoder_name == "microsoft/phi-1_5":
                self.llm = TextEncoderLLM(model_name=cfg.model.text_encoder_name, device=self.device,
                                          freeze_text_encoder=cfg.model.freeze_text_encoder)
            elif cfg.model.text_encoder_name == "bert-base-uncased":
                self.llm = TextEncoderBert(model_name=cfg.model.text_encoder_name, device=self.device,
                                           freeze_text_encoder=cfg.model.freeze_text_encoder)
            else:
                raise NotImplementedError(f"Unsupported text encoder name: {cfg.model.text_encoder_name}")

    def forward(self, image1, image2, prompts=None):
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        B, _, H, W = image1.shape  # 获取原图尺寸

        multi_scale_bev_feats1, multi_scale_bev_feats2 = self.dual_encoder(image1, image2)
        multi_scale_diff_feats = self.feature_diff(multi_scale_bev_feats1, multi_scale_bev_feats2)
        # 将原始图像尺寸传入 FPN Feature Fuser
        merged_feat = self.fpn(multi_scale_diff_feats)

        if self.use_text_description and prompts is not None:
            description_embeddings, _ = self.llm(prompts)
        else:
            # 如果不使用文本描述，可以使用占位符或默认值
            description_embeddings = torch.zeros((B, 4, self.lang_dim), device=self.device)

        mask = self.mask_head(merged_feat, description_embeddings)
        return mask

    def to(self, device):
        self.device = device
        self.dual_encoder = self.dual_encoder.to(device)
        self.feature_diff = self.feature_diff.to(device)
        self.fpn = self.fpn.to(device)
        self.mask_head = self.mask_head.to(device)
        if self.use_text_description:
            self.llm = self.llm.to(device)
        return self


def build_model(cfg, training=False):
    model = ChangeModel(cfg)
    if not training:
        return model

    # 创建损失函数
    losses = TotalLoss(
        num_classes=cfg.model.num_classes,
        weight_ce=cfg.loss.weight_ce,
        weight_dice=cfg.loss.weight_dice,
        weight_focal=cfg.loss.weight_focal,
        alpha=cfg.loss.alpha,
        gamma=cfg.loss.gamma,
        weight_bcl=cfg.loss.weight_bcl
    )

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
