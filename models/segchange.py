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
import torch
from models import MaskHead, DualInputVisualEncoder, FeatureDiffModule, TextEncoderBert, TextEncoderLLM, \
    FPNFeatureFuser, LightweightFPN
from models.losses import TotalLoss


class ChangeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg.device
        self.dual_encoder = DualInputVisualEncoder(cfg).to(self.device)
        self.feature_diff = FeatureDiffModule(in_channels_list=self.cfg.model.out_dims, diff_attention=self.cfg.model.diff_attention).to(self.device)
        if self.cfg.model.fpn_type == 'FPN':
            self.fpn = FPNFeatureFuser(in_channels=self.cfg.model.out_dims, use_token_connector=self.cfg.model.use_token_connector,
                                       use_ega=self.cfg.model.use_ega).to(self.device)
        elif self.cfg.model.fpn_type == 'L-FPN':
            self.fpn = LightweightFPN(in_channels=self.cfg.model.out_dims, use_token_connector=self.cfg.model.use_token_connector,
                                      use_ega=self.cfg.model.use_ega).to(self.device)
        else:
            raise NotImplementedError(f"Unsupported FPN type: {self.cfg.model.fpn_type}")
        self.lang_dim = 2048 if cfg.model.text_encoder_name == 'microsoft/phi-1_5' else 768
        self.use_text_description = cfg.model.use_text_description
        self.desc_embs = cfg.model.desc_embs  # 预训练文本嵌入路径
        self.mask_head = MaskHead(
            vis_dim=self.cfg.model.out_dims[-1] if not self.cfg.model.use_token_connector else self.cfg.model.out_dims[-1]//2,
            lang_dim=self.lang_dim,
            num_classes=self.cfg.model.num_classes,
            n_heads=8
        ).to(self.device)

        if self.use_text_description and self.desc_embs is None:
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

        description_embeddings = None
        if self.use_text_description:
            if self.desc_embs is not None:
                # 加载预计算的描述嵌入调整批次
                description_embeddings = torch.load(self.desc_embs).to(self.device).expand(B, -1,
                                                                                           -1)  # 形状: [B, num_descriptions, lang_dim]
            elif prompts is not None:
                # 实时通过 LLM 生成描述嵌入
                description_embeddings, _ = self.llm(prompts)
            else:
                # 如果不使用文本描述，可以使用占位符或默认值
                description_embeddings = torch.zeros((B, 4, self.lang_dim), device=self.device)
        else:
            # 如果不使用文本描述，可以使用占位符或默认值
            description_embeddings = torch.zeros((B, 4, self.lang_dim), device=self.device)

        # 确保 description_embeddings 不为 None
        if description_embeddings is None:
            description_embeddings = torch.zeros((B, 4, self.lang_dim), device=self.device)

        mask = self.mask_head(merged_feat, description_embeddings)
        return mask

    def to(self, device):
        self.device = device
        self.dual_encoder = self.dual_encoder.to(device)
        self.feature_diff = self.feature_diff.to(device)
        self.fpn = self.fpn.to(device)
        self.mask_head = self.mask_head.to(device)
        if self.use_text_description and self.desc_embs is None:
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
    device = cfg.device  # 从配置中获取设备，例如 'cuda' 或 'cpu'
    cfg.model.backbone_name = 'swin_base_patch4_window7_224'
    cfg.model.desc_embs = None
    model = ChangeModel(cfg).to(device)  # 将整个模型移动到指定设备

    # 输入张量也移动到相同设备
    image1 = torch.randn(2, 3, 512, 512).to(device)
    image2 = torch.randn(2, 3, 512, 512).to(device)
    prompts = ["Detection of building changes",  "Detection of building changes"]

    from thop import profile

    # 计算每个模块的FLOPs
    with torch.no_grad():  # 禁用梯度计算
        # Dual Encoder
        flops, params = profile(model.dual_encoder, inputs=(image1, image2), verbose=False)
        print(f"Dual Encoder - FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

        # Feature Diff
        multi_scale_bev_feats1, multi_scale_bev_feats2 = model.dual_encoder(image1, image2)
        flops, params = profile(model.feature_diff, inputs=(multi_scale_bev_feats1, multi_scale_bev_feats2),
                                verbose=False)
        print(f"Feature Diff - FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

        # FPN
        multi_scale_diff_feats = model.feature_diff(multi_scale_bev_feats1, multi_scale_bev_feats2)
        flops, params = profile(model.fpn, inputs=(multi_scale_diff_feats,), verbose=False)
        print(f"FPN - FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

        # Mask Head
        merged_feat = model.fpn(multi_scale_diff_feats)
        if model.use_text_description and model.desc_embs is not None:
            description_embeddings = torch.load(model.desc_embs).to(device).expand(2, -1, -1)
        else:
            description_embeddings = torch.zeros((2, 4, model.lang_dim), device=device)
        flops, params = profile(model.mask_head, inputs=(merged_feat, description_embeddings), verbose=False)
        print(f"Mask Head - FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

    mask = model(image1, image2, prompts).to(device)
    print(mask.shape)
    print(mask)

    # 总FLOPs
    total_flops, total_params = profile(model, inputs=(image1, image2, prompts), verbose=False)
    print(f"Total Model - FLOPs: {total_flops / 1e9:.2f} G, Params: {total_params / 1e6:.2f} M")
