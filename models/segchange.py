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
from models import MaskHead, FeatureDiffModule, FPNFeatureFuser, LightweightFPN, DualInputVisualEncoder, TotalLoss, \
    build_embs


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
        self.mask_head = MaskHead(
            vis_dim=self.cfg.model.out_dims[-1] if not self.cfg.model.use_token_connector else self.cfg.model.out_dims[-1]//2,
            lang_dim=self.lang_dim,
            num_classes=self.cfg.model.num_classes,
            n_heads=8
        ).to(self.device)


    def forward(self, image1:  torch.Tensor, image2: torch.Tensor, embs: torch.Tensor):
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        B, _, H, W = image1.shape  # 获取原图尺寸

        multi_scale_bev_feats1, multi_scale_bev_feats2 = self.dual_encoder(image1, image2)
        multi_scale_diff_feats = self.feature_diff(multi_scale_bev_feats1, multi_scale_bev_feats2)
        # 将原始图像尺寸传入 FPN Feature Fuser
        merged_feat = self.fpn(multi_scale_diff_feats)

        # mask head
        mask = self.mask_head(merged_feat, embs)
        return mask

    def to(self, device):
        self.device = device
        self.dual_encoder = self.dual_encoder.to(device)
        self.feature_diff = self.feature_diff.to(device)
        self.fpn = self.fpn.to(device)
        self.mask_head = self.mask_head.to(device)
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
    cfg.model.backbone_name = 'hgnetv2'
    cfg.model.desc_embs = None
    model = ChangeModel(cfg).to(device)  # 将整个模型移动到指定设备

    # 输入张量也移动到相同设备
    image1 = torch.randn(2, 3, 512, 512).to(device)
    image2 = torch.randn(2, 3, 512, 512).to(device)
    embs = build_embs(prompts=[''], text_encoder_name=cfg.model.text_encoder_name,
                      freeze_text_encoder=cfg.model.freeze_text_encoder, device=device, batch_size=2)

    from thop import profile
    import time

    # 计算每个模块的FLOPs
    with torch.no_grad():  # 禁用梯度计算
        # Dual Encoder
        start_encoder = time.time()
        flops, params = profile(model.dual_encoder, inputs=(image1, image2), verbose=False)
        print(f"Dual Encoder - FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M, time: {time.time() - start_encoder:.2f}")

        # Feature Diff
        multi_scale_bev_feats1, multi_scale_bev_feats2 = model.dual_encoder(image1, image2)
        start_diff = time.time()
        flops, params = profile(model.feature_diff, inputs=(multi_scale_bev_feats1, multi_scale_bev_feats2),
                                verbose=False)
        print(f"Feature Diff - FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M, time: {time.time() - start_diff:.2f}")

        # FPN
        multi_scale_diff_feats = model.feature_diff(multi_scale_bev_feats1, multi_scale_bev_feats2)
        start_fpn = time.time()
        flops, params = profile(model.fpn, inputs=(multi_scale_diff_feats,), verbose=False)
        print(f"FPN - FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M, time: {time.time() - start_fpn:.2f}")

        # Mask Head
        merged_feat = model.fpn(multi_scale_diff_feats)
        start_mask = time.time()
        flops, params = profile(model.mask_head, inputs=(merged_feat, embs), verbose=False)
        print(f"Mask Head - FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M, time: {time.time() - start_mask:.2f}")

    start = time.time()
    mask = model(image1, image2, embs).to(device)
    last = time.time()
    print(mask.shape)

    # 总FLOPs
    total_flops, total_params = profile(model, inputs=(image1, image2, embs), verbose=False)
    print(f"Total FLOPs: {total_flops / 1e9:.2f} G, Params: {total_params / 1e6:.2f} M, time: {(last - start):.2f} s")

