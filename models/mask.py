# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: mask.py
@Time    : 2025/4/17 下午5:52
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    :  掩码生成器（不含掩码标记的掩码 2Former 标头）
@Usage   :
"""
import torch
import torch.nn as nn
from models import DProjector


class MaskHead(nn.Module):
    def __init__(self, vis_dim, lang_dim, num_classes=1, n_heads=8):
        super().__init__()

        # D-Projector 将语言描述投影到视觉空间
        self.d_projector = DProjector(vis_dim=vis_dim, lang_dim=lang_dim, n_heads=n_heads)

        # Transformer Decoder
        self.num_queries = 5
        self.query_embed = nn.Embedding(self.num_queries, vis_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=vis_dim, nhead=n_heads, dim_feedforward=2048),
            num_layers=2
        )

        # 用于融合 query_vec 的通道注意力模块
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(vis_dim, vis_dim // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(vis_dim // 8, vis_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 掩码预测头
        self.mask_head = nn.Sequential(
            nn.Conv2d(vis_dim * 2, vis_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, vis_dim),
            nn.ReLU(),
            nn.Conv2d(vis_dim, num_classes, kernel_size=1)
        )

    def forward(self, merged_feat, description_embeddings):
        """
        merged_feat: [B, C, H, W] 经过 FPN 融合后的特征图
        description_embeddings: [B, T, L] 描述嵌入
        original_size: tuple (H_orig, W_orig) 原图尺寸
        """
        B, C, H, W = merged_feat.shape

        # 使用 DProjector 获取 query_vec
        query_vec = self.d_projector(description_embeddings, merged_feat)  # [B, C]

        # 展平视觉特征用于 Transformer
        visual_features = merged_feat.flatten(2).permute(2, 0, 1)  # [N, B, C]

        # Transformer Decoder
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [Q, B, C]
        output = self.transformer_decoder(queries, visual_features)  # [Q, B, C]
        output = output.mean(0)  # [B, C]

        # 扩展 query 到空间维度并与 merged_feat 融合
        q_expanded = output.view(B, C, 1, 1).expand(-1, -1, H, W)

        # 使用 channel attention 强化 query_vec 的作用
        attn_weights = self.channel_attn(merged_feat)  # [B, C, 1, 1]
        attended_feat = merged_feat * (attn_weights + query_vec.view(B, C, 1, 1))

        # 拼接并生成掩码
        cat_feat = torch.cat([attended_feat, q_expanded], dim=1)  # [B, 2C, H, W]
        logits = self.mask_head(cat_feat)  # [B, K, H, W]

        return logits


# 测试
if __name__ == '__main__':
    model = MaskHead(vis_dim=256, lang_dim=2048, num_classes=1, n_heads=8).to('cuda')
    merged_feat = torch.randn(2, 256, 128, 128).to('cuda')
    description_embeddings = torch.randn(2, 4, 2048).to('cuda')  # 假设描述嵌入长度为50
    logits = model(merged_feat, description_embeddings)
    print(logits.shape)  # 验证输出形状

    from thop import profile
    flops, params = profile(model, inputs=(merged_feat, description_embeddings))
    print(f"MaskGenerator FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")