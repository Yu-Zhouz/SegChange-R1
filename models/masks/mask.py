# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: masks.py
@Time    : 2025/4/17 下午5:52
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    :  掩码生成器（不含掩码标记的掩码 2Former 标头）
@Usage   :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .d_projector import MultiheadAttention, DProjector


# from models.masks.d_projector import MultiheadAttention, DProjector

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False):
        """
        tgt: [seq_len, batch_size, d_model]
        memory: [src_len, batch_size, d_model]
        tgt_is_causal: whether to apply causal masking on self-attn
        memory_is_causal: not supported in this custom implementation
        """

        # Self Attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross Attention
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed Forward Network
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class MaskHead(nn.Module):
    def __init__(self, vis_dim, lang_dim, num_classes=1, n_heads=8):
        super().__init__()

        # D-Projector 将语言描述投影到视觉空间
        self.d_projector = DProjector(vis_dim=vis_dim, lang_dim=lang_dim, n_heads=n_heads)

        # Transformer Decoder
        self.num_queries = 5
        self.query_embed = nn.Embedding(self.num_queries, vis_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            TransformerDecoderLayer(d_model=vis_dim, nhead=n_heads, dim_feedforward=2048),
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
        # self.mask_head = nn.Sequential(
        #     nn.Conv2d(vis_dim * 2, vis_dim, kernel_size=3, padding=1),
        #     nn.GroupNorm(32, vis_dim),
        #     nn.ReLU(),
        #     nn.Conv2d(vis_dim, num_classes, kernel_size=1)
        # )
        self.mask_head = nn.Sequential(
            nn.Conv2d(vis_dim * 2, vis_dim // 2, kernel_size=1),  # 1x1卷积减少通道数
            nn.GroupNorm(32, vis_dim // 2),
            nn.ReLU(),
            nn.Conv2d(vis_dim // 2, vis_dim // 4, kernel_size=3, padding=1),  # 3x3卷积
            nn.GroupNorm(32, vis_dim // 4),
            nn.ReLU(),
            nn.Conv2d(vis_dim // 4, num_classes, kernel_size=1)
        )

    def forward(self, merged_feat, embs):
        """
        merged_feat: [B, C, H, W] 经过 FPN 融合后的特征图
        embs: [B, T, L] 描述嵌入
        original_size: tuple (H_orig, W_orig) 原图尺寸
        """
        B, C, H, W = merged_feat.shape

        # 使用 DProjector 获取 query_vec
        query_vec = self.d_projector(embs, merged_feat)  # [B, C]

        # 展平视觉特征用于 Transformer
        visual_features = merged_feat.flatten(2).permute(2, 0, 1)  # [N, B, C]

        # Transformer Decoder
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [Q, B, C]
        output = self.transformer_decoder(queries, visual_features)  # [Q, B, C]
        output = output.mean(0)  # [B, C]

        # 扩展 query 到空间维度并与 merged_feat 融合
        q_expanded = output.view(B, C, 1, 1).expand(B, C, H, W)

        # 使用 channel attention 强化 query_vec 的作用
        attn_weights = self.channel_attn(merged_feat)  # [B, C, 1, 1]
        attended_feat = merged_feat * (attn_weights + query_vec.view(B, C, 1, 1))

        # 拼接并生成掩码
        cat_feat = torch.cat([attended_feat, q_expanded], dim=1)  # [B, 2C, H, W]
        logits = self.mask_head(cat_feat)  # [B, K, H, W]

        return logits


# 测试
if __name__ == '__main__':
    model = MaskHead(vis_dim=256, lang_dim=768, num_classes=1, n_heads=8).to('cuda')
    merged_feat = torch.randn(2, 256, 512, 512).to('cuda')
    embs = torch.randn(2, 9, 768).to('cuda')  # 假设描述嵌入长度为9
    import time

    start = time.time()
    logits = model(merged_feat, embs)
    last = time.time()
    print(logits.shape)  # 验证输出形状

    from thop import profile

    # 计算每个模块的FLOPs
    with torch.no_grad():
        # DProjector
        flops, params = profile(model.d_projector, inputs=(embs, merged_feat), verbose=False)
        print(f"DProjector FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

        # Transformer Decoder
        visual_features = merged_feat.flatten(2).permute(2, 0, 1)
        queries = model.query_embed.weight.unsqueeze(1).repeat(1, 2, 1)
        flops, params = profile(model.transformer_decoder, inputs=(queries, visual_features), verbose=False)
        print(f"Transformer Decoder FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

        # Channel Attention
        flops, params = profile(model.channel_attn, inputs=(merged_feat,), verbose=False)
        print(f"Channel Attention FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

        # Mask Head
        query_vec = model.d_projector(embs, merged_feat)
        attended_feat = merged_feat * (model.channel_attn(merged_feat) + query_vec.view(2, 256, 1, 1))
        q_expanded = model.transformer_decoder(queries, visual_features).mean(0).view(2, 256, 1, 1).expand(-1, -1, 512,
                                                                                                           512)
        cat_feat = torch.cat([attended_feat, q_expanded], dim=1)
        flops, params = profile(model.mask_head, inputs=(cat_feat,), verbose=False)
        print(f"Mask Head FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

    # 总FLOPs
    total_flops, total_params = profile(model, inputs=(merged_feat, embs), verbose=False)
    print(f"Total FLOPs: {total_flops / 1e9:.2f} G, Params: {total_params / 1e6:.2f} M, time: {(last - start):.2f} s")
