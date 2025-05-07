# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: d_projector.py
@Time    : 2025/4/17 下午5:30
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 将描述嵌入投影到查询向量
@Usage   :
"""
from torch import nn


class DProjector(nn.Module):
    def __init__(self, vis_dim, lang_dim, n_heads=8):
        super().__init__()
        # project language dim to vis dim if needed
        self.lang2vis = nn.Linear(lang_dim, vis_dim) if lang_dim != vis_dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(vis_dim, n_heads)
        self.lin = nn.Linear(vis_dim, vis_dim)

    def forward(self, desc_embs, vis_feat):
        """
        desc_embs: [B, T, L]  text hidden-states
        vis_feat:  [B, C, H, W]
        """
        B, C, H, W = vis_feat.shape
        # 1) average desc_embs over T
        desc_avg = desc_embs.mean(dim=1)  # [B, L]
        desc_vis = self.lang2vis(desc_avg)  # [B, C]

        # 2) cross-attend to every pixel
        v = vis_feat.flatten(2).permute(2, 0, 1)  # [N, B, C]
        q = desc_vis.unsqueeze(0)  # [1, B, C]
        attn_out, _ = self.cross_attn(q, v, v)  # [1, B, C]
        # residual + linear
        q2 = self.lin(attn_out + q).squeeze(0)  # [B, C]
        return q2  # this is the single query vector per image


# 测试
if __name__ == '__main__':
    import torch

    model = DProjector(vis_dim=256, lang_dim=2048)
    desc_embs = torch.randn(2, 4, 2048)
    vis_feat = torch.randn(2, 256, 512, 512)
    q2 = model(desc_embs, vis_feat)
    print(q2.shape)
