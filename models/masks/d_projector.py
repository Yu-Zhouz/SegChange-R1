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
import math
from torch import nn
import torch
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.batch_first = False
    def forward(self, query, key, value):
        """
        query: [seq_len_q, batch_size, embed_dim]
        key:   [seq_len_k, batch_size, embed_dim]
        value: [seq_len_v, batch_size, embed_dim]
        """
        seq_len_q, batch_size, _ = query.size()
        seq_len_k, _, _ = key.size()
        seq_len_v, _, _ = value.size()

        # 线性变换
        q = self.in_proj_q(query)
        k = self.in_proj_k(key)
        v = self.in_proj_v(value)

        # 分头
        q = q.view(seq_len_q, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len_k, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len_v, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # 注意力得分
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 加权值
        attn_output = torch.bmm(attn_weights, v)

        # 合并头
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len_q, batch_size, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights



class DProjector(nn.Module):
    def __init__(self, vis_dim, lang_dim, n_heads=8):
        super().__init__()
        # project language dim to vis dim if needed
        self.lang2vis = nn.Linear(lang_dim, vis_dim) if lang_dim != vis_dim else nn.Identity()
        self.cross_attn = MultiheadAttention(vis_dim, n_heads)
        self.lin = nn.Linear(vis_dim, vis_dim)

    def forward(self, desc_embs: torch.Tensor, vis_feat: torch.Tensor):
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

    from thop import profile
    flops, params = profile(model, inputs=(desc_embs, vis_feat))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")