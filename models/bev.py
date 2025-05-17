# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: bev.py
@Time    : 2025/4/18 上午9:49
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : BEV空间转换用于将不同视角的图像特征转换到统一的鸟瞰图（BEV）空间，解决建筑物变化检测过程中的错位问题,从而更好地捕捉建筑物的空间布局和变化信息，提升变化检测的准确性。
@Usage   :
"""
import torch
import torch.nn as nn
from einops import rearrange


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim_head * 3, bias=False)
        self.to_out = nn.Linear(dim_head, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q = q * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        out = torch.einsum('b h n d, b h d e -> b h n e', q, context)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class BEVTransformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=in_channels,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            batch_first=True
        )
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten and transpose the input
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, N, C]
        # Transformer
        x_transformed = self.transformer(x_flat, x_flat)
        # Reshape to BEV
        x_bev = x_transformed.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        # Projection
        x_bev = self.proj(x_bev)
        return x_bev


class BEVLinearAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_attention = LinearAttention(in_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten and transpose the input
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, N, C]
        # Apply linear attention
        x_linear_attn = self.linear_attention(x_flat)
        # Reshape to BEV
        x_bev = x_linear_attn.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        # Projection
        x_bev = self.proj(x_bev)
        return x_bev


# 测试
if __name__ == '__main__':
    x = torch.randn(2, 256, 128, 128).to('cuda')
    # model = BEVTransformer(in_channels=256, out_channels=256).to('cuda')
    model = BEVLinearAttention(in_channels=256, out_channels=256).to('cuda')
    import time
    start = time.time()
    bev_output = model(x)
    last = time.time()
    print(f"BEV output shape: {bev_output.shape}")

    from thop import profile

    flops, params = profile(model, inputs=(x,))
    print(f"encoder FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M, time: {(last - start):.2f} s")

