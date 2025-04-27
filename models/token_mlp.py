# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: token_mlp.py
@Time    : 2025/4/17 下午4:31
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : Token 压缩连接器，将视觉 token 转换到多模态空间
@Usage   :
"""
import torch
import torch.nn as nn


class TokenConnector(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=1
                )
            )
            layers.append(
                nn.LayerNorm(out_channels)
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for i in range(0, len(self.layers), 2):
            conv = self.layers[i]
            ln = self.layers[i + 1]
            x = conv(x)
            x = ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


# 测试
if __name__ == '__main__':
    model = TokenConnector(256, 128, num_layers=2)
    x = torch.randn(2, 256, 224, 224)
    tokens = model(x)
    print(tokens.shape)  # 验证输出形状