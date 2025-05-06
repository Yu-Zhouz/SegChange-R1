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
from models import DProjector, TokenConnector


# TODO:待完善，效果不好

class MaskGenerator(nn.Module):
    def __init__(self, in_channels, feature_strides=[4, 8, 16, 32], num_classes=1, lang_dim=2048, n_heads=8):
        super().__init__()
        self.num_classes = num_classes
        self.feature_strides = feature_strides
        self.in_channels = in_channels
        self.lang_dim = lang_dim
        self.n_heads = n_heads

        # 多尺度特征处理
        self.scale_heads = nn.ModuleList()
        for stride in feature_strides:
            self.scale_heads.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
                )
            )

        # 添加 TokenConnector
        self.token_connector = TokenConnector(in_channels=in_channels, out_channels=in_channels // 2)

        # D-Projector
        self.d_projector = DProjector(vis_dim=in_channels // 2, lang_dim=lang_dim, n_heads=n_heads)

        # Transformer 解码器
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=in_channels // 2, nhead=n_heads, dim_feedforward=2048),
            num_layers=2
        )

        # 最终掩码预测层
        self.mask_predictor = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

    def forward(self, multi_scale_feats, description_embeddings):
        """
        multi_scale_feats: list of [B, C, H_i, W_i]
        description_embeddings: [B, T, L]
        """
        # 处理多尺度特征
        processed_feats = []
        for i, feat in enumerate(multi_scale_feats):
            processed = self.scale_heads[i](feat)
            processed_feats.append(processed)

        # 合并多尺度特征
        merged_feat = torch.stack(processed_feats, dim=0).mean(dim=0)  # [B, C, H, W]

        # 使用 TokenConnector 转换视觉特征
        merged_feat = self.token_connector(merged_feat)

        # D-Projector
        query_vec = self.d_projector(description_embeddings, merged_feat)  # [B, C]

        # 将特征图展平为 [N, B, C]
        B, C, H, W = merged_feat.shape
        visual_features = merged_feat.flatten(2).permute(2, 0, 1)  # [N, B, C]

        # Transformer 解码器
        query_vec = query_vec.unsqueeze(0)  # [1, B, C]
        output = self.transformer_decoder(query_vec, visual_features).squeeze(0)  # [B, C]

        # 生成掩码
        mask = self.mask_predictor(merged_feat)  # [B, num_classes, H, W]

        # 使用查询向量进行偏置
        B, C_mask, H_mask, W_mask = mask.shape
        q = output.view(B, C, 1, 1).expand(-1, -1, H_mask, W_mask)
        logits = mask + (q * mask).sum(1, keepdim=True)

        return logits  # [B, 1, H, W]


# 测试
if __name__ == '__main__':
    model = MaskGenerator(in_channels=256, feature_strides=[4, 8, 16, 32], num_classes=1, lang_dim=2048, n_heads=8).to(
        'cuda')
    multi_scale_feats = [torch.randn(2, 256, 128, 128).to('cuda'),
                         torch.randn(2, 256, 64, 64).to('cuda'),
                         torch.randn(2, 256, 32, 32).to('cuda'),
                         torch.randn(2, 256, 16, 16).to('cuda')]
    description_embeddings = torch.randn(2, 4, 2048).to('cuda')  # 假设描述嵌入长度为50
    logits = model(multi_scale_feats, description_embeddings)
    print(logits.shape)  # 验证输出形状

    from thop import profile

    flops, params = profile(model, inputs=(multi_scale_feats, description_embeddings))
    print(f"MaskGenerator FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
