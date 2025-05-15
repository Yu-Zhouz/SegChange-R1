# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: ega.py
@Time    : 2025/5/14 上午11:44
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : https://arxiv.org/pdf/2503.14012 边缘 - 高斯聚合（EGA）模块
@Usage   :
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Scharr(nn.Module):
    def __init__(self, channel):
        super(Scharr, self).__init__()
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        self.norm = nn.BatchNorm2d(channel)
        self.act = nn.ReLU()

    def forward(self, x):
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        scharr_edge = self.act(self.norm(scharr_edge))
        return scharr_edge


class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma):
        super(Gaussian, self).__init__()
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        gaussian = self.gaussian(x)
        gaussian = self.act(self.norm(gaussian))
        return gaussian

    def gaussian_kernel(self, size, sigma):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


class EGA(nn.Module):
    def __init__(self, channel, size=3, sigma=1.0):
        super(EGA, self).__init__()
        self.scharr = Scharr(channel)
        self.gaussian = Gaussian(channel, size, sigma)
        self.conv_extra = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        edges = self.scharr(x)
        gaussian = self.gaussian(edges)
        out = self.conv_extra(x + gaussian)
        return out
