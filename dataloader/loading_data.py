# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: loading_data.py
@Time    : 2025/4/18 下午5:10
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 加载数据
@Usage   :
"""
from torchvision import transforms
from dataloader.building import Building


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def loading_data(cfg):
    data_root = cfg.data.data_root
    data_format = cfg.data.data_format
    train_dataset = Building(data_root, train=True, data_format=data_format, **cfg.data.transforms.to_dict())
    val_dataset = Building(data_root, train=False, data_format=data_format)
    return train_dataset, val_dataset


# 测试
if __name__ == '__main__':
    from utils import load_config

    cfg = load_config("../configs/config.yaml")
    cfg.data.data_root = '../data/change'
    train_dataset, val_dataset = loading_data(cfg)

    print('训练集样本数：', len(train_dataset))
    print('测试集样本数：', len(val_dataset))

    img_a, img_b, prompt, label = train_dataset[0]
    print('训练集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '提示：', prompt, '标注形状：',
          label.shape)

    img_a, img_b, prompt, label = val_dataset[0]
    print('测试集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '提示：', prompt, '标注形状：',
          label.shape)
