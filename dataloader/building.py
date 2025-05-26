# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: building.py
@Time    : 2025/4/18 下午5:10
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 建筑物变化检测数据
@Usage   :
"""
import logging
import os
import random
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataloader import build_transforms


class Building(Dataset):
    def __init__(self, data_root, train=False, test=False, data_format="default", **kwargs):
        self.data_path = data_root
        self.train = train
        self.test = test
        self.data_format = data_format  # 控制数据集格式

        # 根据数据集格式构建数据目录
        if self.data_format == "default":
            if self.test:
                self.data_dir = self.data_path  # 测试数据路径由外部指定
            else:
                self.data_dir = os.path.join(self.data_path, 'train' if self.train else 'val')
            self.a_dir = os.path.join(self.data_dir, 'A')
            self.b_dir = os.path.join(self.data_dir, 'B')
            self.labels_dir = os.path.join(self.data_dir, 'label')
            self.prompts_path = os.path.join(self.data_dir, 'prompts.txt')
        elif self.data_format == "custom":
            self.data_dir = self.data_path
            self.a_dir = os.path.join(self.data_path, 'A')
            self.b_dir = os.path.join(self.data_path, 'B')
            self.labels_dir = os.path.join(self.data_path, 'label')
            self.prompts_path = os.path.join(self.data_path, 'prompts.txt')
        else:
            raise ValueError(f"不支持的数据集格式：{self.data_format}")

        self.img_map = {}
        self.img_list = []
        self.prompts = {}  # 用于存储每个图像的提示信息

        # 根据数据集格式加载图像路径
        if self.data_format == "default":
            a_img_paths = [filename for filename in os.listdir(self.a_dir) if filename.endswith('.png')]
            for filename in a_img_paths:
                a_img_path = os.path.join(self.a_dir, filename)
                b_img_path = os.path.join(self.b_dir, filename)
                label_path = os.path.join(self.labels_dir, filename)
                if os.path.isfile(a_img_path) and os.path.isfile(b_img_path) and os.path.isfile(label_path):
                    self.img_map[a_img_path] = (b_img_path, label_path)
                    self.img_list.append(a_img_path)
        elif self.data_format == "custom":
            # 从对应的txt文件中读取图像路径
            if self.test:
                list_file = os.path.join(self.data_path, 'list', 'test.txt')
            elif self.train:
                list_file = os.path.join(self.data_path, 'list', 'train.txt')
            else:
                list_file = os.path.join(self.data_path, 'list', 'val.txt')

            if not os.path.exists(list_file):
                raise FileNotFoundError(f"未找到列表文件：{list_file}")

            with open(list_file, 'r') as f:
                for line in f:
                    filename = line.strip()
                    if filename:
                        a_img_path = os.path.join(self.a_dir, filename)
                        b_img_path = os.path.join(self.b_dir, filename)
                        label_path = os.path.join(self.labels_dir, filename)
                        if os.path.isfile(a_img_path) and os.path.isfile(b_img_path) and os.path.isfile(label_path):
                            self.img_map[a_img_path] = (b_img_path, label_path)
                            self.img_list.append(a_img_path)

        # 读取 prompts.txt 文件
        self._load_prompts()

        self.img_list = sort_filenames_numerically(self.img_list)

        self.nSamples = len(self.img_list)
        self.transform = build_transforms(**kwargs)

        self.a_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.b_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def _load_prompts(self):
        """读取 prompts.txt 文件并存储提示信息"""
        if not os.path.exists(self.prompts_path):
            logging.warning(f"prompts.txt 文件不存在：{self.prompts_path}")
            return

        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('  ', 1)  # 使用两个空格分割
                if len(parts) == 2:
                    filename, prompt = parts
                    self.prompts[filename] = prompt

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        a_img_path = self.img_list[index]
        b_img_path, label_path = self.img_map[a_img_path]
        filename = os.path.basename(a_img_path)

        # Step 1: 使用 OpenCV 读取图像，并转为 RGB 格式
        a_img = cv2.imread(a_img_path)
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        b_img = cv2.imread(b_img_path)
        b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = (label > 0).astype(np.uint8)

        # Step 2: 读取 prompt
        prompt = self.prompts.get(filename, "")

        # Step 3: 数据增强（在 NumPy 阶段进行）
        if self.train:
            a_img, b_img, label = self.transform(a_img, b_img, label)

        # Step 4: ToTensor 和 Normalize（转换为 Tensor）
        if self.a_transform is not None:
            a_img = self.a_transform(a_img)
        if self.b_transform is not None:
            b_img = self.b_transform(b_img)

        a_img = a_img.float()
        b_img = b_img.float()
        label = torch.tensor(label, dtype=torch.int64).unsqueeze(0)

        return a_img, b_img, prompt, label


def sort_filenames_numerically(filenames):
    def numeric_key(filename):
        numbers = list(map(int, re.findall(r'\d+', filename)))
        return (tuple(numbers), filename) if numbers else ((), filename)

    return sorted(filenames, key=numeric_key)



if __name__ == '__main__':
    from utils import load_config

    cfg = load_config("../configs/config.yaml")
    cfg.data_root = '../data/change'
    cfg.test.test_img_dirs = '../data/change/val'
    a_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    b_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Building(cfg.data_root,
                             a_transform=a_transform,
                             b_transform=b_transform,
                             train=True,
                             **cfg.data.to_dict())
    val_dataset = Building(cfg.data_root, a_transform=a_transform, b_transform=b_transform, train=False)

    test_dataset = Building(
        data_root=cfg.test.test_img_dirs,
        a_transform=a_transform,
        b_transform=b_transform,
        test=True
    )

    print('训练集样本数：', len(train_dataset))
    print('测试集样本数：', len(val_dataset))
    print('测试集样本数：', len(test_dataset))

    img_a, img_b, prompt, label = train_dataset[0]
    print('训练集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '提示：', prompt, '标注形状：', label.shape)

    img_a, img_b, prompt, label = val_dataset[0]
    print('验证集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '提示：', prompt, '标注形状：', label.shape)

    img_a, img_b, prompt, label = test_dataset[0]
    print('测试集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '提示：', prompt, '标注形状：', label.shape)