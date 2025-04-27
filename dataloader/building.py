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
import os
import random
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Building(Dataset):
    def __init__(self, data_root, a_transform=None, b_transform=None, train=False, resizing=False,
                 patch=False, flip=False):
        self.data_path = data_root
        self.train = train
        self.data_dir = os.path.join(self.data_path, 'train' if self.train else 'val')
        self.a_dir = os.path.join(self.data_dir, 'A')
        self.b_dir = os.path.join(self.data_dir, 'B')
        self.labels_dir = os.path.join(self.data_dir, 'label')
        self.prompts_path = os.path.join(self.data_dir, 'prompts.txt')

        self.img_map = {}
        self.img_list = []
        self.prompts = {}  # 用于存储每个图像的提示信息

        a_img_paths = [filename for filename in os.listdir(self.a_dir) if filename.endswith('.png')]
        for filename in a_img_paths:
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
        self.a_transform = a_transform
        self.b_transform = b_transform
        self.resizing = resizing
        self.patch = patch
        self.flip = flip

    def _load_prompts(self):
        """读取 prompts.txt 文件并存储提示信息"""
        if not os.path.exists(self.prompts_path):
            print(f"prompts.txt 文件不存在：{self.prompts_path}")
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

        # 获取图像名用于查找提示
        filename = os.path.basename(a_img_path)

        a_img = cv2.imread(a_img_path)
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        b_img = cv2.imread(b_img_path)
        b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # 将标注图像转换为二值化图像（0 和 1）
        label = (label > 0).astype(np.uint8)

        # 获取对应的提示信息
        prompt = self.prompts.get(filename, "")

        if self.a_transform is not None:
            a_img = self.a_transform(a_img)
        if self.b_transform is not None:
            b_img = self.b_transform(b_img)

        if self.resizing and self.train:
            scale_range = [0.7, 1.3]
            min_size = min(a_img.shape[1:3])
            scale = random.uniform(*scale_range)
            if scale * min_size > 128:
                a_img = torch.nn.functional.interpolate(a_img.unsqueeze(0), scale_factor=scale,
                                                        mode='bilinear').squeeze(0)
                b_img = torch.nn.functional.interpolate(b_img.unsqueeze(0), scale_factor=scale,
                                                        mode='bilinear').squeeze(0)
                label = torch.nn.functional.interpolate(
                    torch.tensor(label, dtype=torch.float32).unsqueeze(0).unsqueeze(0), scale_factor=scale,
                    mode='nearest').squeeze(0)

        if self.patch and self.train:
            a_img, b_img, label = random_crop(a_img, b_img, label)

        if random.random() > 0.5 and self.flip:
            a_img = torch.flip(a_img, [2])
            b_img = torch.flip(b_img, [2])
            label = torch.flip(label.clone().detach(), [1])

        #归一化
        a_img = torch.Tensor(a_img) / 255.0
        b_img = torch.Tensor(b_img) / 255.0
        label = torch.tensor(label, dtype=torch.int64).unsqueeze(0)

        # 返回图像、标注和提示信息
        return a_img, b_img, prompt, label


def sort_filenames_numerically(filenames):
    def numeric_key(filename):
        numbers = list(map(int, re.findall(r'\d+', filename)))
        return (tuple(numbers), filename) if numbers else ((), filename)

    return sorted(filenames, key=numeric_key)


def random_crop(img_a, img_b, label):
    half_h, half_w = 128, 128
    img_a_h, img_a_w = img_a.shape[0], img_a.shape[1]

    # 检查图像尺寸是否满足裁剪要求
    if img_a_h < half_h or img_a_w < half_w:
        return img_a, img_b, label

    start_h = random.randint(0, img_a_h - half_h)
    start_w = random.randint(0, img_a_w - half_w)
    img_a = img_a[start_h:start_h + half_h, start_w:start_w + half_w]
    img_b = img_b[start_h:start_h + half_h, start_w:start_w + half_w]
    label = label[start_h:start_h + half_h, start_w:start_w + half_w]
    return img_a, img_b, label


if __name__ == '__main__':
    data_path = '../data/change'

    a_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    b_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Building(data_path, a_transform=a_transform, b_transform=b_transform, train=True,
                             resizing=False, patch=False, flip=False)
    val_dataset = Building(data_path, a_transform=a_transform, b_transform=b_transform, train=False)

    print('训练集样本数：', len(train_dataset))
    print('测试集样本数：', len(val_dataset))

    img_a, img_b, prompt, label = train_dataset[0]
    print('训练集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '提示：', prompt, '标注形状：', label.shape)

    img_a, img_b, prompt, label = val_dataset[0]
    print('测试集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '提示：', prompt, '标注形状：', label.shape)
