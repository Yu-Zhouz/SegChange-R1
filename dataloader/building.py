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


class Building(Dataset):
    def __init__(self, data_root, a_transform=None, b_transform=None, train=False, test=False, data_format="default",
                 **kwargs):
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
        self.a_transform = a_transform
        self.b_transform = b_transform

        # 数据增强参数
        self.color_jitter = kwargs.get('color_jitter', 0.0)
        self.rotation_degree = kwargs.get('rotation_degree', 0)
        self.gamma_range = kwargs.get('gamma_range', (1.0, 1.0))
        self.affine_degree = kwargs.get('affine_degree', 0)
        self.erase_prob = kwargs.get('erase_prob', 0.0)
        self.erase_ratio = kwargs.get('erase_ratio', (0.02, 0.33))
        self.blur_sigma = kwargs.get('blur_sigma', 0.0)


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
            a_img, b_img, label = self.apply_transforms(a_img, b_img, label)

        # Step 4: ToTensor 和 Normalize（转换为 Tensor）
        if self.a_transform is not None:
            a_img = self.a_transform(a_img)
        if self.b_transform is not None:
            b_img = self.b_transform(b_img)

        a_img = a_img.float()
        b_img = b_img.float()
        label = torch.tensor(label, dtype=torch.int64).unsqueeze(0)

        return a_img, b_img, prompt, label

    def apply_transforms(self, a_img, b_img, label):
        # 应用颜色扰动
        if self.color_jitter > 0.0:
            a_img = self.color_jitter_transform(a_img)
            b_img = self.color_jitter_transform(b_img)

        # 应用旋转
        if self.rotation_degree > 0:
            angle = random.uniform(-self.rotation_degree, self.rotation_degree)
            a_img = self.rotate_image(a_img, angle)
            b_img = self.rotate_image(b_img, angle)
            label = self.rotate_image(label, angle, interpolation=cv2.INTER_NEAREST)

        # 应用 Gamma 校正（光照变化）
        if self.gamma_range[0] != self.gamma_range[1]:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            a_img = ((a_img / 255.0) ** gamma) * 255.0
            b_img = ((b_img / 255.0) ** gamma) * 255.0

        # 应用仿射变换
        if self.affine_degree > 0:
            a_img, b_img, label = self.affine_transform(a_img, b_img, label)

        # 应用随机擦除
        if self.erase_prob > 0.0:
            a_img = self.random_erase(a_img, self.erase_prob, self.erase_ratio)
            b_img = self.random_erase(b_img, self.erase_prob, self.erase_ratio)

        # 应用高斯模糊
        if self.blur_sigma > 0.0:
            a_img = self.gaussian_blur(a_img, self.blur_sigma)
            b_img = self.gaussian_blur(b_img, self.blur_sigma)

        return a_img, b_img, label

    def color_jitter_transform(self, img):
        """颜色扰动"""
        brightness = random.uniform(1 - self.color_jitter, 1 + self.color_jitter)
        contrast = random.uniform(1 - self.color_jitter, 1 + self.color_jitter)
        saturation = random.uniform(1 - self.color_jitter, 1 + self.color_jitter)
        hue = random.uniform(-self.color_jitter, self.color_jitter)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img[:, :, 1] = np.clip(img[:, :, 1] * saturation, 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] * brightness, 0, 255)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        img = np.clip(img * contrast, 0, 255)
        img = np.array(img, dtype=np.uint8)

        return img

    def rotate_image(self, img, angle, interpolation=cv2.INTER_LINEAR):
        """旋转图像"""
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=interpolation)
        return rotated

    def affine_transform(self, a_img, b_img, label):
        """仿射变换"""
        h, w = a_img.shape[:2]
        src_points = np.float32([[0, 0], [w, 0], [0, h]])
        dst_points = np.float32([
            [random.uniform(0, w * 0.1), random.uniform(0, h * 0.1)],
            [random.uniform(w * 0.9, w), random.uniform(0, h * 0.1)],
            [random.uniform(0, w * 0.1), random.uniform(h * 0.9, h)]
        ])
        M = cv2.getAffineTransform(src_points, dst_points)
        a_img = cv2.warpAffine(a_img, M, (w, h), flags=cv2.INTER_LINEAR)
        b_img = cv2.warpAffine(b_img, M, (w, h), flags=cv2.INTER_LINEAR)
        label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST)
        return a_img, b_img, label

    def random_erase(self, img, prob, ratio):
        """随机擦除"""
        if random.random() > prob:
            return img

        h, w = img.shape[:2]

        # 检查图像是否足够大
        if h < 4 or w < 4:  # 如果图像太小，跳过擦除
            return img

        aspect_ratio = random.uniform(ratio[0], ratio[1])
        area = h * w
        erase_area = random.uniform(0.02, 0.4) * area

        erase_h = int(np.sqrt(erase_area * aspect_ratio))
        erase_w = int(np.sqrt(erase_area / aspect_ratio))

        # 防止裁剪超出图像范围
        if erase_h >= h or erase_w >= w:
            return img

        x1 = random.randint(0, w - erase_w)
        y1 = random.randint(0, h - erase_h)
        x2 = x1 + erase_w
        y2 = y1 + erase_h

        img[y1:y2, x1:x2] = np.random.randint(0, 256, size=(erase_h, erase_w, 3), dtype=np.uint8)
        return img

    def gaussian_blur(self, img, sigma):
        """高斯模糊"""
        kernel_size = int(2 * 2 * sigma) + 1  # 根据 sigma 计算核大小
        if kernel_size % 2 == 0:
            kernel_size += 1
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        return img

def sort_filenames_numerically(filenames):
    def numeric_key(filename):
        numbers = list(map(int, re.findall(r'\d+', filename)))
        return (tuple(numbers), filename) if numbers else ((), filename)

    return sorted(filenames, key=numeric_key)


def random_crop(img_a, img_b, label):
    target_h, target_w = 512, 512  # 设置目标尺寸
    img_a_h, img_a_w = img_a.shape[:2]

    # 检查图像尺寸是否满足裁剪要求
    if img_a_h < target_h or img_a_w < target_w:
        # 如果图像太小，进行填充
        new_img_a = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        new_img_b = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        new_label = np.zeros((target_h, target_w), dtype=np.uint8)
        new_img_a[:img_a_h, :img_a_w] = img_a
        new_img_b[:img_a_h, :img_a_w] = img_b
        new_label[:img_a_h, :img_a_w] = label
        img_a, img_b, label = new_img_a, new_img_b, new_label
    else:
        start_h = random.randint(0, img_a_h - target_h)
        start_w = random.randint(0, img_a_w - target_w)
        img_a = img_a[start_h:start_h + target_h, start_w:start_w + target_w]
        img_b = img_b[start_h:start_h + target_h, start_w:start_w + target_w]
        label = label[start_h:start_h + target_h, start_w:start_w + target_w]

    return img_a, img_b, label


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