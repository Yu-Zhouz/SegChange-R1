# -*- coding: utf-8 -*-
"""
@Project : SegChange-R2
@FileName: transforms.py
@Time    : 2025/5/24 下午4:24
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 建筑物变化检测数据增强变换类
@Usage   :
"""
import cv2
import numpy as np
import random


class TransformBase:
    """变换基类"""

    def apply(self, a_img, b_img, label):
        raise NotImplementedError("Subclass must implement abstract method")


class Compose:
    """组合多个变换"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, a_img, b_img, label):
        for t in self.transforms:
            a_img, b_img, label = t.apply(a_img, b_img, label)
        return a_img, b_img, label


class ColorJitterTransform(TransformBase):
    """颜色扰动变换"""

    def __init__(self, color_jitter, apply_prob=0.5):
        self.color_jitter = color_jitter
        self.apply_prob = apply_prob

    def apply(self, a_img, b_img, label):
        if random.random() > self.apply_prob:
            return a_img, b_img, label  # 不进行变换，直接返回原图

        brightness = random.uniform(1 - self.color_jitter, 1 + self.color_jitter)
        contrast = random.uniform(1 - self.color_jitter, 1 + self.color_jitter)
        saturation = random.uniform(1 - self.color_jitter, 1 + self.color_jitter)
        hue = random.uniform(-self.color_jitter, self.color_jitter)

        # 对 a 图像进行变换
        a_img = cv2.cvtColor(a_img, cv2.COLOR_RGB2HSV)
        a_img[:, :, 1] = np.clip(a_img[:, :, 1] * saturation, 0, 255)
        a_img[:, :, 2] = np.clip(a_img[:, :, 2] * brightness, 0, 255)
        a_img = cv2.cvtColor(a_img, cv2.COLOR_HSV2RGB)
        a_img = np.clip(a_img * contrast, 0, 255).astype(np.uint8)

        # 对 b 图像进行相同变换
        b_img = cv2.cvtColor(b_img, cv2.COLOR_RGB2HSV)
        b_img[:, :, 1] = np.clip(b_img[:, :, 1] * saturation, 0, 255)
        b_img[:, :, 2] = np.clip(b_img[:, :, 2] * brightness, 0, 255)
        b_img = cv2.cvtColor(b_img, cv2.COLOR_HSV2RGB)
        b_img = np.clip(b_img * contrast, 0, 255).astype(np.uint8)

        return a_img, b_img, label


class RotateTransform(TransformBase):
    """旋转变换"""

    def __init__(self, rotation_degree, apply_prob=0.5):
        self.rotation_degree = rotation_degree
        self.apply_prob = apply_prob

    def apply(self, a_img, b_img, label):
        if random.random() > self.apply_prob:
            return a_img, b_img, label  # 不进行变换，直接返回原图

        angle = random.uniform(-self.rotation_degree, self.rotation_degree)
        h, w = a_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        a_img = cv2.warpAffine(a_img, M, (w, h), flags=cv2.INTER_LINEAR)
        b_img = cv2.warpAffine(b_img, M, (w, h), flags=cv2.INTER_LINEAR)
        label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST)
        return a_img, b_img, label


class GammaCorrectionTransform(TransformBase):
    """Gamma 校正变换"""

    def __init__(self, gamma_range):
        self.gamma_range = gamma_range

    def apply(self, a_img, b_img, label):
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        a_img = ((a_img / 255.0) ** gamma) * 255.0
        b_img = ((b_img / 255.0) ** gamma) * 255.0
        a_img = np.array(a_img, dtype=np.uint8)
        b_img = np.array(b_img, dtype=np.uint8)
        return a_img, b_img, label


class AffineTransform(TransformBase):
    """仿射变换"""

    def __init__(self, affine_degree, apply_prob=0.2):
        self.affine_degree = affine_degree
        self.apply_prob = apply_prob

    def apply(self, a_img, b_img, label):
        if random.random() > self.apply_prob:
            return a_img, b_img, label  # 不进行变换，直接返回原图

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


class RandomEraseTransform(TransformBase):
    """随机擦除变换"""

    def __init__(self, erase_prob, erase_ratio):
        self.erase_prob = erase_prob
        self.erase_ratio = erase_ratio

    def apply(self, a_img, b_img, label):
        if random.random() > self.erase_prob:
            return a_img, b_img, label

        h, w = a_img.shape[:2]

        if h < 4 or w < 4:
            return a_img, b_img, label

        aspect_ratio = random.uniform(self.erase_ratio[0], self.erase_ratio[1])
        area = h * w
        erase_area = random.uniform(0.02, 0.4) * area

        erase_h = int(np.sqrt(erase_area * aspect_ratio))
        erase_w = int(np.sqrt(erase_area / aspect_ratio))

        if erase_h >= h or erase_w >= w:
            return a_img, b_img, label

        x1 = random.randint(0, w - erase_w)
        y1 = random.randint(0, h - erase_h)
        x2 = x1 + erase_w
        y2 = y1 + erase_h

        a_img[y1:y2, x1:x2] = np.random.randint(0, 256, size=(erase_h, erase_w, 3), dtype=np.uint8)
        b_img[y1:y2, x1:x2] = np.random.randint(0, 256, size=(erase_h, erase_w, 3), dtype=np.uint8)

        return a_img, b_img, label


class GaussianBlurTransform(TransformBase):
    """高斯模糊变换"""

    def __init__(self, blur_sigma):
        self.blur_sigma = blur_sigma

    def apply(self, a_img, b_img, label):
        kernel_size = int(2 * 2 * self.blur_sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        a_img = cv2.GaussianBlur(a_img, (kernel_size, kernel_size), self.blur_sigma)
        b_img = cv2.GaussianBlur(b_img, (kernel_size, kernel_size), self.blur_sigma)
        return a_img, b_img, label


class CLAHETransform(TransformBase):
    """直方图均衡化变换"""

    def __init__(self, apply_prob=0.5):
        self.apply_prob = apply_prob

    def apply(self, a_img, b_img, label):
        if random.random() > self.apply_prob:
            return a_img, b_img, label

        lab_a = cv2.cvtColor(a_img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_a[:, :, 0] = clahe.apply(lab_a[:, :, 0])
        a_img = cv2.cvtColor(lab_a, cv2.COLOR_LAB2RGB)

        lab_b = cv2.cvtColor(b_img, cv2.COLOR_RGB2LAB)
        lab_b[:, :, 0] = clahe.apply(lab_b[:, :, 0])
        b_img = cv2.cvtColor(lab_b, cv2.COLOR_LAB2RGB)

        return a_img, b_img, label


def build_transforms(**kwargs):
    transforms = []

    # 颜色扰动
    if kwargs.get('color_jitter', 0.0) > 0.0:
        transforms.append(ColorJitterTransform(kwargs['color_jitter'], kwargs['apply_prob']))

    # 旋转
    if kwargs.get('rotation_degree', 0) > 0:
        transforms.append(RotateTransform(kwargs['rotation_degree'], kwargs['apply_prob']))

    # Gamma 校正
    gamma_range = kwargs.get('gamma_range', (1.0, 1.0))
    if gamma_range[0] != 1.0 or gamma_range[1] != 1.0:
        transforms.append(GammaCorrectionTransform(gamma_range))

    # 仿射变换
    if kwargs.get('affine_degree', 0) > 0:
        transforms.append(AffineTransform(kwargs['affine_degree']))

    # 随机擦除
    erase_prob = kwargs.get('erase_prob', 0.0)
    erase_ratio = kwargs.get('erase_ratio', (0.02, 0.33))
    if erase_prob > 0.0:
        transforms.append(RandomEraseTransform(erase_prob, erase_ratio))

    # 高斯模糊
    if kwargs.get('blur_sigma', 0.0) > 0.0:
        transforms.append(GaussianBlurTransform(kwargs['blur_sigma']))

    # 直方图均衡化
    if kwargs.get('clahe', 0.0) > 0.0:
        transforms.append(CLAHETransform(kwargs['clahe']))

    return Compose(transforms) if transforms else None
