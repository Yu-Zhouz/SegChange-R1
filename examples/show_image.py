# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: show_image.py
@Time    : 2025/5/21 上午10:43
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 测试数据和标签
@Usage   :
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dataloader import loading_data, DeNormalize


def tensor_to_image(img_tensor):
    """将 PyTorch Tensor 转换为 NumPy 图像（HWC 格式）"""
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img * 255).astype(np.uint8)  # [0,1] -> [0,255]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR for OpenCV


if __name__ == '__main__':
    from utils import load_config

    cfg = load_config("../configs/config.yaml")
    cfg.data_root = '../data/change'
    train_dataset, val_dataset = loading_data(cfg)

    print('训练集样本数：', len(train_dataset))
    print('测试集样本数：', len(val_dataset))

    img_a, img_b, prompt, label = train_dataset[0]
    # 转换为图像
    denorm_a = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    denorm_b = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    rgb_img = tensor_to_image(denorm_a(img_a.clone()))
    tir_img = tensor_to_image(denorm_b(img_b.clone()))

    # 显示图像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("RGB Image with Boxes")
    plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("TIR Image with Boxes")
    plt.imshow(cv2.cvtColor(tir_img, cv2.COLOR_BGR2RGB))  # 修改这里
    plt.axis('off')

    plt.savefig("output.png")  # 保存图像到文件
    print("图像已保存为 output.png")
