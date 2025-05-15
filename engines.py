# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: engines.py
@Time    : 2025/4/21 上午9:48
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score


def train(cfg, model, criterion, dataloader, optimizer, device, epoch):
    model.train()
    criterion.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_pixels = 0

    with tqdm(dataloader, desc=f'Epoch {epoch} [Training]') as pbar:
        for images_a, images_b, prompt, labels in pbar:
            images_a = images_a.to(device)
            images_b = images_b.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images_a, images_b, prompt)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images_a.size(0)
            total_samples += images_a.size(0)

            # Calculate accuracy
            if cfg.model.num_classes == 1:
                preds = (torch.sigmoid(outputs) > cfg.training.threshold).float().squeeze(1)
            else:
                preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_pixels += labels.numel()

            pbar.set_postfix({
                'loss': total_loss / total_samples,
                'oa': total_correct / total_pixels
            })

    epoch_loss = total_loss / total_samples
    epoch_oa = total_correct / total_pixels

    return {'loss': epoch_loss, 'oa': epoch_oa}


def evaluate(cfg, model, criterion, dataloader, device, epoch):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(), tqdm(dataloader, desc=f'Epoch {epoch} [Validation]') as pbar:
        for images_a, images_b, prompt, labels in pbar:
            images_a = images_a.to(device)
            images_b = images_b.to(device)
            labels = labels.to(device)

            outputs = model(images_a, images_b, prompt)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images_a.size(0)
            total_samples += images_a.size(0)

            # Store predictions and labels
            if cfg.model.num_classes == 1:
                preds = (torch.sigmoid(outputs) > cfg.training.threshold).float().squeeze(1).cpu().numpy()
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

            pbar.set_postfix({
                'loss': total_loss / total_samples
            })

    # Calculate evaluation metrics
    val_loss = total_loss / total_samples

    if cfg.model.num_classes == 1:
        # 二分类指标
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        iou = jaccard_score(all_labels, all_preds)
        oa = accuracy_score(all_labels, all_preds)
    else:
        # 多分类指标（使用macro平均）
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        iou = jaccard_score(all_labels, all_preds, average='macro')
        oa = accuracy_score(all_labels, all_preds)

    metrics = {
        'loss': val_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'oa': oa
    }

    return metrics


def evaluate_model(cfg, model, dataloader, device, output_dir):
    model.eval()
    all_preds = []
    all_labels = []

    # 创建保存拼接图像的目录
    comparison_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)

    with torch.no_grad(), tqdm(dataloader, desc='[Testing]') as pbar:
        for images_a, images_b, prompt, labels in pbar:
            images_a = images_a.to(device)
            images_b = images_b.to(device)
            labels = labels.to(device)

            outputs = model(images_a, images_b, prompt)

            # Store predictions and labels
            if cfg.model.num_classes == 1:
                preds = (torch.sigmoid(outputs) > cfg.test.threshold).float().squeeze(1).cpu().numpy()
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(preds.flatten())
            all_labels.extend(labels_np.flatten())

            # Optional: save predicted image and comparison image
            if cfg.test.show:
                pred_save_dir = os.path.join(output_dir, 'predictions')
                os.makedirs(pred_save_dir, exist_ok=True)

                # 根据类别数量选择颜色映射
                if cfg.model.num_classes == 1:
                    # 单类别：黑白颜色映射
                    pred_img = (preds[0] * 255).astype(np.uint8)
                    pred_img = Image.fromarray(pred_img, mode='L')  # 'L' 表示灰度模式
                    pred_mask = preds[0].astype(np.uint8)  # 定义 pred_mask
                else:
                    # 多类别：使用多种颜色映射
                    pred_mask = preds[0].astype(np.uint8)
                    # 使用 matplotlib 的颜色映射（例如 jet、viridis 等）
                    cmap = plt.get_cmap('viridis', cfg.model.num_classes)
                    pred_img = (cmap(pred_mask) * 255).astype(np.uint8)
                    pred_img = Image.fromarray(pred_img[:, :, :3])  # 只取 RGB 通道

                # 保存预测图像
                save_path = os.path.join(pred_save_dir, f'{pbar.n}_pred.png')
                pred_img.save(save_path)

                # 可选：保存带掩码的图像（将预测结果叠加到原始图像上）
                if cfg.test.show_overlay:
                    overlay_dir = os.path.join(output_dir, 'overlays')
                    os.makedirs(overlay_dir, exist_ok=True)

                    # 将掩码叠加到原始图像上
                    overlay_img = overlay_mask_on_image(images_a.cpu().numpy()[0], pred_mask, cfg.model.num_classes)
                    overlay_img.save(os.path.join(overlay_dir, f'{pbar.n}_overlay.png'))

                # 保存A, B, target, pred的对比图像
                # 还原 images_a 和 images_b 到原始图像
                images_a_unnorm = reverse_normalize(images_a.cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                images_b_unnorm = reverse_normalize(images_b.cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                comparison_img = create_comparison_image(images_a_unnorm[0], images_b_unnorm[0], labels_np[0], pred_mask, cfg.model.num_classes)
                comparison_img.save(os.path.join(comparison_dir, f'{pbar.n}_comparison.png'))

    if cfg.model.num_classes == 1:
        # Binary classification metrics
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        iou = jaccard_score(all_labels, all_preds)
        oa = accuracy_score(all_labels, all_preds)
    else:
        # Multi-class classification metrics
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        iou = jaccard_score(all_labels, all_preds, average='macro')
        oa = accuracy_score(all_labels, all_preds)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'oa': oa
    }

    return metrics


def reverse_normalize(image_tensor, mean, std):
    """
    逆归一化操作，将归一化的图像张量还原为原始图像
    Args:
        image_tensor: 归一化的图像张量 (Tensor, B x C x H x W)
        mean: 均值 (list)
        std: 标准差 (list)
    Returns:
        还原后的图像张量 (Tensor, B x C x H x W)
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return image_tensor * std + mean


def overlay_mask_on_image(image, mask, num_classes, alpha=0.5):
    """
    将掩码叠加到原始图像上
    Args:
        image: 原始图像 (numpy array, H x W x C)
        mask: 掩码 (numpy array, H x W)
        num_classes: 类别数量
        alpha: 背景图像的透明度
    Returns:
        PIL Image
    """
    # 将图像从 numpy 格式转换为 PIL Image
    image = Image.fromarray((image * 255).astype(np.uint8))

    # 创建掩码图像
    if num_classes == 1:
        # 单类别：黑白掩码
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    else:
        # 多类别：彩色掩码
        cmap = plt.get_cmap('viridis', num_classes)
        mask_colored = (cmap(mask) * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_colored[:, :, :3])

    # 将掩码图像调整为与原始图像相同的尺寸
    mask_img = mask_img.resize(image.size)

    # 将掩码叠加到图像上
    overlay = Image.new('RGBA', image.size)
    overlay.paste(mask_img.convert('RGBA'), (0, 0), mask_img.convert('L'))
    overlayed_img = Image.alpha_composite(image.convert('RGBA'), overlay)

    return overlayed_img



def create_comparison_image(image_a, image_b, target_mask, pred_mask, num_classes):
    """
    创建对比图像，并在原始图像上绘制掩码边界（支持控制线条粗细）
    """
    # 将张量转换为 NumPy 数组并调整形状
    image_a = image_a.numpy().transpose(1, 2, 0)  # C x H x W -> H x W x C
    image_b = image_b.numpy().transpose(1, 2, 0)
    image_a = (image_a * 255).astype(np.uint8)
    image_b = (image_b * 255).astype(np.uint8)

    # 单类别处理
    if num_classes == 1:
        if target_mask.ndim == 3:
            target_mask = target_mask[0]
        if pred_mask.ndim == 3:
            pred_mask = pred_mask[0]

        target_mask_np = (target_mask * 255).astype(np.uint8)
        pred_mask_np = (pred_mask * 255).astype(np.uint8)

        contours_target, _ = cv2.findContours(target_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_pred, _ = cv2.findContours(pred_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 设置颜色和线宽
        color_target = (0, 0, 255)  # BGR 格式，红色
        color_pred = (255, 255, 255)  # 白色
        thickness = 2  # 控制线条粗细


        # 在图像 A 上同时绘制 target（红）和 pred（白）
        image_a_contour = image_a.copy()
        image_a_contour = cv2.drawContours(image_a_contour, contours_target, -1, color_target, thickness)
        image_a_contour = cv2.drawContours(image_a_contour, contours_pred, -1, color_pred, thickness)

        # 在图像 B 上也同时绘制 target（红）和 pred（白）
        image_b_contour = image_b.copy()
        image_b_contour = cv2.drawContours(image_b_contour, contours_target, -1, color_target, thickness)
        image_b_contour = cv2.drawContours(image_b_contour, contours_pred, -1, color_pred, thickness)

    else:
        # 多类别处理（仅绘制前景区域）
        target_mask_binary = (target_mask > 0).astype(np.uint8) * 255
        pred_mask_binary = (pred_mask > 0).astype(np.uint8) * 255

        contours_target, _ = cv2.findContours(target_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_pred, _ = cv2.findContours(pred_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color_target = (0, 0, 255)
        color_pred = (0, 255, 0)
        thickness = 2

        image_a_contour = cv2.drawContours(image_a.copy(), contours_target, -1, color_target, thickness)
        image_b_contour = cv2.drawContours(image_b.copy(), contours_pred, -1, color_pred, thickness)

    # 转换回 PIL 图像用于拼接
    image_a_pil = Image.fromarray(image_a_contour)
    image_b_pil = Image.fromarray(image_b_contour)

    # 掩码图像（保持不变）
    if num_classes == 1:
        target_img = Image.fromarray((target_mask * 255).astype(np.uint8), mode='L')
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
    else:
        cmap = plt.get_cmap('viridis', num_classes)
        target_colored = (cmap(target_mask) * 255).astype(np.uint8)
        pred_colored = (cmap(pred_mask) * 255).astype(np.uint8)
        target_img = Image.fromarray(target_colored[:, :, :3])
        pred_img = Image.fromarray(pred_colored[:, :, :3])

    # 拼接图像
    comparison_img = Image.new('RGB', (image_a_pil.width * 4, image_a_pil.height))
    comparison_img.paste(image_a_pil, (0, 0))
    comparison_img.paste(image_b_pil, (image_a_pil.width, 0))
    comparison_img.paste(target_img.convert('RGB'), (image_a_pil.width * 2, 0))
    comparison_img.paste(pred_img.convert('RGB'), (image_a_pil.width * 3, 0))

    return comparison_img

