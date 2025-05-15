# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: inference.py
@Time    : 2025/4/24 下午5:24
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 测试建筑物变化检测模型
@Usage   :
"""
import argparse
import time

import cv2
import torch
import numpy as np
import os

from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter, GaussianBlur
from models import build_model
from utils import load_config, setup_logging, get_output_dir
from tqdm import tqdm  # 引入tqdm包用于进度显示


def get_args_config():
    """
    参数包括了: input_dir weights_dir output_dir threshold
    """
    parser = argparse.ArgumentParser('SegChange')
    parser.add_argument('-c', '--config', type=str, required=True, help='The path of config file')
    args = parser.parse_args()
    if args.config is not None:
        cfg = load_config(args.config)
    else:
        raise ValueError('Please specify the config file')
    return cfg


def load_model(cfg):
    """
    加载模型函数
    Args:
        cfg: 配置文件
    Returns:
        model: nn.Module，加载的模型
    """
    model = build_model(cfg, training=False)
    model.to(cfg.infer.device)

    if cfg.infer.weights_dir is not None:
        checkpoint = torch.load(cfg.infer.weights_dir, map_location=cfg.infer.device)
        model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def crop_image(img_a, img_b, coord, crop_size=512, overlap=256):
    """
    裁剪图像函数
    Args:
        img_a: 第一期图像数据（numpy array）
        img_b: 第二期图像数据（numpy array）
        coord: 裁剪区域的左上角坐标（x, y）
        crop_size: 裁剪区域的大小
        overlap: 裁剪窗口的重叠区域
    Returns:
        cropped_img_a: 裁剪后的第一期图像
        cropped_img_b: 裁剪后的第二期图像
    """
    x, y = coord

    # 确保裁剪区域不会因为重叠而超出图像边界
    # 计算裁剪区域的右下角坐标
    x_end = x + crop_size
    y_end = y + crop_size

    # 如果裁剪区域超出图像边界，调整裁剪区域的起点
    if x_end > img_a.shape[1]:
        x = img_a.shape[1] - crop_size
    if y_end > img_a.shape[0]:
        y = img_a.shape[0] - crop_size

    cropped_img_a = img_a[y:y + crop_size, x:x + crop_size]
    cropped_img_b = img_b[y:y + crop_size, x:x + crop_size]

    return cropped_img_a, cropped_img_b


def preprocess_image(img_a, img_b, device):
    """
    预处理图像函数
    Args:
        img_a: 第一期图像数据（numpy array）
        img_b: 第二期图像数据（numpy array）
        device: 设备（CPU/GPU）
    Returns:
        img_a_tensor: torch.Tensor，处理后的第一期图像
        img_b_tensor: torch.Tensor，处理后的第二期图像
    """
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_a = transform(img_a)
    img_b = transform(img_b)

    img_a = img_a.unsqueeze(0).to(device)
    img_b = img_b.unsqueeze(0).to(device)

    return img_a, img_b


def postprocess_mask(mask):
    """
    mask后处理函数
        1.实现50*50的mask区域过滤，只需要满足高或宽小于50即过滤掉
        2.区域连通处理，例如有些分割区域中间有没分割的区域，则将其变为分割区域，形成一个连通区域
    Args:
        mask: 推理得到的分割掩码
    Returns:
        mask: 处理后的分割掩码
    """
    # 创建一个空数组，用于存储处理后的结果
    result_mask = np.zeros_like(mask)

    # 使用OpenCV的connectedComponents函数找到mask中的连通区域
    num_labels, labels = cv2.connectedComponents(mask)

    # 遍历每个连通区域
    for label in range(1, num_labels):
        # 获取当前区域的坐标
        coords = np.column_stack(np.where(labels == label))

        # 计算区域的高度和宽度
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]

        height = np.max(y_coords) - np.min(y_coords) + 1
        width = np.max(x_coords) - np.min(x_coords) + 1

        # 如果区域的高度大于等于50或者宽度大于等于50，则保留该区域
        if height >= 50 or width >= 50:
            result_mask[labels == label] = 255

    # 对结果进行连通区域处理，确保分割区域连通
    # 检测边缘
    contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 创建一个与输入掩码大小相同的空白掩码
    filled_mask = np.zeros_like(result_mask)
    # 填充边缘
    for contour in contours:
        # 填充轮廓内的区域
        cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filled_mask


def slide_window_inference(model, img_a, img_b, device, output_dir, crop_size=512, overlap=0):
    """
    滑动窗口推理函数
    Args:
        model: 训练好的模型
        img_a: 第一期图像数据（numpy array）
        img_b: 第二期图像数据（numpy array）
        device: 设备（CPU/GPU）
        output_dir: 输出目录
        crop_size: 裁剪窗口大小
        overlap: 裁剪窗口的重叠区域
    Returns:
        mask: 推理得到的分割掩码
    """
    height, width, _ = img_a.shape

    # 创建一个空的结果数组
    result_mask = np.zeros((height, width), dtype=np.uint8)

    # 计算需要裁剪的行数和列数
    stride = crop_size - overlap
    num_rows = int(np.ceil((height - crop_size) / stride)) + 1
    num_cols = int(np.ceil((width - crop_size) / stride)) + 1
    total_windows = num_rows * num_cols

    # 创建一个计数器，用于计算每个像素被覆盖的次数（用于平均融合）
    count_mask = np.zeros((height, width), dtype=np.uint8)

    # 初始化进度条
    pbar = tqdm(total=total_windows, desc="Processing windows", unit="window", mininterval=1.0)

    for i in range(num_rows):
        for j in range(num_cols):
            # 计算裁剪区域的左上角坐标
            x = j * stride
            y = i * stride

            # 处理边界情况
            if x + crop_size > width:
                x = width - crop_size
            if y + crop_size > height:
                y = height - crop_size

            # 裁剪图像
            img_a_patch, img_b_patch = crop_image(img_a, img_b, (x, y), crop_size, overlap)

            # 预处理裁剪后的图像
            img_a_tensor, img_b_tensor = preprocess_image(img_a_patch, img_b_patch, device)

            # 推理
            with torch.no_grad():
                outputs = model(img_a_tensor, img_b_tensor)
                preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(1).cpu().numpy()

            # 提取掩码
            mask = (preds[0] * 255).astype('uint8')

            # mask后处理
            mask = postprocess_mask(mask)
            # 将结果保存到指定目录
            mask_dir = os.path.join(output_dir, 'masks')
            os.makedirs(mask_dir, exist_ok=True)
            # mask_path = os.path.join(mask_dir, f"{x}_{y}.jpg")
            # cv2.imwrite(mask_path, mask)

            # 绘制mask边界
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_boundary = np.zeros_like(mask)
            cv2.drawContours(mask_boundary, contours, -1, (255, 0, 0), 2)

            # 在原图上绘制mask边界
            img_a_patch_bgr = cv2.add(img_a_patch, cv2.cvtColor(mask_boundary, cv2.COLOR_GRAY2BGR))
            img_b_patch_bgr = cv2.add(img_b_patch, cv2.cvtColor(mask_boundary, cv2.COLOR_GRAY2BGR))

            # 将 NumPy 数组转换为 PIL 图像
            img_a_pil = Image.fromarray(img_a_patch_bgr)
            img_b_pil = Image.fromarray(img_b_patch_bgr)
            mask_pil = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

            # 拼接三张图像 (水平方向)
            combined_pil = Image.new('RGB', (img_a_pil.width + img_b_pil.width + mask_pil.width,
                                             max(img_a_pil.height, img_b_pil.height, mask_pil.height)))

            # 依次粘贴图像
            combined_pil.paste(img_a_pil, (0, 0))
            combined_pil.paste(img_b_pil, (img_a_pil.width, 0))
            combined_pil.paste(mask_pil, (img_a_pil.width + img_b_pil.width, 0))

            # 构建路径并保存
            combined_path = os.path.join(mask_dir, f"{x}_{y}_combined.jpg")
            combined_pil.save(combined_path)

            # 将结果融合到最终掩码
            result_mask[y:y + crop_size, x:x + crop_size] += mask
            count_mask[y:y + crop_size, x:x + crop_size] += 1

            # 更新进度条
            pbar.update(1)  # 更新进度条

    pbar.close()  # 关闭进度条

    # 计算最终掩码（平均融合）
    result_mask = (result_mask / count_mask).astype(np.uint8)

    return result_mask


def predict(cfg):
    """
    推理结果
    Args:
        cfg: 配置文件
    Returns:
        mask: 预测结果
    """
    output_dir = get_output_dir(cfg.infer.output_dir, cfg.infer.name)
    logger = setup_logging(cfg, output_dir)
    logger.info('Inference Log %s' % time.strftime("%c"))
    input_dir = cfg.infer.input_dir

    # 初始化变量来存储 a 和 b 图像的路径及对应的数字
    filename = [filename for filename in os.listdir(input_dir) if filename.endswith('.tif')]
    # 对filename列表进行排序
    filename = sorted(filename)
    a_image_path = os.path.join(input_dir, filename[0])
    b_image_path = os.path.join(input_dir, filename[1])
    # 如果没有找到合适的 a 或 b 图像，抛出错误
    if a_image_path is None or b_image_path is None:
        logger.error("No suitable a or b image found.")

    # 读取图像
    img_a = cv2.imread(a_image_path)
    img_b = cv2.imread(b_image_path)

    # 确保两幅图像大小一致
    print(img_a.shape, img_b.shape)
    if img_a.shape != img_b.shape:
        logger.warning(f"Warning: Image sizes differ. Resizing image b {img_b.shape} to match a {img_a.shape}")
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    # 将BGR修改为RGB
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

    # 加载模型
    model = load_model(cfg)

    # 添加推理进度条
    print("Starting inference...")
    mask = slide_window_inference(model, img_a, img_b, cfg.infer.device, output_dir)

    # 保存最终掩码为 tif 格式
    output_path = os.path.join(output_dir, "result_mask.tif")
    cv2.imwrite(output_path, mask)

    logger.info("✅ 图像推理完成！")
    return mask


def main():
    cfg = get_args_config()
    mask = predict(cfg)


if __name__ == "__main__":
    main()
