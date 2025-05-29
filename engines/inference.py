# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: infer.py
@Time    : 2025/5/20 下午2:51
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 测试建筑物变化检测模型
@Usage   :
"""
import time
import cv2
import math
import torch
import numpy as np
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from models import build_model, build_embs, PostProcessor
from utils import setup_logging, get_output_dir
from tqdm import tqdm
import rasterio


def read_image_in_chunks(image_path, chunk_size=25600):
    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height
        meta = src.meta

        chunks = []
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                window = rasterio.windows.Window(x, y, chunk_size, chunk_size)
                chunk = src.read(window=window)
                chunk = np.transpose(chunk, (1, 2, 0))  # HWC 格式
                chunks.append((x, y, chunk))

        return chunks, (height, width)


def load_model(cfg, weights_dir, device):
    """
    加载模型函数
    Args:
        cfg: 配置文件
    Returns:
        model: nn.Module，加载的模型
    """
    model = build_model(cfg, training=False)
    model.to(device)

    if weights_dir is not None:
        checkpoint = torch.load(weights_dir, map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def crop_image(img_a, img_b, coord, crop_size=512, overlap=0):
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


def calculate_brightness(img):
    """
    计算图像的平均亮度
    Args:
        img: 输入图像（numpy array）
    Returns:
        brightness: 图像的平均亮度
    """
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算平均亮度
    brightness = np.mean(gray)
    return brightness


def gamma_correction(img, target_brightness=128):
    """
    伽马矫正函数，动态调整伽马值
    Args:
        img: 输入图像（numpy array）
        target_brightness: 目标亮度，默认为 128
    Returns:
        img: 经过伽马矫正的图像
    """
    # 计算当前图像的平均亮度
    current_brightness = calculate_brightness(img)
    # 计算伽马值
    gamma = 1.0
    if current_brightness > 0:
        gamma = math.log(target_brightness, 2) / math.log(current_brightness, 2)
    # 将图像归一化到 [0, 1] 范围
    img = img.astype(np.float32) / 255.0
    # 应用伽马矫正
    img = np.power(img, gamma)
    # 将图像恢复到 [0, 255] 范围
    img = (img * 255).astype(np.uint8)
    return img


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
    # # 如果图像尺寸不是 512x512，则进行缩放
    # if img_a.shape[:2] != (512, 512):
    #     img_a = cv2.resize(img_a, (512, 512))
    # if img_b.shape[:2] != (512, 512):
    #     img_b = cv2.resize(img_b, (512, 512))

    # 对图像进行伽马矫正
    # img_a = gamma_correction(img_a)
    # img_b = gamma_correction(img_b)

    # 定义图像预处理变换
    transform = Compose([
        ToTensor(),  # 将图像转换为 PyTorch 张量
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化图像
    ])

    # 应用图像预处理变换
    img_a = transform(img_a)
    img_b = transform(img_b)

    # 添加批次维度并移动到指定设备
    img_a = img_a.unsqueeze(0).to(device)
    img_b = img_b.unsqueeze(0).to(device)

    return img_a, img_b


# def postprocess_mask(mask):
#     """
#     mask后处理函数
#         1. 实现像素面积过滤，分割区域像素面积小于1000像素则舍弃
#         2. 区域连通处理，确保分割区域连通
#     Args:
#         mask: 推理得到的分割掩码
#     Returns:
#         masks: 处理后的分割掩码
#     """
#     # 创建一个空数组，用于存储处理后的结果
#     result_mask = np.zeros_like(mask)
#
#     # 使用OpenCV的connectedComponents函数找到mask中的连通区域
#     num_labels, labels = cv2.connectedComponents(mask)
#
#     # 遍历每个连通区域
#     for label in range(1, num_labels):
#         # 获取当前区域的坐标
#         coords = np.column_stack(np.where(labels == label))
#
#         # 计算区域的高度和宽度
#         y_coords = coords[:, 0]
#         x_coords = coords[:, 1]
#
#         height = np.max(y_coords) - np.min(y_coords) + 1
#         width = np.max(x_coords) - np.min(x_coords) + 1
#
#         # 如果区域的高度大于等于50或者宽度大于等于50，则保留该区域
#         if height >= 50 or width >= 50:
#             result_mask[labels == label] = 255
#
#     # 对结果进行连通区域处理，确保分割区域连通
#     # 检测边缘
#     contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # 创建一个与输入掩码大小相同的空白掩码
#     filled_mask = np.zeros_like(result_mask)
#     # 填充边缘
#     for contour in contours:
#         # 填充轮廓内的区域
#         cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)
#
#     return filled_mask


def slide_window_inference(model, postprocessor, img_a, img_b, embs, device, output_dir, threshold=0.5, crop_size=512, overlap=0,
                           global_coord_offset=None):
    """
    滑动窗口推理函数
    Args:
        model: 训练好的模型
        img_a: 第一期图像数据（numpy array）
        img_b: 第二期图像数据（numpy array）
        device: 设备（CPU/GPU）
        output_dir: 输出目录
        threshold: 推理阈值
        crop_size: 裁剪窗口大小
        overlap: 裁剪窗口的重叠区域
        global_coord_offset: 全局坐标偏移量，用于分块推理时的命名
    Returns:
        masks: 推理得到的分割掩码
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
                outputs = model(img_a_tensor, img_b_tensor, embs)
                preds = (torch.sigmoid(outputs) > threshold).float().squeeze(1).cpu().numpy()

            # 提取掩码
            mask = (preds[0] * 255).astype('uint8')

            # mask后处理
            mask, metrics = postprocessor(mask)

            # 将结果保存到指定目录
            mask_dir = os.path.join(output_dir, 'masks')
            os.makedirs(mask_dir, exist_ok=True)

            # 使用全局坐标偏移量来命名文件
            if global_coord_offset:
                global_x = global_coord_offset[0] + x
                global_y = global_coord_offset[1] + y
                combined_path = os.path.join(mask_dir, f"{global_x}_{global_y}_combined.jpg")
            else:
                combined_path = os.path.join(mask_dir, f"{x}_{y}_combined.jpg")

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

            # 保存拼接后的图像
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
    output_dir = get_output_dir(cfg.infer.output_dir, cfg.infer.name)
    logger = setup_logging(cfg, output_dir)
    logger.info('Inference Log %s' % time.strftime("%c"))
    input_dir = cfg.infer.input_dir
    logger.info('Input on %s' % input_dir)

    device = cfg.infer.device
    threshold = cfg.infer.threshold
    logger.info(f'device: {device}, Threshold: {threshold}')

    # 加载模型
    model = load_model(cfg, cfg.infer.weights_dir, device)

    # 创建后处理实例
    postprocessor = PostProcessor(min_area=2500, max_p_a_ratio=10, min_convexity=0.8)

    # TODO: prompt
    # prompts = [cfg.prompt] if hasattr(cfg, 'prompt') else ['Buildings with changes']
    # 构建词嵌入向量
    prompt = [cfg.prompt] if cfg.prompt is not None else None
    embs = build_embs(prompts=prompt, text_encoder_name=cfg.model.text_encoder_name,
                      freeze_text_encoder=cfg.model.freeze_text_encoder, device=device)
    logger.info(f'desc_embs shape:{embs.shape}')

    filename = [filename for filename in os.listdir(input_dir) if filename.endswith('.tif')]
    filename = sorted(filename)
    a_image_path = os.path.join(input_dir, filename[0])
    b_image_path = os.path.join(input_dir, filename[1])

    if a_image_path is None or b_image_path is None:
        logger.error("No suitable a or b image found.")

    # 判断是否启用分块处理
    chunk_size = getattr(cfg.infer, 'chunk_size', 0)

    if chunk_size > 0:
        logger.info(f"Using chunked inference with chunk_size={chunk_size} and overlap=0")

        # 分块读取图像
        a_chunks, (a_height, a_width) = read_image_in_chunks(a_image_path, chunk_size)
        b_chunks, (b_height, b_width) = read_image_in_chunks(b_image_path, chunk_size)

        logger.info(
            f"a_chunks: {len(a_chunks)}, b_chunks: {len(b_chunks)}, a_size: {a_height, a_width}, b_size: {b_height, b_width}")
        # 创建空掩码用于最终结果
        result_mask = np.zeros((a_height, a_width), dtype=np.uint8)
        count_mask = np.zeros_like(result_mask)

        logger.info("Starting chunk-based inference...")

        # 遍历每个块进行推理
        for (x_a, y_a, img_a_patch), (x_b, y_b, img_b_patch) in zip(a_chunks, b_chunks):
            # 确保两幅图像总大小一致
            if img_a_patch.shape != img_b_patch.shape:
                logger.warning(f"Warning: Image sizes differ. Resizing image b to match a.")
                img_b_patch = cv2.resize(img_b_patch, (img_a_patch.shape[1], img_a_patch.shape[0]))

            # 推理
            with torch.no_grad():
                mask = slide_window_inference(model, postprocessor, img_a_patch, img_b_patch, embs, device, output_dir, threshold,
                                              global_coord_offset=(x_a, y_a))

            # 合并到最终掩码
            result_mask[y_a:y_a + mask.shape[0], x_a:x_a + mask.shape[1]] += mask
            count_mask[y_a:y_a + mask.shape[0], x_a:x_a + mask.shape[1]] += 1

        # 平均融合
        result_mask = (result_mask / count_mask).astype(np.uint8)

    else:
        logger.info("Using full-image inference")
        # 读取图像
        img_a = cv2.imread(a_image_path)
        img_b = cv2.imread(b_image_path)

        # 确保两幅图像大小一致
        if img_a.shape != img_b.shape:
            logger.warning(f"Warning: Image sizes differ. Resizing image b {img_b.shape} to match a {img_a.shape}")
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

        # 将BGR修改为RGB
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        # 推理
        logger.info("Starting full-image inference...")
        result_mask = slide_window_inference(model, postprocessor, img_a, img_b, embs, device, output_dir, threshold)

    # 保存最终掩码为 tif 格式
    output_path = os.path.join(output_dir, "result_mask.tif")
    cv2.imwrite(output_path, result_mask)

    logger.info("✅ 图像推理完成！")
