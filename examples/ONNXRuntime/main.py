# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: main.py
@Time    : 2025/5/23 上午9:03
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 使用 ONNX Runtime 进行推理（支持 Prompt + args 参数）
@Usage   : 显存占用: 大约8G ,推理速度6tb/s(512*512)
"""
import logging
import os
import time
import cv2
import math
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
import rasterio
import torch
from torch import nn
from typing import List
from datetime import datetime
from torchvision.transforms import Compose, ToTensor, Normalize
from transformers import BertTokenizer, BertModel


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='SegChange ONNX Inference')

    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录（包含两个.tif 文件）')
    parser.add_argument('--onnx_model', type=str, required=True, help='ONNX 模型路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--prompt', type=str, default=None, help='文本提示（如 Buildings with changes）')
    parser.add_argument('--chunk_size', type=int, default=0, help='分块大小（默认为 0 表示不分块）')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--threshold', type=float, default=0.5, help='阈值用于二值化输出')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型（cpu/cuda）')

    args = parser.parse_args()

    return args


def get_output_dir(output_dir, name):
    """创建唯一输出目录，若目录已存在则自动添加后缀"""
    base_output_dir = os.path.join(output_dir, name)

    suffix = 0
    while os.path.exists(base_output_dir):
        base_output_dir = f"{os.path.join(output_dir, name)}_{suffix}"
        suffix += 1

    os.makedirs(base_output_dir, exist_ok=True)  # 安全创建目录
    return base_output_dir


def setup_logging(log_dirs, level='INFO'):
    """日志模块。"""
    # 日期命名
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(log_dirs, exist_ok=True)
    log_file = os.path.join(log_dirs, f"{current_date}.log")

    level = getattr(logging, level.upper(), logging.INFO)  # 将字符串转换为日志级别常量

    # 创建日志格式
    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 配置日志记录器
    logger = logging.getLogger()
    if logger.hasHandlers():  # 如果已经存在日志处理器，则先清除
        logger.handlers.clear()  # 清空默认的日志处理器
    logger.setLevel(level)

    # 添加控制台日志处理器
    ch = logging.StreamHandler()
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    # 如果指定了日志文件，添加文件日志处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)  # 确保日志文件的目录存在
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.info(f"日志已初始化，级别为 {logging.getLevelName(level)}。")
    return logger


class TextEncoderBert(nn.Module):
    def __init__(self, model_name="bert-base-uncased", device='cuda', freeze_text_encoder=True):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name).to(self.device)

        # 冻结模型
        if freeze_text_encoder:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, prompts):
        """
        prompts: list of str
        """
        # 对每个prompt单独编码，并收集输入ids
        input_ids_list = []
        for prompt in prompts:
            inputs = self.tokenizer(text=prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids_list.append(inputs["input_ids"])

        # 找到最长的输入长度
        max_length = max([input_ids.shape[1] for input_ids in input_ids_list])

        # 对每个输入进行填充，使其长度一致
        padded_input_ids = []
        for input_ids in input_ids_list:
            # 计算需要填充的长度
            pad_length = max_length - input_ids.shape[1]
            # 填充
            padded_input_ids.append(
                torch.cat([input_ids, torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long)],
                          dim=1))

        # 将填充后的输入合并成一个批次
        batch_input_ids = torch.cat(padded_input_ids, dim=0).to(self.device)
        attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).to(self.device)

        # 通过模型获取隐藏状态
        outputs = self.bert(batch_input_ids, attention_mask=attention_mask)
        # last hidden state: [B, seq_len, hidden_size]
        return outputs.last_hidden_state, batch_input_ids  # 返回CLS token的特征

    def to(self, device):
        self.device = device
        self.bert = self.bert.to(device)
        return self


def build_embs(
        prompts: List[str],
        text_encoder_name: str,
        freeze_text_encoder: bool,
        device: str,
        desc_embs_dir: str = None,
        batch_size: int = 1,
        seq_len: int = 8,
) -> torch.Tensor:
    """
    生成文本描述的嵌入向量，并根据指定长度进行填充或截断。
    """
    if prompts is None or prompts[0] == '':
        # 如果没有提供 prompts，返回零向量
        lang_dim = 2048 if text_encoder_name == "microsoft/phi-1_5" else 768
        return torch.zeros((batch_size, seq_len, lang_dim), device=device)

    # 扩展 prompts 到 batch_size 数量
    prompts = prompts * batch_size if len(prompts) != batch_size else prompts

    # 初始化对应的文本编码器模型
    if text_encoder_name == "bert-base-uncased":
        model = TextEncoderBert(model_name=text_encoder_name, device=device, freeze_text_encoder=freeze_text_encoder)
    else:
        raise NotImplementedError(f"Unsupported text encoder name: {text_encoder_name}")

    # 获取文本嵌入
    desc_embs, _ = model(prompts)

    # # 序列长度调整
    # if desc_embs.shape[1] < seq_len:
    #     pad_size = (0, 0, 0, seq_len - desc_embs.shape[1])  # 只对 seq_len 维度进行填充
    #     desc_embs = torch.nn.functional.pad(desc_embs, pad_size)
    # elif desc_embs.shape[1] > seq_len:
    #     desc_embs = desc_embs[:, :seq_len, :]

    # 保存嵌入向量（如果指定了路径）
    if desc_embs_dir is not None:
        os.makedirs(os.path.dirname(desc_embs_dir), exist_ok=True)
        torch.save(desc_embs, desc_embs_dir)

    return desc_embs


def read_image_in_chunks(image_path, chunk_size=25600):
    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height
        chunks = []
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                window = rasterio.windows.Window(x, y, chunk_size, chunk_size)
                chunk = src.read(window=window)
                chunk = np.transpose(chunk, (1, 2, 0))  # HWC 格式
                if chunk.shape[2] == 4:
                    chunk = chunk[:, :, :3]  # 丢弃 alpha 通道
                # 如果是 uint16，转换为 uint8
                if chunk.dtype == np.uint16:
                    chunk = (chunk / 65535.0 * 255).astype(np.uint8)
                chunks.append((x, y, chunk))
        return chunks, (height, width)


def crop_image(img_a, img_b, coord, crop_size=512, overlap=0):
    x, y = coord
    if x + crop_size > img_a.shape[1]:
        x = img_a.shape[1] - crop_size
    if y + crop_size > img_a.shape[0]:
        y = img_a.shape[0] - crop_size
    cropped_img_a = img_a[y:y + crop_size, x:x + crop_size]
    cropped_img_b = img_b[y:y + crop_size, x:x + crop_size]
    return cropped_img_a, cropped_img_b


def calculate_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def gamma_correction(img, target_brightness=128):
    current_brightness = calculate_brightness(img)
    gamma = 1.0
    if current_brightness > 0:
        gamma = math.log(target_brightness, 2) / math.log(current_brightness, 2)
    img = img.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    return (img * 255).astype(np.uint8)


def preprocess_image(img_a, img_b):
    # TODO: 前处理
    # # 如果图像尺寸不是 512x512，则进行缩放
    # if img_a.shape[:2] != (512, 512):
    #     img_a = cv2.resize(img_a, (512, 512))
    # if img_b.shape[:2] != (512, 512):
    #     img_b = cv2.resize(img_b, (512, 512))

    # 对图像进行伽马矫正
    # img_a = gamma_correction(img_a)
    # img_b = gamma_correction(img_b)

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_a = transform(img_a).unsqueeze(0).numpy()
    img_b = transform(img_b).unsqueeze(0).numpy()
    return img_a, img_b


def postprocess_mask(mask, area_threshold=2500, perimeter_area_ratio_threshold=10, convexity_threshold=0.8):
    # 连通域分析
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白图像用于绘制最终结果
    result_mask = np.zeros_like(mask)

    # 初始化统计信息列表
    valid_areas = []
    valid_perimeter_area_ratios = []
    valid_convexities = []

    for contour in contours:
        # 计算轮廓的面积和周长
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # 跳过面积过小的区域
        if area < area_threshold:
            continue

        # 计算周长与面积比
        if perimeter == 0:
            continue
        perimeter_area_ratio = perimeter / area

        # 如果周长与面积比过高，跳过
        if perimeter_area_ratio > perimeter_area_ratio_threshold:
            continue

        # 计算轮廓的凸包
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        # 计算凸包面积与轮廓面积的比值
        if hull_area == 0:
            continue
        convexity = area / hull_area

        # 如果凸包面积与轮廓面积的比值过低，跳过
        if convexity < convexity_threshold:
            continue

        # 收集有效区域的统计信息
        valid_areas.append(area)
        valid_perimeter_area_ratios.append(perimeter_area_ratio)
        valid_convexities.append(convexity)

        # 绘制保留的区域
        cv2.drawContours(result_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 计算最终统计信息
    min_area = min(valid_areas) if valid_areas else None
    max_ratio = max(valid_perimeter_area_ratios) if valid_perimeter_area_ratios else None
    min_convexity = min(valid_convexities) if valid_convexities else None

    return result_mask, (min_area, max_ratio, min_convexity)


def slide_window_inference_onnx(session, img_a, img_b, embs, output_dir, threshold=0.5, crop_size=512, overlap=0,
                                global_coord_offset=None):
    height, width, _ = img_a.shape
    result_mask = np.zeros((height, width), dtype=np.uint8)
    stride = crop_size - overlap
    num_rows = int(np.ceil((height - crop_size) / stride)) + 1
    num_cols = int(np.ceil((width - crop_size) / stride)) + 1
    total_windows = num_rows * num_cols
    pbar = tqdm(total=total_windows, desc="Processing windows", unit="window", mininterval=1.0)

    for i in range(num_rows):
        for j in range(num_cols):
            x = j * stride
            y = i * stride
            if x + crop_size > width:
                x = width - crop_size
            if y + crop_size > height:
                y = height - crop_size
            img_a_patch, img_b_patch = crop_image(img_a, img_b, (x, y), crop_size, overlap)
            img_a_tensor, img_b_tensor = preprocess_image(img_a_patch, img_b_patch)

            inputs = {
                session.get_inputs()[0].name: img_a_tensor.astype(np.float32),
                session.get_inputs()[1].name: img_b_tensor.astype(np.float32),
                session.get_inputs()[2].name: embs.detach().cpu().numpy().astype(np.float32),
            }

            outputs = session.run(None, inputs)
            preds = (outputs[0] > threshold).astype('uint8') * 255
            mask = preds[0, 0]

            # TODO: 后处理 PostProcessor
            # mask, _ = postprocess_mask(mask)

            mask_dir = os.path.join(output_dir, 'masks')
            os.makedirs(mask_dir, exist_ok=True)
            if global_coord_offset:
                gx, gy = global_coord_offset
                combined_path = os.path.join(mask_dir, f"{gx + x}_{gy + y}_combined.jpg")
            else:
                combined_path = os.path.join(mask_dir, f"{x}_{y}_combined.jpg")

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_boundary = np.zeros_like(mask)
            cv2.drawContours(mask_boundary, contours, -1, (255, 0, 0), 2)
            img_a_patch_bgr = cv2.add(img_a_patch, cv2.cvtColor(mask_boundary, cv2.COLOR_GRAY2BGR))
            img_b_patch_bgr = cv2.add(img_b_patch, cv2.cvtColor(mask_boundary, cv2.COLOR_GRAY2BGR))
            mask_pil = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

            combined_pil = Image.new('RGB', (img_a_patch.shape[1] + img_b_patch.shape[1] + mask_pil.width,
                                             max(img_a_patch.shape[0], img_b_patch.shape[0], mask_pil.height)))
            combined_pil.paste(Image.fromarray(img_a_patch_bgr), (0, 0))
            combined_pil.paste(Image.fromarray(img_b_patch_bgr), (img_a_patch.shape[1], 0))
            combined_pil.paste(mask_pil, (img_a_patch.shape[1] + img_b_patch.shape[1], 0))
            combined_pil.save(combined_path)

            result_mask[y:y + crop_size, x:x + crop_size] += mask
            pbar.update(1)
    pbar.close()
    result_mask = (result_mask / np.clip(result_mask.max(), 1, None)).astype(np.uint8) * 255
    return result_mask


def predict_onnx(args):
    name = os.path.basename(os.path.normpath(args.input_dir))
    output_dir = get_output_dir(args.output_dir, name)
    logger = setup_logging(log_dirs=output_dir)
    logger.info('Inference Log %s' % time.strftime("%c"))
    logger.info(f'Input on %s {args.input_dir}，Output on {output_dir}')

    logger.info('ONNX Inference Log %s' % time.strftime("%c"))

    device = args.device
    threshold = args.threshold
    batch_size = args.batch_size
    logger.info(f'device: {device}, Threshold: {threshold}, Batch Size: {batch_size}')

    # TODO: prompt
    # prompts = [cfg.prompt] if hasattr(cfg, 'prompt') else ['Buildings with changes']
    # 构建词嵌入向量
    prompt = [args.prompt] if args.prompt is not None else None
    embs = build_embs(prompts=prompt, text_encoder_name="bert-base-uncased",
                      freeze_text_encoder=True, device=device)
    logger.info(f'Embeddings shape:{embs.shape}')

    # 获取输入图像路径
    filename = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.tif')])
    a_image_path = os.path.join(args.input_dir, filename[0])
    b_image_path = os.path.join(args.input_dir, filename[1])

    if not os.path.exists(a_image_path) or not os.path.exists(b_image_path):
        raise FileNotFoundError("No suitable a or b image found.")

    # 加载ONNX模型
    logger.info(f"Loading ONNX model from {args.onnx_model}")
    session = ort.InferenceSession(
        args.onnx_model,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    chunk_size = args.chunk_size
    use_chunking = chunk_size > 0

    if use_chunking:
        logger.info(f"Using chunked inference with chunk_size={chunk_size} and overlap=0")
        a_chunks, (a_height, a_width) = read_image_in_chunks(a_image_path, chunk_size)
        b_chunks, (b_height, b_width) = read_image_in_chunks(b_image_path, chunk_size)

        result_mask = np.zeros((a_height, a_width), dtype=np.uint8)
        count_mask = np.zeros_like(result_mask)

        for (x_a, y_a, img_a_patch), (x_b, y_b, img_b_patch) in zip(a_chunks, b_chunks):
            if img_a_patch.shape != img_b_patch.shape:
                img_b_patch = cv2.resize(img_b_patch, (img_a_patch.shape[1], img_a_patch.shape[0]))
            mask = slide_window_inference_onnx(session, img_a_patch, img_b_patch, embs, output_dir, threshold,
                                               global_coord_offset=(x_a, y_a))
            result_mask[y_a:y_a + mask.shape[0], x_a:x_a + mask.shape[1]] += mask
            count_mask[y_a:y_a + mask.shape[0], x_a:x_a + mask.shape[1]] += 1
        result_mask = (result_mask / np.clip(count_mask, 1, None)).astype(np.uint8)
    else:
        logger.info("Using full-image inference")
        img_a = cv2.imread(a_image_path)
        img_b = cv2.imread(b_image_path)
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        result_mask = slide_window_inference_onnx(session, img_a, img_b, embs, output_dir, threshold)

    output_path = os.path.join(output_dir, "result_mask.tif")
    cv2.imwrite(output_path, result_mask)
    logger.info("✅ 图像推理完成！")


if __name__ == "__main__":
    args = get_args()
    predict_onnx(args)
