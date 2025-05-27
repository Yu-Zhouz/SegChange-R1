# -*- coding: utf-8 -*-
"""
@Project : SegChange-R2
@FileName: main_parallel.py
@Time    : 2025/5/26 下午4:55
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 多线程推理引擎
@Usage   : 
"""
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from argparse import ArgumentParser
from torch import nn
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
import rasterio
import logging
import math
from datetime import datetime


def get_args():
    parser = ArgumentParser(description='SegChange ONNX Inference with Batch Processing')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录（包含两个.tif 文件）')
    parser.add_argument('--onnx_model', type=str, required=True, help='ONNX 模型路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--prompt', type=str, default=None, help='文本提示（如 Buildings with changes）')
    parser.add_argument('--chunk_size', type=int, default=0, help='分块大小（默认为 0 表示不分块）')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='设备类型')
    parser.add_argument('--threshold', type=float, default=0.5, help='阈值用于二值化输出')
    args = parser.parse_args()
    return args


def setup_logging(log_dirs, level='INFO'):
    os.makedirs(log_dirs, exist_ok=True)
    log_file = os.path.join(log_dirs, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, 'w', 'utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f'Logging initialized at level {level}')
    return logger


class TextEncoderBert(nn.Module):
    def __init__(self, model_name="bert-base-uncased", device='cuda', freeze_text_encoder=True):
        super().__init__()
        self.device = device
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name).to(device)
        if freeze_text_encoder:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, prompts):
        input_ids = []
        for prompt in prompts:
            inputs = self.tokenizer.encode_plus(
                prompt, add_special_tokens=True, max_length=512,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            input_ids.append(inputs['input_ids'].to(self.device))
        outputs = self.bert(torch.cat(input_ids, dim=0))
        return outputs.last_hidden_state

    def to(self, device):
        self.device = device
        self.bert = self.bert.to(device)
        return self


def build_embs(prompts, text_encoder_name, freeze_text_encoder, device, batch_size=1):
    if not prompts:
        return torch.zeros((batch_size, 512, 768), device=device)
    model = TextEncoderBert(text_encoder_name, device, freeze_text_encoder)
    return model(prompts)


def read_image_in_chunks(image_path, chunk_size=256):
    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height
        chunks = []
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                window = rasterio.windows.Window(x, y, chunk_size, chunk_size)
                chunk = src.read(window=window)
                chunk = np.transpose(chunk, (1, 2, 0))
                chunks.append((x, y, chunk))
        return chunks, (height, width)


def preprocess_image(img_a, img_b):
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_a = transform(img_a).unsqueeze(0)
    img_b = transform(img_b).unsqueeze(0)
    return img_a, img_b


def postprocess_mask(mask, threshold=0.5):
    mask = (mask > threshold).astype(np.uint8) * 255
    mask, _ = cv2.connectedComponents(mask)
    return mask


def batch_inference(
        session,
        img_a_list,
        img_b_list,
        embs_list,
        batch_size=16
):
    batch_img_a = torch.cat(img_a_list, dim=0).numpy()
    batch_img_b = torch.cat(img_b_list, dim=0).numpy()
    batch_embs = torch.cat(embs_list, dim=0).numpy()
    outputs = session.run(
        None,
        {
            session.get_inputs()[0].name: batch_img_a,
            session.get_inputs()[1].name: batch_img_b,
            session.get_inputs()[2].name: batch_embs
        }
    )
    return outputs


def predict_onnx(args):
    output_dir = get_output_dir(args.output_dir, 'infer_results')
    logger = setup_logging(output_dir)
    logger.info('Inference Log %s' % time.strftime("%c"))
    logger.info(f'Input on %s {args.input_dir}，Output on {output_dir}')

    device = args.device
    threshold = args.threshold
    batch_size = args.batch_size
    logger.info(f'device: {device}, Threshold: {threshold}, Batch Size: {batch_size}')

    # TODO: prompt
    prompt = [args.prompt] if args.prompt else None
    embs = build_embs(prompt, "bert-base-uncased", True, device)
    logger.info(f'Embeddings shape:{embs.shape}')

    a_image_path = os.path.join(args.input_dir,
                                sorted([f for f in os.listdir(args.input_dir) if f.endswith('.tif')])[0])
    b_image_path = os.path.join(args.input_dir,
                                sorted([f for f in os.listdir(args.input_dir) if f.endswith('.tif')])[1])
    logger.info(f"Loading images from {a_image_path} and {b_image_path}")

    session = ort.InferenceSession(
        args.onnx_model,
        providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider']
    )
    logger.info(f"Loaded ONNX model from {args.onnx_model}")

    chunk_size = args.chunk_size
    use_chunking = chunk_size > 0

    if use_chunking:
        logger.info(f"Using chunked inference with chunk_size={chunk_size}")
        a_chunks, (a_height, a_width) = read_image_in_chunks(a_image_path, chunk_size)
        b_chunks, (b_height, b_width) = read_image_in_chunks(b_image_path, chunk_size)
        result_mask = np.zeros((a_height, a_width), dtype=np.float32)
        count_mask = np.zeros_like(result_mask)

        batch_img_a = []
        batch_img_b = []
        batch_embs = []
        batch_positions = []

        for (x_a, y_a, img_a_patch), (x_b, y_b, img_b_patch) in zip(a_chunks, b_chunks):
            if img_a_patch.shape != img_b_patch.shape:
                logger.warning(f"Mismatched patch shapes at ({x_a}, {y_a})")
                continue

            img_a_tensor, img_b_tensor = preprocess_image(img_a_patch, img_b_patch)
            batch_img_a.append(img_a_tensor)
            batch_img_b.append(img_b_tensor)
            batch_embs.append(embs)
            batch_positions.append((x_a, y_a))

            if len(batch_img_a) >= batch_size:
                outputs = batch_inference(
                    session,
                    batch_img_a,
                    batch_img_b,
                    batch_embs,
                    batch_size
                )
                for i, (x, y) in enumerate(batch_positions):
                    mask = outputs[0][i, 0]
                    mask = postprocess_mask(mask, threshold)
                    result_mask[y:y + chunk_size, x:x + chunk_size] += mask
                    count_mask[y:y + chunk_size, x:x + chunk_size] += 1
                batch_img_a = []
                batch_img_b = []
                batch_embs = []
                batch_positions = []

        if batch_img_a:
            outputs = batch_inference(
                session,
                batch_img_a,
                batch_img_b,
                batch_embs,
                len(batch_img_a)
            )
            for i, (x, y) in enumerate(batch_positions):
                mask = outputs[0][i, 0]
                mask = postprocess_mask(mask, threshold)
                result_mask[y:y + chunk_size, x:x + chunk_size] += mask
                count_mask[y:y + chunk_size, x:x + chunk_size] += 1

        result_mask = (result_mask / np.clip(count_mask, 1, None)).astype(np.uint8)
    else:
        img_a = cv2.imread(a_image_path)
        img_b = cv2.imread(b_image_path)
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
        img_a_tensor, img_b_tensor = preprocess_image(img_a, img_b)
        outputs = session.run(
            None,
            {
                session.get_inputs()[0].name: img_a_tensor.numpy(),
                session.get_inputs()[1].name: img_b_tensor.numpy(),
                session.get_inputs()[2].name: embs.detach().cpu().numpy()
            }
        )
        result_mask = postprocess_mask(outputs[0][0, 0], threshold)

    output_path = os.path.join(output_dir, "result_mask.tif")
    cv2.imwrite(output_path, result_mask)
    logger.info("✅ 图像推理完成！")


def get_output_dir(output_dir, name):
    base_output_dir = os.path.join(output_dir, name)
    suffix = 0
    while os.path.exists(base_output_dir):
        base_output_dir = f"{os.path.join(output_dir, name)}_{suffix}"
        suffix += 1
    os.makedirs(base_output_dir, exist_ok=True)
    return base_output_dir


if __name__ == "__main__":
    args = get_args()
    predict_onnx(args)
