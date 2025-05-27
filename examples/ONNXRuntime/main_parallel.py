# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: main.py
@Time    : 2025/5/23 上午9:03
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 使用 ONNX Runtime 进行异步多线程推理（支持 Prompt + args 参数）
@Usage   : 显存占用: 大约8G ,推理速度6tb/s(512*512)
--input_dir ../../data/BL1_100 --onnx_model ./weights/segchange.onnx --output_dir ./results/ --chunk_size 25600 --device 'cuda'

export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=1 python ./examples/ONNXRuntime/main_parallel.py \
  --input_dir ./data/BL1_100 \
  --onnx_model ./examples/ONNXRuntime/weights/segchange.onnx \
  --output_dir ./examples/ONNXRuntime/results/ \
  --chunk_size 25600 \
  --device 'cuda'
"""
import logging
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import rasterio
import torch
from torch import nn
from typing import List
from datetime import datetime
from torchvision.transforms import Compose, ToTensor, Normalize
from transformers import BertTokenizer, BertModel
import threading
import queue
from dataclasses import dataclass
from collections import deque
import gc
import psutil
import GPUtil


@dataclass
class WindowTask:
    """窗口任务数据结构"""
    x: int
    y: int
    img_a_patch: np.ndarray
    img_b_patch: np.ndarray
    global_x: int = 0
    global_y: int = 0


@dataclass
class BatchTask:
    """批次任务数据结构"""
    tasks: List[WindowTask]
    batch_id: int


@dataclass
class InferenceResult:
    """推理结果数据结构"""
    x: int
    y: int
    mask: np.ndarray
    global_x: int = 0
    global_y: int = 0


@dataclass
class BatchResult:
    """批次推理结果数据结构"""
    results: List[InferenceResult]
    batch_id: int


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, logger):
        self.logger = logger
        self.start_time = time.time()
        self.crop_times = deque(maxlen=100)  # 保留最近100个切片时间
        self.inference_times = deque(maxlen=100)  # 保留最近100个推理时间
        self.total_windows = 0  # 统一使用total_windows
        self.completed_windows = 0  # 统一使用completed_windows
        self.total_batches = 0
        self.completed_batches = 0
        self.lock = threading.Lock()

    def add_crop_time(self, crop_time: float):
        with self.lock:
            self.crop_times.append(crop_time)

    def add_inference_time(self, inference_time: float, batch_size: int = 1):
        with self.lock:
            self.inference_times.append(inference_time)
            self.completed_windows += batch_size
            self.completed_batches += 1

    def set_total_windows(self, total: int):
        with self.lock:
            self.total_windows = total

    def set_total_batches(self, total: int):
        with self.lock:
            self.total_batches = total

    def get_avg_crop_time(self) -> float:
        with self.lock:
            return sum(self.crop_times) / len(self.crop_times) if self.crop_times else 0

    def get_avg_inference_time(self) -> float:
        with self.lock:
            return sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0

    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time

    def get_remaining_windows(self) -> int:
        with self.lock:
            return max(0, self.total_windows - self.completed_windows)

    def get_completion_rate(self) -> float:
        with self.lock:
            if self.total_windows == 0:
                return 0.0
            return (self.completed_windows / self.total_windows) * 100

    @staticmethod
    def format_seconds(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def log_progress(self, crop_queue_size: int, inference_queue_size: int):
        elapsed = self.format_seconds(self.get_elapsed_time())
        remaining = self.get_remaining_windows()
        avg_crop = self.get_avg_crop_time()
        avg_inference = self.get_avg_inference_time()
        completion_rate = self.get_completion_rate()

        with self.lock:
            self.logger.info(
                f"进度监控 - 运行时间: {elapsed:.1f}s | "
                f"剩余切片: {remaining} | "
                f"切片队列: {crop_queue_size} | "
                f"推理队列: {inference_queue_size} | "
                f"平均切片时间: {avg_crop:.6f}s | "
                f"平均推理时间: {avg_inference:.6f}s | "
                f"批次进度: {self.completed_batches}/{self.total_batches} | "
                f"完成率: {completion_rate:.1f}%"
            )


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='SegChange ONNX Inference')

    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录（包含两个.tif 文件）')
    parser.add_argument('--onnx_model', type=str, required=True, help='ONNX 模型路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--prompt', type=str, default='Buildings with changes, Mound changes.',
                        help='文本提示（如 Buildings with changes）')
    parser.add_argument('--chunk_size', type=int, default=0, help='分块大小（默认为 0 表示不分块）')
    parser.add_argument('--batch_size', type=int, default=2, help='推理批次大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型（cpu/cuda）')
    parser.add_argument('--crop_threads', type=int, default=2, help='切片线程数')
    parser.add_argument('--inference_threads', type=int, default=1, help='推理线程数')
    parser.add_argument('--max_queue_size', type=int, default=50, help='最大队列大小')
    parser.add_argument('--log_interval', type=int, default=1, help='日志输出间隔（秒）')

    args = parser.parse_args()
    return args


def get_output_dir(output_dir, name):
    """创建唯一输出目录，若目录已存在则自动添加后缀"""
    base_output_dir = os.path.join(output_dir, name)

    suffix = 0
    while os.path.exists(base_output_dir):
        base_output_dir = f"{os.path.join(output_dir, name)}_{suffix}"
        suffix += 1

    os.makedirs(base_output_dir, exist_ok=True)
    return base_output_dir


def setup_logging(log_dirs, level='INFO'):
    """日志模块。"""
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(log_dirs, exist_ok=True)
    log_file = os.path.join(log_dirs, f"{current_date}.log")

    level = getattr(logging, level.upper(), logging.INFO)
    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.info(f"日志已初始化，级别为 {logging.getLevelName(level)}。")
    return logger


def get_memory_info():
    """获取内存和显存信息"""
    # CPU内存
    cpu_memory = psutil.virtual_memory()
    cpu_used_gb = cpu_memory.used / (1024 ** 3)
    cpu_total_gb = cpu_memory.total / (1024 ** 3)

    # GPU显存
    gpu_info = ""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # 假设使用第一个GPU
            gpu_used_gb = gpu.memoryUsed / 1024
            gpu_total_gb = gpu.memoryTotal / 1024
            gpu_info = f"GPU: {gpu_used_gb:.1f}/{gpu_total_gb:.1f}GB ({gpu.memoryUtil * 100:.1f}%)"
    except:
        gpu_info = "GPU: 信息获取失败"

    return f"CPU: {cpu_used_gb:.1f}/{cpu_total_gb:.1f}GB ({cpu_memory.percent:.1f}%), {gpu_info}"


def create_optimized_onnx_session(model_path, device='cuda'):
    """创建优化的ONNX会话，减少显存占用"""
    providers = []
    session_options = ort.SessionOptions()

    if device == 'cuda':
        # CUDA优化选项
        cuda_provider_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',  # 按需分配显存
            'gpu_mem_limit': 6 * 1024 * 1024 * 1024,  # 限制显存使用到6GB
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }
        providers.append(('CUDAExecutionProvider', cuda_provider_options))

    providers.append('CPUExecutionProvider')

    # 会话优化
    session_options.enable_mem_pattern = False  # 禁用内存模式优化，减少显存占用
    session_options.enable_cpu_mem_arena = False  # 禁用CPU内存池
    session_options.enable_profiling = False

    session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
    return session


def adaptive_batch_size(initial_batch_size, available_memory_gb):
    """根据可用显存动态调整批次大小"""
    if available_memory_gb < 4:
        return 1
    elif available_memory_gb < 6:
        return min(2, initial_batch_size)
    elif available_memory_gb < 8:
        return min(4, initial_batch_size)
    else:
        return initial_batch_size


class TextEncoderBert(nn.Module):
    def __init__(self, model_name="bert-base-uncased", device='cuda', freeze_text_encoder=True):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name).to(self.device)

        if freeze_text_encoder:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, prompts):
        input_ids_list = []
        for prompt in prompts:
            inputs = self.tokenizer(text=prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids_list.append(inputs["input_ids"])

        max_length = max([input_ids.shape[1] for input_ids in input_ids_list])

        padded_input_ids = []
        for input_ids in input_ids_list:
            pad_length = max_length - input_ids.shape[1]
            padded_input_ids.append(
                torch.cat([input_ids, torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long)],
                          dim=1))

        batch_input_ids = torch.cat(padded_input_ids, dim=0).to(self.device)
        attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).to(self.device)

        outputs = self.bert(batch_input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, batch_input_ids

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
    """生成文本描述的嵌入向量"""
    if prompts is None or prompts[0] == '':
        lang_dim = 2048 if text_encoder_name == "microsoft/phi-1_5" else 768
        return torch.zeros((batch_size, seq_len, lang_dim), device=device)

    prompts = prompts * batch_size if len(prompts) != batch_size else prompts

    if text_encoder_name == "bert-base-uncased":
        model = TextEncoderBert(model_name=text_encoder_name, device=device, freeze_text_encoder=freeze_text_encoder)
    else:
        raise NotImplementedError(f"Unsupported text encoder name: {text_encoder_name}")

    desc_embs, _ = model(prompts)

    if desc_embs_dir is not None:
        os.makedirs(os.path.dirname(desc_embs_dir), exist_ok=True)
        torch.save(desc_embs, desc_embs_dir)

    return desc_embs


def read_image_in_chunks(image_path, chunk_size=25600):
    """分块读取大图像"""
    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height
        chunks = []
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                window = rasterio.windows.Window(x, y, chunk_size, chunk_size)
                chunk = src.read(window=window)
                chunk = np.transpose(chunk, (1, 2, 0))  # HWC 格式
                chunks.append((x, y, chunk))
        return chunks, (height, width)


def crop_image(img_a, img_b, coord, crop_size=512, overlap=0):
    """裁剪图像"""
    x, y = coord
    if x + crop_size > img_a.shape[1]:
        x = img_a.shape[1] - crop_size
    if y + crop_size > img_a.shape[0]:
        y = img_a.shape[0] - crop_size
    cropped_img_a = img_a[y:y + crop_size, x:x + crop_size]
    cropped_img_b = img_b[y:y + crop_size, x:x + crop_size]
    return cropped_img_a, cropped_img_b


def preprocess_image_batch(img_patches_a: List[np.ndarray], img_patches_b: List[np.ndarray]):
    """批量图像预处理"""
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_a = []
    batch_b = []

    for img_a, img_b in zip(img_patches_a, img_patches_b):
        tensor_a = transform(img_a)
        tensor_b = transform(img_b)
        batch_a.append(tensor_a)
        batch_b.append(tensor_b)

    # 堆叠成批次张量
    batch_a_tensor = torch.stack(batch_a).numpy()
    batch_b_tensor = torch.stack(batch_b).numpy()

    return batch_a_tensor, batch_b_tensor


def preprocess_image(img_a, img_b):
    """单个图像预处理（保持向后兼容）"""
    return preprocess_image_batch([img_a], [img_b])


def postprocess_mask(mask, area_threshold=2500, perimeter_area_ratio_threshold=10, convexity_threshold=0.8):
    """后处理掩码"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_mask = np.zeros_like(mask)
    valid_areas = []
    valid_perimeter_area_ratios = []
    valid_convexities = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < area_threshold:
            continue

        if perimeter == 0:
            continue
        perimeter_area_ratio = perimeter / area

        if perimeter_area_ratio > perimeter_area_ratio_threshold:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue
        convexity = area / hull_area

        if convexity < convexity_threshold:
            continue

        valid_areas.append(area)
        valid_perimeter_area_ratios.append(perimeter_area_ratio)
        valid_convexities.append(convexity)

        cv2.drawContours(result_mask, [contour], -1, 255, thickness=cv2.FILLED)

    min_area = min(valid_areas) if valid_areas else None
    max_ratio = max(valid_perimeter_area_ratios) if valid_perimeter_area_ratios else None
    min_convexity = min(valid_convexities) if valid_convexities else None

    return result_mask, (min_area, max_ratio, min_convexity)


def create_cropping_tasks(img_a, img_b, crop_size=512, overlap=0, global_coord_offset=None):
    """创建裁剪任务列表"""
    height, width, _ = img_a.shape
    stride = crop_size - overlap
    num_rows = int(np.ceil((height - crop_size) / stride)) + 1
    num_cols = int(np.ceil((width - crop_size) / stride)) + 1

    tasks = []
    for i in range(num_rows):
        for j in range(num_cols):
            x = j * stride
            y = i * stride
            if x + crop_size > width:
                x = width - crop_size
            if y + crop_size > height:
                y = height - crop_size

            img_a_patch, img_b_patch = crop_image(img_a, img_b, (x, y), crop_size, overlap)

            global_x = global_coord_offset[0] + x if global_coord_offset else x
            global_y = global_coord_offset[1] + y if global_coord_offset else y

            task = WindowTask(x, y, img_a_patch, img_b_patch, global_x, global_y)
            tasks.append(task)

    return tasks


def create_batch_tasks(tasks: List[WindowTask], batch_size: int) -> List[BatchTask]:
    """将窗口任务分组成批次任务"""
    batch_tasks = []
    batch_id = 0

    for i in range(0, len(tasks), batch_size):
        batch_tasks.append(BatchTask(
            tasks=tasks[i:i + batch_size],
            batch_id=batch_id
        ))
        batch_id += 1

    return batch_tasks


def crop_worker_optimized(crop_queue: queue.Queue, inference_queue: queue.Queue,
                          monitor: PerformanceMonitor, stop_event: threading.Event,
                          batch_size: int, max_inference_queue_size: int):
    """优化的切片工作线程，支持背压控制"""
    task_buffer = []
    batch_id = 0

    while not stop_event.is_set():
        try:
            # 背压控制：如果推理队列太满，暂停切片
            while inference_queue.qsize() > max_inference_queue_size and not stop_event.is_set():
                time.sleep(0.1)

            task = crop_queue.get(timeout=1)
            if task is None:  # 毒丸信号
                if task_buffer:
                    batch_task = BatchTask(task_buffer.copy(), batch_id)
                    inference_queue.put(batch_task)
                    task_buffer.clear()
                break

            start_time = time.time()
            task_buffer.append(task)
            crop_time = time.time() - start_time
            monitor.add_crop_time(crop_time)

            # 当缓冲区达到批次大小时立即发送
            if len(task_buffer) >= batch_size:
                batch_task = BatchTask(task_buffer.copy(), batch_id)
                inference_queue.put(batch_task)
                task_buffer.clear()
                batch_id += 1

            crop_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"切片工作线程出错: {e}")

    # 处理剩余任务
    if task_buffer:
        batch_task = BatchTask(task_buffer.copy(), batch_id)
        inference_queue.put(batch_task)


def inference_worker_optimized(inference_queue: queue.Queue, result_queue: queue.Queue,
                               session: ort.InferenceSession, embs: torch.Tensor,
                               output_dir: str, monitor: PerformanceMonitor,
                               stop_event: threading.Event, max_batch_size: int,
                               worker_id: int = 0):
    """优化的推理工作线程，包含显存管理"""
    logger = logging.getLogger()
    current_batch_size = max_batch_size
    consecutive_errors = 0

    while not stop_event.is_set():
        try:
            batch_task = inference_queue.get(timeout=1)
            if batch_task is None:  # 毒丸信号
                break

            start_time = time.time()

            # 动态调整批次大小
            actual_batch_size = min(len(batch_task.tasks), current_batch_size)
            tasks_to_process = batch_task.tasks[:actual_batch_size]
            remaining_tasks = batch_task.tasks[actual_batch_size:]

            # 如果有剩余任务，重新放回队列
            if remaining_tasks:
                remaining_batch = BatchTask(remaining_tasks, batch_task.batch_id)
                inference_queue.put(remaining_batch)

            # 准备批量数据
            img_patches_a = [task.img_a_patch for task in tasks_to_process]
            img_patches_b = [task.img_b_patch for task in tasks_to_process]

            try:
                # 批量预处理
                img_a_batch, img_b_batch = preprocess_image_batch(img_patches_a, img_patches_b)

                # 调整嵌入向量以匹配批次大小
                if embs.shape[0] == 1 and actual_batch_size > 1:
                    batch_embs = embs.repeat(actual_batch_size, 1, 1)
                elif embs.shape[0] >= actual_batch_size:
                    batch_embs = embs[:actual_batch_size]
                else:
                    batch_embs = torch.cat([
                        embs,
                        embs[-1:].repeat(actual_batch_size - embs.shape[0], 1, 1)
                    ])

                # ONNX批量推理
                inputs = {
                    session.get_inputs()[0].name: img_a_batch.astype(np.float32),
                    session.get_inputs()[1].name: img_b_batch.astype(np.float32),
                    session.get_inputs()[2].name: batch_embs.detach().cpu().numpy().astype(np.float32),
                }

                outputs = session.run(None, inputs)
                batch_preds = (outputs[0] > 0.5).astype('uint8') * 255

                # 处理批量结果
                results = []
                for i, task in enumerate(tasks_to_process):
                    mask = batch_preds[i, 0]  # 取第i个样本的第0个通道

                    # 后处理
                    mask, _ = postprocess_mask(mask)

                    # 保存可视化结果
                    save_visualization(task, mask, output_dir)

                    # 创建结果
                    result = InferenceResult(task.x, task.y, mask, task.global_x, task.global_y)
                    results.append(result)

                # 将批量结果放入结果队列
                batch_result = BatchResult(results, batch_task.batch_id)
                result_queue.put(batch_result)

                inference_time = time.time() - start_time
                monitor.add_inference_time(inference_time, actual_batch_size)

                # 推理成功，可以尝试增加批次大小
                consecutive_errors = 0
                if current_batch_size < max_batch_size:
                    current_batch_size = min(current_batch_size + 1, max_batch_size)

            except Exception as e:
                if "Failed to allocate memory" in str(e) or "out of memory" in str(e).lower():
                    # 显存不足，减少批次大小
                    consecutive_errors += 1
                    current_batch_size = max(1, current_batch_size // 2)

                    logger.warning(f"Worker {worker_id}: 显存不足，将批次大小调整为 {current_batch_size}")
                    logger.info(f"当前内存状态: {get_memory_info()}")

                    # 强制垃圾回收
                    gc.collect()

                    # 将任务重新放回队列
                    inference_queue.put(batch_task)

                    # 如果连续错误太多，暂停一下
                    if consecutive_errors > 3:
                        time.sleep(1)

                else:
                    logger.error(f"Worker {worker_id}: 推理出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

            inference_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker {worker_id}: 推理工作线程出错: {e}")


def save_visualization(task: WindowTask, mask: np.ndarray, output_dir: str):
    """保存可视化结果"""
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)

    combined_path = os.path.join(mask_dir, f"{task.global_x}_{task.global_y}_combined.jpg")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_boundary = np.zeros_like(mask)
    cv2.drawContours(mask_boundary, contours, -1, (255, 0, 0), 2)

    img_a_patch_bgr = cv2.add(task.img_a_patch, cv2.cvtColor(mask_boundary, cv2.COLOR_GRAY2BGR))
    img_b_patch_bgr = cv2.add(task.img_b_patch, cv2.cvtColor(mask_boundary, cv2.COLOR_GRAY2BGR))
    mask_pil = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    combined_pil = Image.new('RGB', (task.img_a_patch.shape[1] + task.img_b_patch.shape[1] + mask_pil.width,
                                     max(task.img_a_patch.shape[0], task.img_b_patch.shape[0], mask_pil.height)))
    combined_pil.paste(Image.fromarray(img_a_patch_bgr), (0, 0))
    combined_pil.paste(Image.fromarray(img_b_patch_bgr), (task.img_a_patch.shape[1], 0))
    combined_pil.paste(mask_pil, (task.img_a_patch.shape[1] + task.img_b_patch.shape[1], 0))
    combined_pil.save(combined_path)


def async_slide_window_inference_optimized(img_a, img_b, session: ort.InferenceSession,
                                           embs: torch.Tensor, output_dir: str,
                                           args, global_coord_offset=None):
    """优化的异步滑窗批量推理"""
    logger = logging.getLogger()
    monitor = PerformanceMonitor(logger)

    # 记录初始内存状态
    logger.info(f"推理开始前内存状态: {get_memory_info()}")

    # 动态调整批次大小
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            available_memory_gb = (gpus[0].memoryTotal - gpus[0].memoryUsed) / 1024
            optimized_batch_size = adaptive_batch_size(args.batch_size, available_memory_gb)
            if optimized_batch_size != args.batch_size:
                logger.info(
                    f"根据可用显存 {available_memory_gb:.1f}GB，将批次大小从 {args.batch_size} 调整为 {optimized_batch_size}")
                args.batch_size = optimized_batch_size
    except:
        logger.warning("无法获取GPU信息，使用原始批次大小")

    # 创建任务队列 - 减小队列大小以减少内存占用
    max_queue_size = min(args.max_queue_size, 50)  # 限制最大队列大小
    crop_queue = queue.Queue(maxsize=max_queue_size)
    inference_queue = queue.Queue(maxsize=max_queue_size // args.batch_size + 1)
    result_queue = queue.Queue()

    # 创建停止事件
    stop_event = threading.Event()

    # 创建所有裁剪任务
    tasks = create_cropping_tasks(img_a, img_b, 512, 0, global_coord_offset)
    monitor.set_total_windows(len(tasks))

    # 计算批次数量
    total_batches = (len(tasks) + args.batch_size - 1) // args.batch_size
    monitor.set_total_batches(total_batches)

    logger.info(f"创建了 {len(tasks)} 个窗口任务，将分为 {total_batches} 个批次处理")
    logger.info(f"优化后批次大小: {args.batch_size}")

    # 启动工作线程 - 减少线程数以减少资源竞争
    crop_threads = []
    for i in range(min(args.crop_threads, 2)):  # 最多2个切片线程
        t = threading.Thread(target=crop_worker_optimized,
                             args=(crop_queue, inference_queue, monitor, stop_event,
                                   args.batch_size, max_queue_size // 2))
        t.start()
        crop_threads.append(t)

    inference_threads = []
    for i in range(min(args.inference_threads, 2)):  # 最多2个推理线程
        t = threading.Thread(target=inference_worker_optimized,
                             args=(inference_queue, result_queue, session, embs,
                                   output_dir, monitor, stop_event, args.batch_size, i))
        t.start()
        inference_threads.append(t)

    # 启动监控线程
    def monitor_progress():
        last_log_time = time.time()
        while not stop_event.is_set():
            current_time = time.time()
            if current_time - last_log_time >= args.log_interval:
                monitor.log_progress(crop_queue.qsize(), inference_queue.qsize())
                logger.info(f"内存状态: {get_memory_info()}")
                last_log_time = current_time
            time.sleep(1)

    monitor_thread = threading.Thread(target=monitor_progress)
    monitor_thread.start()

    # 分批添加任务，避免一次性占用大量内存
    def add_tasks_gradually():
        try:
            batch_add_size = 50  # 每次添加50个任务
            for i in range(0, len(tasks), batch_add_size):
                batch_tasks = tasks[i:i + batch_add_size]
                for task in batch_tasks:
                    crop_queue.put(task)
                time.sleep(0.01)  # 小延迟让其他线程有机会处理
        except Exception as e:
            logger.error(f"添加任务时出错: {e}")

    task_thread = threading.Thread(target=add_tasks_gradually)
    task_thread.start()

    # 等待所有任务完成
    task_thread.join()
    crop_queue.join()
    inference_queue.join()

    # 停止所有线程
    stop_event.set()

    # 发送毒丸信号
    for _ in range(len(crop_threads)):
        crop_queue.put(None)
    for _ in range(len(inference_threads)):
        inference_queue.put(None)

    # 等待线程结束
    for t in crop_threads:
        t.join()
    for t in inference_threads:
        t.join()
    monitor_thread.join()

    # 收集结果
    height, width, _ = img_a.shape
    result_mask = np.zeros((height, width), dtype=np.uint8)

    all_results = []
    while not result_queue.empty():
        batch_result = result_queue.get()
        all_results.extend(batch_result.results)

    logger.info(f"收集到 {len(all_results)} 个推理结果")

    for result in all_results:
        mask_h, mask_w = result.mask.shape
        result_mask[result.y:result.y + mask_h, result.x:result.x + mask_w] += result.mask

    # 归一化结果
    result_mask = (result_mask / np.clip(result_mask.max(), 1, None)).astype(np.uint8) * 255

    # 最终清理
    gc.collect()

    monitor.log_progress(0, 0)
    logger.info(f"推理完成后内存状态: {get_memory_info()}")
    logger.info("优化异步批量推理完成")

    return result_mask


def predict_onnx(args):
    """主推理函数 - 修复版本"""
    output_dir = get_output_dir(args.output_dir, 'infer_results')
    logger = setup_logging(log_dirs=output_dir)
    logger.info('异步推理开始 %s' % time.strftime("%c"))
    logger.info(f'输入: {args.input_dir}，输出: {output_dir}')

    device = args.device
    batch_size = args.batch_size
    logger.info(f'设备: {device}, 批次大小: {batch_size}')
    logger.info(f'线程配置 - 切片线程: {args.crop_threads}, 推理线程: {args.inference_threads}')

    # 构建词嵌入向量
    prompt = [args.prompt] if args.prompt is not None else None
    embs = build_embs(prompts=prompt, text_encoder_name="bert-base-uncased",
                      freeze_text_encoder=True, device=device, batch_size=batch_size)
    logger.info(f'嵌入向量形状: {embs.shape}')

    # 获取输入图像路径
    filename = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.tif')])
    a_image_path = os.path.join(args.input_dir, filename[0])
    b_image_path = os.path.join(args.input_dir, filename[1])

    if not os.path.exists(a_image_path) or not os.path.exists(b_image_path):
        raise FileNotFoundError("找不到合适的图像文件")

    # 加载ONNX模型
    logger.info(f"加载ONNX模型: {args.onnx_model}")
    session = ort.InferenceSession(
        args.onnx_model,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    logger.info(f"模型初始化完成，所在的设备：{session.get_providers()}")

    chunk_size = args.chunk_size
    use_chunking = chunk_size > 0

    if use_chunking:
        logger.info(f"使用分块推理，块大小: {chunk_size}")
        a_chunks, (a_height, a_width) = read_image_in_chunks(a_image_path, chunk_size)
        b_chunks, (b_height, b_width) = read_image_in_chunks(b_image_path, chunk_size)

        result_mask = np.zeros((a_height, a_width), dtype=np.uint8)
        count_mask = np.zeros_like(result_mask)

        total_chunks = len(a_chunks)
        logger.info(f"总共有 {total_chunks} 个块需要处理")

        for chunk_idx, ((x_a, y_a, img_a_patch), (x_b, y_b, img_b_patch)) in enumerate(zip(a_chunks, b_chunks)):
            logger.info(f"处理第 {chunk_idx + 1}/{total_chunks} 个块 (位置: {x_a}, {y_a})")

            if img_a_patch.shape != img_b_patch.shape:
                img_b_patch = cv2.resize(img_b_patch, (img_a_patch.shape[1], img_a_patch.shape[0]))

            mask = async_slide_window_inference_optimized(img_a_patch, img_b_patch, session, embs,
                                                          output_dir, args, (x_a, y_a))

            result_mask[y_a:y_a + mask.shape[0], x_a:x_a + mask.shape[1]] += mask
            count_mask[y_a:y_a + mask.shape[0], x_a:x_a + mask.shape[1]] += 1

        result_mask = (result_mask / np.clip(count_mask, 1, None)).astype(np.uint8)
    else:
        logger.info("使用全图推理")
        img_a = cv2.imread(a_image_path)
        img_b = cv2.imread(b_image_path)
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        result_mask = async_slide_window_inference_optimized(img_a, img_b, session, embs, output_dir, args)

    output_path = os.path.join(output_dir, "result_mask.tif")
    cv2.imwrite(output_path, result_mask)
    logger.info("✅ 异步图像推理完成！")


if __name__ == "__main__":
    args = get_args()
    predict_onnx(args)
