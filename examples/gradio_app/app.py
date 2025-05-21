# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: app.py
@Time    : 2025/5/20 下午3:23
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : Gradio 创建交互式接口
@Usage   :
"""
import os
import socket
import sys
import gradio as gr
import cv2
import torch

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from engines import load_model, preprocess_image, postprocess_mask
from utils import load_config


# 动态获取主机的 IP 地址
def get_host_ip():
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except:
        host_ip = "127.0.0.1"  # 默认回退到本地地址
    return host_ip


# 设置 Gradio 的服务器参数
os.environ["GRADIO_SERVER_NAME"] = get_host_ip()
os.environ["GRADIO_SERVER_PORT"] = "7860"
os.environ["ORIGINS"] = "*"
# 30110

# 加载参数和模型
cfg = load_config('/sxs/zhoufei/SegChange-R1/configs/config.yaml')
cfg.infer.weights_dir = '/sxs/zhoufei/SegChange-R1/work_dirs/train_0/checkpoints/best_iou.pth'
model = load_model(cfg)


def process_images(img_a, img_b, prompt, threshold=0.5):
    """
    处理输入的两张图像和文本提示，返回变化掩码和标记后的图像
    Args:
        img_a: 第一张图像 (numpy array)
        img_b: 第二张图像 (numpy array)
        prompt: 文本提示
        threshold: 二值化阈值
    Returns:
        img_a_marked: 标记变化区域的第一张图像
        img_b_marked: 标记变化区域的第二张图像
        mask: 生成的变化掩码
    """
    # 预处理图像
    img_a_tensor, img_b_tensor = preprocess_image(img_a, img_b, cfg.device)

    # 推理
    with torch.no_grad():
        outputs = model(img_a_tensor, img_b_tensor, [prompt])
        preds = (torch.sigmoid(outputs) > threshold).float().squeeze(1).cpu().numpy()

    # 提取掩码
    mask = (preds[0] * 255).astype('uint8')
    # 后处理掩码
    mask = postprocess_mask(mask)

    # 在图像上绘制变化区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_a_marked = cv2.drawContours(img_a, contours, -1, (0, 255, 0), 2)
    img_b_marked = cv2.drawContours(img_b, contours, -1, (0, 255, 0), 2)

    return img_a_marked, img_b_marked, mask


# 创建 Gradio 接口
demo = gr.Interface(
    fn=process_images,
    inputs=[
        gr.Image(type="numpy"),
        gr.Image(type="numpy"),
        gr.Textbox(label="Prompt"),
        gr.Slider(0, 1, value=0.5, label="Threshold")
    ],
    outputs=[
        gr.Image(label="Marked Image A"),
        gr.Image(label="Marked Image B"),
        gr.Image(label="Binary Mask", image_mode="L")
    ],
    title="变化图斑监测",
    description="上传两张图像和一个文本提示，查看变化区域"
)

# 启动 Gradio 应用
demo.launch(
    server_name=os.environ["GRADIO_SERVER_NAME"],
    server_port=int(os.environ["GRADIO_SERVER_PORT"]),
    show_error=True,
)

# 自定义打印信息
custom_message = "* Port Forwarding Post URL: "
port_forwarding_url = "http://172.16.15.10:30110"  # 你的自定义 URL
print(custom_message + port_forwarding_url)