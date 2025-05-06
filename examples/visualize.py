# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: visualize.py
@Time    : 2025/4/29 下午3:40
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 特征图可视化工具（支持任意层 hook + 图像加载）
@Usage   : 查看模型结构的命令：python -c "from models.segchange import ChangeModel; from utils import load_config; model = ChangeModel(load_config('./configs/config.yaml')); print(model)"
"""
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from models.segchange import ChangeModel
from utils import load_config
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot前设置
import matplotlib.pyplot as plt



def load_model(model_path, device='cuda'):
    """从文件中加载完整模型"""
    cfg = load_config("../configs/config.yaml")
    model = ChangeModel(cfg)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'])

    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def register_hooks(model, layer_names):
    """
    在模型的指定模块上注册钩子，捕获其输出
    Args:
        model: PyTorch 模型
        layer_names: list of str，要可视化的层名（如 'kernel_gen_conv'）
    Returns:
        activations: dict，存储各层激活值
    """
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, (tuple, list)):
                # 提取所有元素的张量
                activations[name] = [o.detach().cpu() for o in output]
            else:
                activations[name] = output.detach().cpu()
        return hook

    # 遍历所有模块，找到匹配的层名
    for name, module in model.named_modules():
        if name in layer_names:
            print(f"Registering hook on layer: {name}")
            module.register_forward_hook(get_activation(name))

    return activations


def visualize_fm(feature_map, save_path=None, n_cols=8):
    """
    可视化 feature map
    Args:
        feature_map: Tensor [C, H, W]
        save_path: str or None，保存路径
        n_cols: int，每行显示多少个 channel
    """
    feature_map = feature_map.detach().cpu()
    C, H, W = feature_map.shape
    n_rows = int(np.ceil(C / n_cols))

    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i in range(min(C, n_cols * n_rows)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(feature_map[i], cmap="jet")
        plt.colorbar(shrink=0.5)
        plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature map saved to {save_path}")

    plt.show()
    plt.close()


def load_image_pair(image1_path, image2_path):
    """
    加载双时相图像对，并应用图像增强和归一化

    Args:
        image1_path: str，第一张图像路径
        image2_path: str，第二张图像路径

    Returns:
        img1: Tensor [C, H, W]，增强并归一化处理后的图像张量
        img2: Tensor [C, H, W]，增强并归一化处理后的图像张量
    """
    # 定义图像增强和归一化变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载图像
    img1 = cv2.imread(image1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = transform(img1)

    img2 = cv2.imread(image2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = transform(img2)

    return img1, img2


def nested_tensor_from_tensor_list(tensor_list):
    """
    将一组 tensor 转换为 nested tensor

    Args:
        tensor_list: list of Tensor，输入 tensor 列表

    Returns:
        nested_tensor: nested tensor
    """
    # 这里可以实现 nested tensor 的转换逻辑
    # 目前直接返回输入列表
    return tensor_list


def main():
    # 设置路径
    model_path = "../work_dirs/train/checkpoints/best_iou.pth"
    image1_path = "./images/1_a.png"
    image2_path = "./images/1_b.png"
    desc_text = ["Detection of building changes"]  # 示例文本
    save_dir = "./output_fm"
    os.makedirs(save_dir, exist_ok=True)

    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model = load_model(model_path, device=device)

    # 要可视化的层名称（根据你的模型结构填写）
    layers_to_visualize = [
        # feature_diff 模块中的层
        'feature_diff.diff_conv.0.0',  # 第一个 diff_conv 中的卷积层
        'feature_diff.attention.0.conv',  # 第一个 attention 中的卷积层

        # fpn 模块中的层
        'fpn'
        
        # scale_heads 模块中的层
        'mask_generator',
        'mask_generator.scale_heads.0.0',  # 第一个 scale_head 中的卷积层
        'mask_generator.scale_heads.1.0',  # 第二个 scale_head 中的卷积层
        'mask_generator.scale_heads.2.0',  # 第三个 scale_head 中的卷积层
        'mask_generator.scale_heads.3.0',  # 第四个 scale_head 中的卷积层
    ]

    # 注册钩子
    activations = register_hooks(model, layers_to_visualize)

    # 加载图像对
    img_a, img_b = load_image_pair(image1_path, image2_path)

    # 前向推理一次以触发钩子
    with torch.no_grad():
        logits = model(img_a.unsqueeze(0), img_b.unsqueeze(0), desc_text)  # 增加批次维度
        print("Model output (logits):", logits)

    # 对每个层可视化 feature map 并保存
    for layer_name in layers_to_visualize:
        if layer_name not in activations:
            print(f"No activation found for layer: {layer_name}")
            continue

        print(f"Visualizing layer: {layer_name}")
        fm = activations[layer_name]

        # 统一处理所有输出类型
        if isinstance(fm, list):  # 处理列表输出
            for level_idx, tensor in enumerate(fm):
                batch_size = tensor.size(0)
                for img_idx in range(batch_size):
                    single_fm = tensor[img_idx]
                    save_path = os.path.join(save_dir, f"{layer_name}_level{level_idx}_img{img_idx}.png")
                    visualize_fm(single_fm, save_path=save_path)
        else:  # 处理单张量输出
            batch_size = fm.size(0)
            for img_idx in range(batch_size):
                single_fm = fm[img_idx]
                save_path = os.path.join(save_dir, f"{layer_name}_img{img_idx}.png")
                visualize_fm(single_fm, save_path=save_path)


if __name__ == '__main__':
    main()
