import os
import sys

import torch
import torch.onnx

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from models import build_model, build_embs
from utils import get_args_config


def export_to_onnx(cfg):
    """
    导出模型到 ONNX 格式
    Args:
        cfg: 配置文件
    """
    # 设备选择 (CPU or GPU)
    device = cfg.device
    prompt = [cfg.prompt]
    embs = build_embs(prompts=prompt, text_encoder_name=cfg.model.text_encoder_name,
                              freeze_text_encoder=cfg.model.freeze_text_encoder, device=device)
    print(f'desc_embs shape:{embs.shape}')

    # 构建模型结构
    model = build_model(cfg, training=False)
    model.to(device)

    # 加载权重并过滤掉多余的 key
    print(f'加载模型权重所在地址：{cfg.infer.weights_dir}')
    if cfg.infer.weights_dir is not None:
        checkpoint = torch.load(cfg.infer.weights_dir, map_location=device)

        # 创建一个只保留当前模型中存在的键的新 state_dict
        filtered_state_dict = {
            k: v for k, v in checkpoint['model'].items()
            if k in model.state_dict() and v.shape == model.state_dict()[k].shape
        }

        # 打印未匹配的 key（调试用）
        missing_keys = set(checkpoint['model'].keys()) - set(filtered_state_dict.keys())
        if missing_keys:
            print("⚠️ 以下权重未被加载（可能因为模型结构变化）：")
            for key in missing_keys:
                print(f"   {key}")

        # 加载过滤后的权重
        model.load_state_dict(filtered_state_dict, strict=False)

    model.eval()

    # 构造输入数据
    image1 = torch.randn(1, 3, 512, 512).to(device)  # 输入图像A
    image2 = torch.randn(1, 3, 512, 512).to(device)  # 输入图像B

    # 模型输出路径
    output_path = os.path.join(cfg.onnx_weights, "segchange.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 导出 TorchScript 模型（可选）
    # traced_model = torch.jit.script(model)
    traced_model = torch.jit.trace(model, (image1, image2, embs))
    torch.jit.save(traced_model, output_path.replace(".onnx", ".pt"))
    print(f"✅ TorchScript 模型已保存：{output_path.replace('.onnx', '.pt')}")

    # 加载 TorchScript 模型用于 ONNX 导出
    scripted_model = torch.jit.load(output_path.replace(".onnx", ".pt"))

    # ONNX 导出配置
    input_names = ["image1", "image2", "embs"]
    output_names = ["mask"]

    # 指定动态维度
    dynamic_axes = {
        'image1': {0: 'batch_size'},
        'image2': {0: 'batch_size'},
        'embs': {0: 'batch_size', 1: 'seq_len'},  # 指定 embs 的动态维度 {0: 'batch_size', 1: 'seq_len',  2: 'lang_dim'}
        'mask': {0: 'batch_size'}
    }

    # 执行 ONNX 导出
    torch.onnx.export(
        scripted_model,
        (image1, image2, embs),
        output_path,
        optimize_for_inference=True,  # 启用推理优化
        export_params=True,           # 存储训练参数
        opset_version=16,             # ONNX 算子集版本
        do_constant_folding=True,     # 优化常量
        input_names=input_names,      # 输入名称
        output_names=output_names,    # 输出名称
        dynamic_axes=dynamic_axes     # 动态维度
    )

    print(f"✅ 模型已成功导出为 ONNX 格式：{output_path}")


if __name__ == "__main__":
    config = get_args_config()
    export_to_onnx(config)