# -*- coding: utf-8 -*-
"""
@Project : SegChange-R2
@FileName: check_dict_usage.py
@Time    : 2025/5/23 下午8:09
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 检查模型中是否使用了 dict 结构（可能导致 ONNX 导出失败）
"""

import os
import sys
import torch
import torch.fx

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from models import build_model, build_embs
from utils import get_args_config


class DictFinder(torch.fx.Interpreter):
    def call_function(self, target, args=None, kwargs=None):
        if 'dict' in str(target):
            print("⚠️ 可能使用了 dict 相关操作：", target)
        return super().call_function(target, args, kwargs)


def check_model_for_dict():
    # 加载配置
    cfg = get_args_config()

    # 构建模型
    model = build_model(cfg, training=False).eval()

    # 构造输入数据
    device = cfg.device
    pre = torch.randn(1, 3, 512, 512).to(device)  # 输入图像A
    late = torch.randn(1, 3, 512, 512).to(device)  # 输入图像B
    lang_dim = 2048 if cfg.model.text_encoder_name == 'microsoft/phi-1_5' else 768
    if cfg.prompt is not None:
        # 构建词嵌入向量
        prompt = [cfg.prompt]
        embs = build_embs(prompts=prompt, text_encoder_name=cfg.model.text_encoder_name,
                          freeze_text_encoder=cfg.model.freeze_text_encoder, device=device)
    else:
        lang_dim = 2048 if cfg.model.text_encoder_name == 'microsoft/phi-1_5' else 768
        embs = torch.zeros((1, 4, lang_dim), device=device)

    # 使用 FX symbolic trace 分析模型结构
    try:
        graph_module: torch.fx.GraphModule = torch.fx.symbolic_trace(model)
    except Exception as e:
        print("❌ symbolic_trace 失败，请检查模型 forward 是否可追踪")
        raise e

    interpreter = DictFinder(graph_module)
    interpreter.run(pre, late, embs)

    input_args = model.forward.__code__.co_varnames[1:]  # 忽略 self
    required_inputs = {'image1', 'image2', 'embs'}

    missing = required_inputs - set(input_args)
    assert not missing, f"⚠️ 模型 forward 缺少必要输入参数: {missing}"
    print("✅ 模型 forward 包含必要输入参数:", required_inputs)
    # 尝试直接 script 模型
    try:
        scripted_model = torch.jit.script(model)
    except Exception as e:
        print("❌ TorchScript 失败，模型中可能存在不支持结构")
        raise e


if __name__ == "__main__":
    check_model_for_dict()
