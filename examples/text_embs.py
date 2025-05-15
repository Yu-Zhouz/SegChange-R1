# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: text_embs.py
@Time    : 2025/5/12 下午3:37
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 预处理提示词并保存描述嵌入
@Usage   :
"""
import argparse
import os
import sys
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models import TextEncoderLLM, TextEncoderBert
from utils import load_config


def get_args_config():
    parser = argparse.ArgumentParser('SegChange')
    parser.add_argument('-c', '--config', type=str, required=True, help='The path of config file')
    args = parser.parse_args()
    if args.config is not None:
        cfg = load_config(args.config)
    else:
        raise ValueError('Please specify the config file')
    return cfg


def emb_model():
    cfg = get_args_config()
    prompts = cfg.additional_text if hasattr(cfg, 'additional_text') else 'Buildings with changes'
    if cfg.model.text_encoder_name == "microsoft/phi-1_5":
        model = TextEncoderLLM(model_name=cfg.model.text_encoder_name, device=cfg.device,
                               freeze_text_encoder=cfg.model.freeze_text_encoder)
    elif cfg.model.text_encoder_name == "bert-base-uncased":
        model = TextEncoderBert(model_name=cfg.model.text_encoder_name, device=cfg.device,
                                freeze_text_encoder=cfg.model.freeze_text_encoder)
    else:
        raise NotImplementedError(f"Unsupported text encoder name: {cfg.model.text_encoder_name}")
    desc_embs, _ = model(prompts)
    print(desc_embs.shape)
    os.makedirs(os.path.dirname(cfg.model.desc_embs), exist_ok=True)
    torch.save(desc_embs, cfg.model.desc_embs)

    print("Description embeddings saved successfully!")



if __name__ == '__main__':
    emb_model()