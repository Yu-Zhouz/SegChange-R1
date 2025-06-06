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

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models import build_embs
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



if __name__ == '__main__':
    cfg = get_args_config()
    desc_embs = build_embs(
        prompts=cfg.prompt,
        text_encoder_name=cfg.model.text_encoder_name,
        freeze_text_encoder=cfg.model.freeze_text_encoder,
        device=cfg.device,
        batch_size=cfg.training.batch_size)
    print(f'✅ desc_embs shape:{desc_embs.shape}')
