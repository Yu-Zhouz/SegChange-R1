# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: inference.py
@Time    : 2025/4/24 下午5:24
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 测试建筑物变化检测模型
@Usage   :
"""
from engines import predict
from utils import get_args_config


def main():
    cfg = get_args_config()
    # predict(cfg)
    names = ['BL1_100', 'HC1_100', 'HY1_100', 'DYW1_100']
    # 遍历名称列表，动态生成 input_dir 并调用 predict
    for name in names:
        cfg.infer.name = name
        cfg.input_dir = f'./data/{name}'  # 修改 input_dir 路径
        predict(cfg)


if __name__ == "__main__":
    main()
