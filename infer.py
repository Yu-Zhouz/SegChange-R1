# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: infer.py
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
    # names = ['BL1_5', 'BL2_5', 'BL3_5', 'BL4_5', 'BL5_5', 'HC_5', 'ZK_5']
    names = ['HY_10_2']
    # 遍历名称列表，动态生成 input_dir 并调用 predict
    for name in names:
        cfg.infer.name = name
        cfg.infer.input_dir = f'/sxs/zhoufei/SegChange-R2/data/Base1/{name}'  # 修改 input_dir 路径
        predict(cfg)


if __name__ == "__main__":
    main()
