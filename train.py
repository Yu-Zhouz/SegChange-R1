# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: train.py
@Time    : 2025/4/18 上午10:24
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 建筑物变化检测训练
@Usage   :
"""
from utils import get_args_config
from engines import TrainingEngine


def main():
    cfg = get_args_config()
    # 创建训练引擎并运行
    trainer = TrainingEngine(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
