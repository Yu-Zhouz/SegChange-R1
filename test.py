# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: test.py
@Time    : 2025/4/24 下午5:24
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 测试建筑物变化检测模型
@Usage   :
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import Building
from engines import evaluate_model, load_model
import pprint
import time
from utils import get_args_config, get_output_dir, setup_logging


def main():
    cfg = get_args_config()
    output_dir = get_output_dir(cfg.test.save_dir, cfg.test.name)
    logger = setup_logging(cfg, output_dir)
    logger.info('Test Log %s' % time.strftime("%c"))
    logger.info('Running with config:')
    logger.info(pprint.pformat(cfg.__dict__))
    device = cfg.test.device

    model = load_model(cfg, cfg.test.weights_dir, cfg.test.device)

    # Build test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = Building(
        data_root=cfg.test.img_dirs,
        a_transform=transform,
        b_transform=transform,
        test=True,
        data_format=cfg.data_format,
    )

    test_loader = DataLoader(test_dataset, batch_size=cfg.test.batch_size, shuffle=False)

    # Run evaluation
    logger.info("Start testing...")
    metrics = evaluate_model(cfg, model, test_loader, device, output_dir)

    logger.info(
        "[Test] Precision: %.4f, Recall: %.4f, F1: %.4f, IoU: %.4f, OA: %.4f" % (
            metrics['precision'], metrics['recall'], metrics['f1'],
            metrics['iou'], metrics['oa']
        )
    )


if __name__ == '__main__':
    main()
