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

# import pandas as pd
# import os
# import pprint
# import numpy as np
# import torch
# import datetime
# import logging
# import random
# import time
# import warnings
# from tensorboardX import SummaryWriter
# from dataloader import build_dataset
# from engines import train, evaluate
# from models import build_model, PostProcessor
# from utils import *
#
# warnings.filterwarnings('ignore')
#
#
# def main():
#     # cfg = load_config("./configs/config.yaml")
#     cfg = get_args_config()
#     output_dir = get_output_dir(cfg.output_dir, cfg.name)
#     logger = setup_logging(cfg, output_dir)
#     logger.info('Train Log %s' % time.strftime("%c"))
#     env_info = get_environment_info()
#     logger.info(env_info)
#     # backup the arguments
#     logger.info('Running with config:')
#     logger.info(pprint.pformat(cfg.__dict__))  # 修改这里以递归打印 Config
#     device = cfg.device
#     # fix the seed for reproducibility
#     seed = cfg.seed + get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     logger.info('------------------------ model params ------------------------')
#     model, criterion = build_model(cfg, training=True)
#     # move to GPU
#     model.to(device)
#     if isinstance(criterion, tuple) and len(criterion) == 2:
#         for loss in criterion:
#             loss.to(device)
#     else:
#         criterion.to(device)
#
#     postprocessor = PostProcessor(min_area=2500, max_p_a_ratio=10, min_convexity=0.8)
#
#     model_without_ddp = model
#     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.info('number of params: %d', n_parameters)
#     # 对模型的不同部分使用不同的优化参数
#     param_dicts = [{
#         "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad],
#         "lr": cfg.training.lr
#     },
#         {
#             "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
#             "lr": cfg.training.lr_backbone,
#         },
#     ]
#
#     optimizer = torch.optim.AdamW(param_dicts, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
#     lr_scheduler = None
#     # 配置学习率调度器
#     if cfg.training.scheduler == 'step':
#         lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.lr_drop, gamma=0.1)
#     elif cfg.training.scheduler == 'plateau':
#         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
#
#     # 打印优化器详细信息
#     optimizer_info = f"optimizer: Adam(lr={cfg.training.lr})"
#     optimizer_info += " with parameter groups "
#     for i, param_group in enumerate(optimizer.param_groups):
#         optimizer_info += f"{len(param_group['params'])} weight(decay={param_group['weight_decay']}), "
#     optimizer_info = optimizer_info.rstrip(', ')
#     logger.info(optimizer_info)
#
#     # 用于训练的采样器
#     dataloader_train, dataloader_val = build_dataset(cfg=cfg)
#
#     # 如果存在，则恢复权重和训练状态
#     if cfg.resume:
#         logger.info('------------------------ Continue training ------------------------')
#         logging.warning(f"loading from {cfg.resume}")
#         checkpoint = torch.load(cfg.resume, map_location='cpu')
#         model_without_ddp.load_state_dict(checkpoint['model'])
#         if not cfg.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#             cfg.start_epoch = checkpoint['epoch'] + 1
#
#     logger.info("------------------------ Start training ------------------------")
#     start_time = time.time()
#     # 保存训练期间的指标
#     precision_list = []
#     recall_list = []
#     f1_list = []
#     iou_list = []
#     accuracy_list = []
#     # 创建tensorboard
#     tensorboard_dir = os.path.join(str(output_dir), 'tensorboard')
#     os.makedirs(tensorboard_dir, exist_ok=True)
#     writer = SummaryWriter(tensorboard_dir)
#
#     # 初始化 CSV 文件
#     csv_file_path = os.path.join(str(output_dir), 'result.csv')
#     # 初始化DataFrame用于缓存结果
#     results_df = pd.DataFrame(columns=[
#         'epoch', 'train_loss', 'train_oa',
#         'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_iou', 'val_oa'
#     ])
#
#     step = 0
#     # 开始训练
#     for epoch in range(cfg.training.start_epoch, cfg.training.epochs):
#         t1 = time.time()
#
#         stat = train(cfg, model, criterion, dataloader_train, optimizer, device, epoch)
#         time.sleep(1)  # 避免tensorboard卡顿
#         t2 = time.time()
#         # 记录训练损失和OA
#         if writer is not None:
#             logger.info("[ep %d][lr %.7f][%.2fs] loss: %.4f, oa: %.4f", epoch, optimizer.param_groups[0]['lr'], t2 - t1,
#                         stat['loss'], stat['oa'])
#             writer.add_scalar('loss/loss', stat['loss'], epoch)
#             writer.add_scalar('metric/o_a', stat['oa'], epoch)
#
#         # 在训练完成后，更新训练指标
#         results_df.loc[epoch] = {'epoch': epoch, 'train_loss': stat['loss'], 'train_oa': stat['oa'],
#                                  'val_loss': '', 'val_precision': '', 'val_recall': '', 'val_f1': '', 'val_iou': '', 'val_oa': ''
#                                  }
#
#         # 调整学习率
#         if cfg.training.scheduler == 'step':
#             lr_scheduler.step()
#         # 每隔一纪元保存最新权重
#         ckpt_dir = os.path.join(str(output_dir), 'checkpoints')
#         os.makedirs(ckpt_dir, exist_ok=True)
#         checkpoint_latest_path = os.path.join(ckpt_dir, 'latest.pth')
#         torch.save({
#             'epoch': epoch,
#             'step': step,
#             'model': model_without_ddp.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'lr_scheduler': lr_scheduler.state_dict(),
#             'loss': stat['loss'],
#         }, checkpoint_latest_path)
#         # 开始评估
#         if epoch % cfg.training.eval_freq == 0 and epoch >= cfg.training.start_eval:
#             t1 = time.time()
#             # 假设 evaluate 函数返回 precision、recall、f1、iou、accuracy
#             metrics = evaluate(cfg, model, criterion, postprocessor, dataloader_val, device, epoch)
#             t2 = time.time()
#             if cfg.training.scheduler == 'plateau':
#                 lr_scheduler.step(metrics['loss'])  # 根据验证损失调整学习率
#
#             precision_list.append(metrics['precision'])
#             recall_list.append(metrics['recall'])
#             f1_list.append(metrics['f1'])
#             iou_list.append(metrics['iou'])
#             accuracy_list.append(metrics['oa'])
#             fps = len(dataloader_val.dataset) / (t2 - t1)
#             # document the results of the assessment
#             logger.info(
#                 "[ep %d][%.3fs][%.5ffps] loss: %.4f, precision: %.4f, recall: %.4f, f1: %.4f, iou: %.4f, oa: %.4f ---- @best f1: %.4f, @best iou: %.4f" % \
#                 (epoch, t2 - t1, fps, metrics['loss'], metrics['precision'], metrics['recall'], metrics['f1'],
#                  metrics['iou'], metrics['oa'], np.max(f1_list), np.max(iou_list)))
#
#             # recored the evaluation results
#             if writer is not None:
#                 writer.add_scalar('metric/val_loss', metrics['loss'], step)
#                 writer.add_scalar('metric/precision', metrics['precision'], step)
#                 writer.add_scalar('metric/recall', metrics['recall'], step)
#                 writer.add_scalar('metric/f1', metrics['f1'], step)
#                 writer.add_scalar('metric/iou', metrics['iou'], step)
#                 writer.add_scalar('metric/oa', metrics['oa'], step)
#                 step += 1
#
#             # 更新验证指标
#             results_df.loc[epoch, [
#                 'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_iou', 'val_oa'
#             ]] = [metrics['loss'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['iou'],
#                   metrics['oa']]
#
#             # 从一开始就保存最好的模型
#             if metrics['f1'] == np.max(f1_list):
#                 checkpoint_best_path = os.path.join(ckpt_dir, 'best_f1.pth')
#                 torch.save({
#                     'epoch': epoch,
#                     'step': step,
#                     'model': model_without_ddp.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'lr_scheduler': lr_scheduler.state_dict(),
#                     'loss': stat['loss'],
#                 }, checkpoint_best_path)
#             if metrics['iou'] == np.max(iou_list):
#                 checkpoint_best_iou_path = os.path.join(ckpt_dir, 'best_iou.pth')
#                 torch.save({
#                     'epoch': epoch,
#                     'step': step,
#                     'model': model_without_ddp.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'lr_scheduler': lr_scheduler.state_dict(),
#                     'loss': stat['loss'],
#                 }, checkpoint_best_iou_path)
#         # 每个 epoch 后都保存一次 CSV
#         results_df.to_csv(csv_file_path, index=False)
#
#     # 培训总时间
#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     logger.info('Summary of results')
#     logger.info(env_info)
#     logger.info('Training time {}'.format(total_time_str))
#     # 打印最好的指标
#     logger.info("Best F1: %.4f, Best IoU: %.4f, Best Accuracy: %.4f" % (
#     np.max(f1_list), np.max(iou_list), np.max(accuracy_list)))
#     logger.info('Results saved to {}'.format(cfg.output_dir))


from utils import get_args_config
from engines import TrainingEngine


def main():
    cfg = get_args_config()
    # 创建训练引擎并运行
    trainer = TrainingEngine(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
