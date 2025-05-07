# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: engines.py
@Time    : 2025/4/21 上午9:48
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score


def train(cfg, model, criterion, dataloader, optimizer, device, epoch):
    model.train()
    criterion.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_pixels = 0

    with tqdm(dataloader, desc=f'Epoch {epoch} [Training]') as pbar:
        for images_a, images_b, prompt, labels in pbar:
            images_a = images_a.to(device)
            images_b = images_b.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images_a, images_b, prompt)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images_a.size(0)
            total_samples += images_a.size(0)

            # Calculate accuracy
            if cfg.model.num_classes == 1:
                preds = torch.round(torch.sigmoid(outputs)).squeeze(1)
            else:
                preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_pixels += labels.numel()

            pbar.set_postfix({
                'loss': total_loss / total_samples,
                'oa': total_correct / total_pixels
            })

    epoch_loss = total_loss / total_samples
    epoch_oa = total_correct / total_pixels

    stat = {
        'loss': epoch_loss,
        'oa': epoch_oa
    }

    return stat, {'loss': epoch_loss, 'oa': epoch_oa}


def evaluate(cfg, model, criterion, dataloader, device, epoch):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(), tqdm(dataloader, desc=f'Epoch {epoch} [Validation]') as pbar:
        for images_a, images_b, prompt,  labels in pbar:
            images_a = images_a.to(device)
            images_b = images_b.to(device)
            labels = labels.to(device)

            outputs = model(images_a, images_b, prompt)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images_a.size(0)
            total_samples += images_a.size(0)

            # Store predictions and labels
            if cfg.model.num_classes == 1:
                preds = torch.round(torch.sigmoid(outputs)).squeeze(1).cpu().numpy()
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

            pbar.set_postfix({
                'loss': total_loss / total_samples
            })

    # Calculate evaluation metrics
    if cfg.model.num_classes == 1:
        # 二分类指标
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        iou = jaccard_score(all_labels, all_preds)
        oa = accuracy_score(all_labels, all_preds)
    else:
        # 多分类指标（使用macro平均）
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        iou = jaccard_score(all_labels, all_preds, average='macro')
        oa = accuracy_score(all_labels, all_preds)

    return precision, recall, f1, iou, oa