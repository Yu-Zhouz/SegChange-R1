import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, targets):
        """
        pred: [B, 1, H, W] 未激活的logits
        targets: [B, H, W] 真实标签 (0/1)
        """
        loss = F.binary_cross_entropy_with_logits(pred.squeeze(1), targets.float())
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, targets):
        """
        pred: [B, K, H, W] 未激活的logits
        targets: [B, H, W] 真实标签 (0~K-1)
        """
        loss = F.cross_entropy(pred, targets.long())
        return loss



class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=1):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, pred, targets):
        """
        pred: [B, K, H, W] 未激活的logits
        targets: [B, H, W] 真实标签 (0~K-1)
        """
        if self.num_classes == 1:
            # 二分类：使用Sigmoid
            pred = torch.sigmoid(pred).squeeze(1)
            targets = targets.float()
            intersection = (pred * targets).sum()
            union = pred.sum() + targets.sum()
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        else:
            # 多分类：使用Softmax
            pred = F.softmax(pred, dim=1)
            targets = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            intersection = (pred * targets).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_score = dice_score.mean()

        dice_loss = 1.0 - dice_score
        return dice_loss



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred, targets):
        """
        pred: [B, K, H, W] 未激活的logits
        targets: [B, H, W] 真实标签 (0~K-1)
        """
        if self.num_classes == 1:
            # 二分类
            pred = pred.squeeze(1)
            pt = torch.sigmoid(pred)
            pt = pt * targets + (1 - pt) * (1 - targets)
            focal_loss = -self.alpha * (1.0 - pt) ** self.gamma * torch.log(pt + 1e-6)
        else:
            # 多分类
            logpt = F.log_softmax(pred, dim=1)
            pt = torch.exp(logpt)
            targets = targets.unsqueeze(1)
            logpt = torch.gather(logpt, 1, targets).squeeze(1)
            pt = torch.gather(pt, 1, targets).squeeze(1)
            focal_loss = -self.alpha * (1.0 - pt) ** self.gamma * logpt

        return focal_loss.mean()


class BatchBalancedContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0, num_classes=1):
        super().__init__()
        self.margin = margin
        self.num_classes = num_classes

    def forward(self, pred, targets):
        """
        pred: [B, 1, H, W] 未经过sigmoid的模型输出
        targets: [B, H, W] 真实掩码标签 (LongTensor)，值为0或1
        """
        if self.num_classes == 1:
            # 二分类：保持原逻辑
            # 将inputs和targets调整为合适的形状
            pred = pred.squeeze(1)  # [B, H, W]
            targets = targets.float()    # [B, H, W]

            # 计算每个像素对的欧氏距离
            distance = torch.abs(pred - targets)  # [B, H, W]

            # 计算正样本和负样本的数量
            positive_pairs = targets == 1
            negative_pairs = targets == 0
            num_positive = positive_pairs.sum()
            num_negative = negative_pairs.sum()

            w_positive = 1.0 / num_positive if num_positive > 0 else 0.0
            w_negative = 1.0 / num_negative if num_negative > 0 else 0.0

            loss_positive = w_positive * (distance * positive_pairs).sum()
            loss_negative = w_negative * (torch.clamp(self.margin - distance, min=0.0) * negative_pairs).sum()
            total_loss = loss_positive + loss_negative

            return total_loss
        else:
            loss = 0.0

            # 遍历所有类别组合
            for i in range(self.num_classes):
                for j in range(i + 1, self.num_classes):
                    # 提取类别i和j的样本
                    mask_i = (targets == i)
                    mask_j = (targets == j)
                    if not mask_i.any() or not mask_j.any():
                        continue  # 跳过无样本的类别

                    # 计算i与j的样本对距离
                    feat_i = pred[mask_i]
                    feat_j = pred[mask_j]
                    distance = torch.cdist(feat_i, feat_j)  # 计算两两距离

                    # 正对（同类）：i内部的样本对
                    pos_pairs = distance.diagonal()
                    loss += (pos_pairs ** 2).mean()

                    # 负对（异类）：i与j的样本对
                    neg_pairs = distance
                    loss += torch.clamp(self.margin - neg_pairs, min=0.0).mean()

            # 归一化损失
            loss /= (self.num_classes * (self.num_classes - 1) / 2)
            return loss


class TotalLoss(nn.Module):
    def __init__(self, num_classes=1, weight_ce=1.0, weight_dice=1.0, weight_focal=1.0, alpha=0.25, gamma=2.0, weight_bcl=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_bcl = weight_bcl

        if num_classes == 1:
            self.cls_loss = BCEWithLogitsLoss()
        else:
            self.cls_loss = CrossEntropyLoss()

        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes)
        self.bcl_loss = BatchBalancedContrastiveLoss(num_classes=num_classes)

    def forward(self, pred, targets):
        """
        pred: [B, K, H, W] 未激活的logits
        targets: [B, H, W] 真实标签 (0~K-1)
        """
        ce = self.cls_loss(pred, targets)
        dice = self.dice_loss(pred, targets)
        focal = self.focal_loss(pred, targets)
        bcl = self.bcl_loss(pred, targets)

        total_loss = (
            self.weight_ce * ce +
            self.weight_dice * dice +
            self.weight_focal * focal +
            self.weight_bcl * bcl
        )
        return total_loss


# 测试
if __name__ == '__main__':
    pred = torch.randn(2, 1, 256, 256).cuda()  # 单类输出
    targets = torch.randint(0, 2, (2, 256, 256)).cuda()  # 值为0或1
    criterion = TotalLoss().cuda()
    loss = criterion(pred, targets)
    print(loss)