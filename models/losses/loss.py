import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: [B, 1, H, W] 未经过sigmoid的模型输出
        targets: [B, H, W] 真实掩码标签 (LongTensor)，值为0或1
        """
        # 对于单类输出，使用BCEWithLogitsLoss而不是CrossEntropyLoss
        loss = F.binary_cross_entropy_with_logits(inputs.squeeze(1), targets.float())
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: [B, 1, H, W] 经过sigmoid的模型输出
        targets: [B, H, W] 真实掩码标签 (LongTensor)，值为0或1
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.squeeze(1)

        # 将targets转换为浮点类型
        targets = targets.float()

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score
        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        inputs: [B, 1, H, W] 未经过sigmoid的模型输出
        targets: [B, H, W] 真实掩码标签 (LongTensor)，值为0或1
        """
        inputs = inputs.squeeze(1)
        targets = targets.float()

        pt = torch.sigmoid(inputs)
        pt = pt * targets + (1 - pt) * (1 - targets)
        focal_loss = -self.alpha * (1.0 - pt) ** self.gamma * torch.log(pt + 1e-6)
        return focal_loss.mean()


class BatchBalancedContrastiveLoss(nn.Module):
    """https://github.com/justchenhao/STANet"""
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        """
        inputs: [B, 1, H, W] 未经过sigmoid的模型输出
        targets: [B, H, W] 真实掩码标签 (LongTensor)，值为0或1
        """
        # 将inputs和targets调整为合适的形状
        inputs = inputs.squeeze(1)  # [B, H, W]
        targets = targets.float()    # [B, H, W]

        # 计算每个像素对的欧氏距离
        distance = torch.abs(inputs - targets)  # [B, H, W]

        # 计算正样本和负样本的数量
        positive_pairs = targets == 1
        negative_pairs = targets == 0
        num_positive = positive_pairs.sum()
        num_negative = negative_pairs.sum()

        # 计算正样本和负样本的权重
        if num_positive.item() > 0:
            w_positive = 1.0 / num_positive
        else:
            w_positive = 0.0
        if num_negative.item() > 0:
            w_negative = 1.0 / num_negative
        else:
            w_negative = 0.0

        # 计算损失
        loss_positive = w_positive * (distance * positive_pairs).sum()
        loss_negative = w_negative * (torch.clamp(self.margin - distance, min=0.0) * negative_pairs).sum()
        total_loss = loss_positive + loss_negative

        return total_loss


class TotalLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, weight_focal=1.0, weight_bcl=1.0):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_bcl = weight_bcl
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.bcl_loss = BatchBalancedContrastiveLoss()

    def forward(self, inputs, targets):
        """
        inputs: [B, 1, H, W] 未经过sigmoid的模型输出
        targets: [B, H, W] 真实掩码标签 (LongTensor)，值为0或1
        """
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        bcl = self.bcl_loss(inputs, targets)
        total_loss = self.weight_ce * ce + self.weight_dice * dice + self.weight_focal * focal + self.weight_bcl * bcl
        return total_loss


# 测试
if __name__ == '__main__':
    inputs = torch.randn(2, 1, 256, 256).cuda()  # 单类输出
    targets = torch.randint(0, 2, (2, 256, 256)).cuda()  # 值为0或1
    criterion = TotalLoss().cuda()
    loss = criterion(inputs, targets)
    print(loss)