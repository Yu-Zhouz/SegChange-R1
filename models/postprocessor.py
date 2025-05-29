# -*- coding: utf-8 -*-
"""
@Project : SegChange-R2
@FileName: postprocessor.py
@Time    : 2025/5/26 下午12:32
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : Mask 后处理
@Usage   : 
"""
import cv2
import numpy as np


class PostProcessor:
    def __init__(self, min_area=2500, max_p_a_ratio=10, min_convexity=0.8):
        self.min_area = min_area
        self.max_p_a_ratio = max_p_a_ratio
        self.min_convexity = min_convexity

    def __call__(self, mask):
        """
        统一入口：根据输入类型自动调用 forward 或 postprocess_mask

        :param mask: 可以是 float [0,1] 或 uint8 [0,255]
        :return: 处理后的 mask + 统计信息，格式与输入一致
        """
        if isinstance(mask, np.ndarray):
            if mask.dtype == np.float32 or mask.dtype == np.float64:
                return self.forward(mask)
            elif mask.dtype == np.uint8:
                return self.postprocess_mask(mask)
            else:
                raise ValueError(f"Unsupported dtype: {mask.dtype}")
        else:
            raise TypeError(f"Unsupported input type: {type(mask)}")

    def forward(self, preds: np.ndarray):
        """
        输入：float 类型的二值化掩码 [0,1]，形状为 (H, W) 或 (B, H, W)
        输出：处理后的 float 掩码 [0,1]

        :param preds: 模型输出的预测结果（numpy array）
        :return: processed_preds, stats
        """
        if preds.ndim == 3:
            batch_size, h, w = preds.shape
            processed_preds = np.zeros_like(preds)
            stats_list = []

            for i in range(batch_size):
                binary_mask = (preds[i] > 0.5).astype(np.uint8) * 255  # 转为 uint8
                processed, stats = self._process_single(binary_mask)
                processed_preds[i] = (processed > 0).astype(float)  # 转回 float
                stats_list.append(stats)

            return processed_preds, stats_list
        elif preds.ndim == 2:
            binary_mask = (preds > 0.5).astype(np.uint8) * 255
            processed, stats = self._process_single(binary_mask)
            return (processed > 0).astype(float), stats
        else:
            raise ValueError("Input preds must be 2D or 3D numpy array.")

    def postprocess_mask(self, mask: np.ndarray):
        """
        输入：uint8 类型的二值化图像 [0,255]，形状为 (H, W)
        输出：处理后的 uint8 图像 [0,255]

        :param mask: 二值图像
        :return: processed_mask, stats
        """
        assert mask.dtype == np.uint8, "mask 必须是 uint8 类型"
        return self._process_single(mask)

    def _process_single(self, mask: np.ndarray):
        """
        内部通用方法，实际进行后处理

        :param mask: 输入必须是 uint8 类型，值范围 [0,255]
        :return: 处理后的图像和统计信息
        """
        # 确保是单通道二值图像
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_mask = np.zeros_like(mask)

        valid_areas = []
        valid_p_a_ratios = []
        valid_convexities = []

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if area < self.min_area:
                continue

            if perimeter == 0:
                continue

            p_a_ratio = perimeter / area
            if p_a_ratio > self.max_p_a_ratio:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            if hull_area == 0:
                continue

            convexity = area / hull_area
            if convexity < self.min_convexity:
                continue

            valid_areas.append(area)
            valid_p_a_ratios.append(p_a_ratio)
            valid_convexities.append(convexity)

            cv2.drawContours(result_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # 统计信息
        min_area = min(valid_areas) if valid_areas else None
        max_ratio = max(valid_p_a_ratios) if valid_p_a_ratios else None
        min_convexity = min(valid_convexities) if valid_convexities else None

        return result_mask, (min_area, max_ratio, min_convexity)