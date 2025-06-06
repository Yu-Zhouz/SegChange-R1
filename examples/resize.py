# -*- coding: utf-8 -*-
"""
@Project : SegChange-R2
@FileName: resize.py
@Time    : 2025/6/6 下午1:39
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 修改mask图像尺寸
@Usage   : 
"""

import os
from PIL import Image

SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def resize_masks(input_folder, output_folder, target_size):
    """
    修改指定文件夹下所有 mask 图像的尺寸，并保存到新目录。

    参数:
        input_folder (str): 包含原始 mask 图像的文件夹路径。
        output_folder (str): 保存调整尺寸后图像的目标文件夹路径。
        target_size (tuple): 目标尺寸，格式为 (width, height)。
    """
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 使用配置中的支持格式进行过滤
        if filename.lower().endswith(SUPPORTED_IMAGE_FORMATS):
            try:
                # 打开图像
                with Image.open(file_path) as img:
                    # 缩放图像
                    resized_img = img.resize(target_size, Image.NEAREST)

                    # 构建输出文件路径
                    base_name = os.path.splitext(filename)[0]
                    output_file_path = os.path.join(output_folder, f"{base_name}.jpg")

                    # 保存缩放后的图像
                    resized_img.save(output_file_path)

                    print(f"已处理: {filename}， 输出文件: {output_file_path}")
            except Exception as e:
                print(f"处理 {filename} 出错: {e}")


if  __name__ == "__main__":
    input_folder = "/sxs/DataSets/CD/DSIFN/train/mask_256"  # 替换为你的输入文件夹路径
    output_folder = "/sxs/DataSets/CD/DSIFN/train/label"  # 替换为你的输出文件夹路径
    target_size = (512, 512)  # 替换为你想要的目标尺寸

    resize_masks(input_folder, output_folder, target_size)
