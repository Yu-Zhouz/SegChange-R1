# -*- coding: utf-8 -*-
"""
@Project : SegChange-R2
@FileName: tif_to_jpg.py
@Time    : 2025/6/10 上午9:07
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 对指定文件夹下的tif文件修改为jpg另存为新路径
@Usage   : 
"""
from PIL import Image
import numpy as np
import os


def tif_to_jpg(input_folder, output_folder):
    """
    将.tif标注图像转换为.jpg格式，并保持视觉可见性。

    :param input_folder: 输入的.tif文件夹路径
    :param output_folder: 输出的.jpg文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.tif'):
            input_path = os.path.join(input_folder, filename)
            with Image.open(input_path) as img:
                # 转换为 numpy 数组进行处理
                img_array = np.array(img)

                # 假设是二值图像（0 和 1）
                img_array = img_array * 255  # 将 1 映射为 255（白色）

                # 确保数据类型为 uint8
                img_uint8 = img_array.astype(np.uint8)

                # 转换为 PIL 图像并保存为 jpg
                img_pil = Image.fromarray(img_uint8)
                jpg_filename = os.path.splitext(filename)[0] + '.jpg'
                output_path = os.path.join(output_folder, jpg_filename)
                img_pil.convert('L').save(output_path, 'JPEG')


# 示例用法
input_folder = r'/sxs/zhoufei/SegChange-R2/data/CD/DSIFN/test/label'  # 替换为你的.tif文件夹路径
output_folder = r'/sxs/zhoufei/SegChange-R2/data/CD/DSIFN/test/mask'  # 替换为你想保存.jpg文件的路径

tif_to_jpg(input_folder, output_folder)
