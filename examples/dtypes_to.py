# -*- coding: utf-8 -*-
"""
@Project : SegChange-R2
@FileName: dtypes_to.py
@Time    : 2025/6/4 下午4:38
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   : 
"""
import rasterio
from rasterio.enums import Resampling
import numpy as np
from pathlib import Path


def convert_tif_to_uint8(input_tif_path: str, output_tif_path: str):
    """
    将输入的TIF文件转换为uint8格式的TIF文件。

    Args:
        input_tif_path (str): 输入TIF文件的路径。
        output_tif_path (str): 输出uint8格式TIF文件的路径。
    """
    input_path = Path(input_tif_path)
    output_path = Path(output_tif_path)

    if not input_path.exists():
        print(f"错误：输入文件不存在：{input_tif_path}")
        return

    try:
        with rasterio.open(input_path) as src:
            # 读取原始数据
            data = src.read()

            # 将数据转换为uint8类型
            # 归一化到0-255范围，然后转换为uint8
            # 确保数据不是空的，并且有最大最小值
            if data.size == 0:
                print(f"警告：输入文件 {input_tif_path} 不包含任何数据。")
                return

            # 处理可能存在的NaN值，将其替换为0或一个合适的值
            data[np.isnan(data)] = 0

            # 计算数据的最小值和最大值，用于归一化
            min_val = np.min(data)
            max_val = np.max(data)

            if max_val == min_val:
                # 如果所有像素值都相同，直接设置为0或255
                uint8_data = np.full(data.shape, 0, dtype=np.uint8)
                if max_val > 0:
                    uint8_data = np.full(data.shape, 255, dtype=np.uint8)
            else:
                # 归一化到0-1，然后乘以255，最后转换为uint8
                uint8_data = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            # 更新元数据以匹配新的数据类型和尺寸
            profile = src.profile
            profile.update(
                dtype=rasterio.uint8,
                count=src.count,  # 保持波段数不变
                compress='lzw'  # 压缩方式
            )

            # 写入新的TIF文件
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(uint8_data)

        print(f"成功将 {input_tif_path} 转换为 {output_tif_path} (uint8格式)。")

    except Exception as e:
        print(f"处理文件 {input_tif_path} 时发生错误：{e}")


if __name__ == "__main__":
    # 示例用法
    # 请将 'input.tif' 替换为您的输入TIF文件路径r
    # 将 'output_uint8.tif' 替换为您希望保存的输出TIF文件路径
    input_file = r"/sxs/zhoufei/SegChange-R2/data/Base1/HY_10_1/HY202201.tif"
    output_file = r"/sxs/DataSets/Base/HY_10_2/HY202201.tif"

    # 确保替换为实际的文件路径
    # 例如：
    # input_file = "e:/data/test.tif"
    # output_file = "e:/data/test_uint8.tif"

    convert_tif_to_uint8(input_file, output_file)
