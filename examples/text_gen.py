# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: test.py
@Time    : 2025/4/24 下午5:24
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 文本生成器
@Usage   :
"""
import argparse
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import load_config


# 支持的图像格式
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']


def get_args_config():
    parser = argparse.ArgumentParser('SegChange')
    parser.add_argument('-c', '--config', type=str, required=True, help='The path of config file')
    args = parser.parse_args()
    if args.config is not None:
        cfg = load_config(args.config)
    else:
        raise ValueError('Please specify the config file')
    return cfg


def export_filenames_to_txt(directory, output_file, prompt):
    """
    导出指定目录下的所有图像文件名到指定的 .txt 文件中，每个文件名占一行，并在文件名后添加指定的文本内容。
    :param directory: 包含图像文件的目录路径
    :param output_file: 输出的 .txt 文件路径
    :param prompt: 要添加到每个文件名后面的文本内容
    """
    # 获取目录下所有文件
    files = os.listdir(directory)

    # 过滤出指定格式的图像文件
    image_files = [file for file in files if any(file.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS)]

    # 将文件名和附加文本写入 .txt 文件
    with open(output_file, 'w') as f:
        for image_file in image_files:
            f.write(image_file + '  ' + prompt + '\n')

    print(f"已导出 {len(image_files)} 个图像文件名到 {output_file}，并添加了附加文本。")


def generate_prompts_txt(cfg):
    """
    根据配置文件生成 prompts.txt 文件
    """
    data_format = cfg.data_format
    prompt = cfg.prompt if hasattr(cfg, 'prompt') else 'Buildings with changes'

    if data_format == 'default':
        # 默认数据集结构
        subsets = ['train', 'val', 'test']
        for subset in subsets:
            # 构建目录路径
            a_dir = os.path.join(cfg.data_root, subset, 'A')
            # 构建输出文件路径
            output_txt_path = os.path.join(cfg.data_root, subset, 'prompts.txt')
            # 导出文件名到 prompts.txt
            export_filenames_to_txt(a_dir, output_txt_path, prompt)
    elif data_format == 'custom':
        # 自定义数据集结构
        # 构建输入目录路径
        a_dir = os.path.join(cfg.data_root, 'A')
        # 构建输出文件路径
        output_txt_path = os.path.join(cfg.data_root, 'prompts.txt')
        # 导出文件名到 prompts.txt
        export_filenames_to_txt(a_dir, output_txt_path, prompt)

    else:
        raise ValueError(f"Unsupported data format: {data_format}")


def main():
    cfg = get_args_config()
    generate_prompts_txt(cfg)


if __name__ == "__main__":
    main()
