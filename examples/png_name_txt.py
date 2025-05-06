import os


def export_png_filenames_to_txt(directory, output_file, additional_text):
    """
    导出指定目录下的所有 .png 文件名到指定的 .txt 文件中，每个文件名占一行，并在文件名后添加指定的文本内容。

    :param directory: 包含 .png 文件的目录路径
    :param output_file: 输出的 .txt 文件路径
    :param additional_text: 要添加到每个文件名后面的文本内容
    """
    # 获取目录下所有文件
    files = os.listdir(directory)

    # 过滤出 .png 文件
    png_files = [file for file in files if file.lower().endswith('.png')]

    # 将文件名和附加文本写入 .txt 文件
    with open(output_file, 'w') as f:
        for png_file in png_files:
            f.write(png_file + '  ' + additional_text + '\n')

    print(f"已导出 {len(png_files)} 个 .png 文件名到 {output_file}，并添加了附加文本。")


# 使用示例
directory_path = '../data/change/val/A'  # 替换为你的图像目录路径
output_txt_path = '../data/change/val/prompts.txt'  # 输出的 .txt 文件路径
additional_text = 'Buildings with changes'
export_png_filenames_to_txt(directory_path, output_txt_path, additional_text)
