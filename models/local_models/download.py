# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: download.py
@Time    : 2025/4/17 下午4:24
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 下载模型
@Usage   :
"""
from huggingface_hub import snapshot_download
model_id="microsoft/bitnet-b1.58-2B-4T"
snapshot_download(repo_id=model_id, local_dir="./bitnet-b1.58-2B-4T",
                          local_dir_use_symlinks=False, revision="main")