<!--# [SegChange-R1: LLM-Augmented Remote Sensing Change Detection](https://arxiv.org/abs/2506.17944) -->

简体中文 | [English](README.md) | [简体中文](README_zh.md) | [CSDN Blog](https://blog.csdn.net/weixin_62828995?spm=1000.2115.3001.5343)

<h2 align="center">
  SegChange-R1: LLM-Augmented Remote Sensing Change Detection
</h2>

<p align="center">
    <a href="https://huggingface.co/spaces/yourusername/TAPNet">
        <img alt="hf" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
    </a>
    <a href="https://github.com/Yu-Zhouz/SegChange-R1/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://github.com/Yu-Zhouz/SegChange-R1/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/Yu-Zhouz/SegChange-R1">
    </a>
    <a href="https://github.com/Yu-Zhouz/SegChange-R1/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/Yu-Zhouz/SegChange-R1?color=olive">
    </a>
    <a href="https://arxiv.org/abs/2506.17944">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.06937v1-red">
    </a>
    <a href="https://results.pre-commit.ci/latest/github/Yu-Zhouz/SegChange-R1/master">
        <img alt="pre-commit.ci status" src="https://results.pre-commit.ci/badge/github/Yu-Zhouz/SegChange-R1/master.svg">
    </a>
    <a href="https://github.com/Yu-Zhouz/SegChange-R1">
        <img alt="stars" src="https://img.shields.io/github/stars/Yu-Zhouz/SegChange-R1">
    </a>
</p>

<p align="center">
    📄 This is the official implementation of the paper:
    <br>
    <a href="https://arxiv.org/abs/2506.17944">SegChange-R1: LLM-Augmented Remote Sensing Change Detection</a>
</p>

<p align="center">
Fei Zhou
</p>

<p align="center">
Neusoft Institute Guangdong, China & Airace Technology Co.,Ltd., China
</p>

<p align="center">
<strong>如果您喜欢 SegChange-R1，请为我们点赞⭐！您的支持是我们不断进步的动力！</strong>
</p>

遥感变化检测通过分析同一区域的特征随时间的变化，可用于城市规划、地形分析和环境监测。 在本文中，我们提出了一种大语言模型（LLM）增强推理方法（SegChange-R1），它通过整合文本描述信息来增强检测能力，并引导模型关注相关变化区域，从而加快收敛速度。 我们设计了一个基于注意力的线性空间转换模块（BEV），通过将不同时间的特征统一到一个 BEV 空间来解决模态错位问题。 此外，我们还介绍了 DVCD，这是一种用于从无人机视点进行建筑物变化检测的新型数据集。 在四个广泛使用的数据集上进行的实验表明，与现有方法相比，DVCD 有了显著的改进。

![Baseline](https://i-blog.csdnimg.cn/direct/574c18c2382c442a8cb60b31de9c01ba.png)

## 🚀 更新

- ✅ **[2024.06.01]** 开源代码
- ✅ **[2025.06.22]** 上传到 [arXiv](https://arxiv.org/abs/2506.17944)。

## 模型评估结果

![SOTA](https://i-blog.csdnimg.cn/direct/186edd2273c149bcb19f9aafb60b835b.png)


## 入门指南

### 系统要求

- Python 3.12
- CUDA + PyTorch
- HuggingFace
- 稳定的网络连接
- 高质量代理IP（重要）

### 安装步骤

#### 1. 创建虚拟环境

```bash
conda create -n segchange python=3.12 -y
conda activate segchange
```

#### 2. 安装依赖包

```bash
pip install -r requirements.txt
```

#### 3. 配置HuggingFace镜像

```bash
vim ~/.bashrc

export HF_ENDPOINT="https://hf-mirror.com"

source ~/.bashrc
```

### 数据集介绍

支持两种数据集结构格式：

#### 1. 默认结构：

数据集的结构如下：
```text
data/
  ├── train/
  │   ├── A/                  # 第一时相训练图像
  │   ├── B/                  # 第二时相训练图像
  │   ├── label/              # 训练标签（变化掩码）
  │   └── prompts.txt         # 训练集文本描述提示
  ├── val/
  │   ├── A/                  # 第一时相验证图像
  │   ├── B/                  # 第二时相验证图像
  │   ├── label/              # 验证标签（变化掩码）
  │   └── prompts.txt         # 验证集文本描述提示
  └── test/
      ├── A/                  # 第一时相测试图像
      ├── B/                  # 第二时相测试图像
      ├── label/              # 测试标签（变化掩码）
      └── prompts.txt         # 测试集文本描述提示
```
修改参数文件[configs](./configs/config.yaml)中的`data_format`为`default`。

#### 2. 自定义结构：

数据集的结构如下：
```text
data/
  ├── A/                      # 第一时相训练图像
  │── B/                      # 第二时相训练图像
  │── label/                  # 标签（变化掩码）
  │── list                    # 列表文件
  │   ├── train.txt           # 训练集列表
  │   ├── val.txt             # 验证集列表
  │   └── test.txt            # 测试集列表
  └── prompts.txt             # 文本描述提示
```
修改参数文件[configs](./configs/config.yaml)中的`data_format`为`custom`。

### 训练

#### 生成词嵌入文件

使用[文本生成](./examples/text_gen.py), 并修改参数文件[configs](./configs/config.yaml)中的`desc_embs`为`None`，执行脚本。

```bash
python ./examples/text_gen.py -c ./configs/config.yaml
```
如果需要做多类别变化检测，则需要手工标注类别描述文本。

#### 命令行训练
```bash
python train.py -c ./configs/config.yaml
```

### 测试
```bash
python test.py -c ./configs/config.yaml
```

### 推理TIF
```bash
python infer.py -c ./configs/config.yaml
```

### app demo

```bash
cd examples/gradio_app
chmod +x ./run.sh
bash run.sh
```
### 贡献

欢迎提交问题和代码改进。请确保遵循项目的代码风格和贡献指南。

## 许可证

本项目使用 [Apache License 2.0](LICENSE)

### 引用
如果您在研究中使用 SegChange-R1，请引用：

<details open>
<summary> bibtex </summary>

```bibtex
@article{zhou2025segchange-r1,
  title={SegChange-R1: LLM-Augmented Remote Sensing Change Detection},
  author={Zhou, Fei},
  journal={arXiv preprint arXiv:2506.17944},
  year={2025},
  eprint={/2506.17944},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
</details>
