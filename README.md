# SegChange-R1

## 项目介绍

SegChange-R1 是一个基于深度学习的变化检测模型项目，主要用于分析和识别图像中的变化区域（如建筑物变化等）。该项目结合了视觉编码器与文本描述信息，通过 双时相视觉编码器 提取双时态图像的多尺度特征，利用特征差异模块进行特征差异建模，并引入多尺度特征融合模块融合多尺度特征。此外，它支持集成文本描述信息以增强检测能力，使用掩码预测头生成最终的变化掩码。项目还提供了完整的训练、测试流程及损失函数配置，适用于遥感图像、城市规划、环境监测等领域中的变化检测任务。

## 系统要求

- Python 3.12
- CUDA + PyTorch
- HuggingFace
- 稳定的网络连接
- 高质量代理IP（重要）

## 安装步骤

### 1. 创建虚拟环境

```bash
conda create -n segchange python=3.12 -y
conda activate segchange
```

### 2. 安装依赖包

```bash
pip install -r requirements.txt
```

### 3. 配置HuggingFace镜像

```bash
vim ~/.bashrc
 
export HF_ENDPOINT="https://hf-mirror.com"

source ~/.bashrc
```

## 数据集介绍
支持两种数据集结构格式：
### 1. 默认结构：
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

### 2. 自定义结构：
数据集的结构如下：
```text
data/
  ├── A/                  # 第一时相训练图像
  │── B/                  # 第二时相训练图像
  │── label/              # 训练标签（变化掩码）
  │── list                # 列表文件
  │   ├── train.txt       # 训练集列表
  │   ├── val.txt         # 训练集列表
  │   └── test.txt        # 验证集列表
  └── prompts.txt         # 训练集文本描述提示
```

## 训练

### 命令行训练
```shell
python train.py -c ./configs/config.yaml
```

## 测试
```shell
python test.py -c ./configs/config.yaml
```

## 贡献

欢迎提交问题和代码改进。请确保遵循项目的代码风格和贡献指南。

## 许可证

本项目使用 [MIT许可证](LICENSE)


## 参考
https://blog.csdn.net/weixin_45679938/article/details/142030784
https://www.arxiv.org/pdf/2503.11070
https://www.arxiv.org/abs/2503.16825
https://zhuanlan.zhihu.com/p/627646794