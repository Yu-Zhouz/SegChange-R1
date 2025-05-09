# SegChange-R1

## 项目介绍

## 系统要求

- Python 3.12
- CUDA
- PyTorch
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

数据集的结构如下：
```text
data
    |--train
        |--A
        |--B
        |--label
        |--prompts.txt
    |--val
        |--A
        |--B
        |--label
        |--prompts.txt
    |___test
        |--A
        |--B
        |--label
        |--prompts.txt
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