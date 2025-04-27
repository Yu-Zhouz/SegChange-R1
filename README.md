# SegChange-R1





## 环境搭建

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

## 数据集

数据集的结构为：
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
