# 🧪 SegChange-R1 ONNX 推理说明文档

本项目基于 `SegChange-R1` 构建了一个变化检测模型，并支持将其导出为 ONNX 格式以便在多种平台上进行部署和推理。本文档详细介绍了如何安装依赖项、将 PyTorch 模型转换为 ONNX 格式，以及如何使用 ONNX Runtime 进行推理。

---

## 📦 一、环境准备

### 1. 安装依赖

```bash

```

### 安装 ONNX 运行时后端
您需要根据您的硬件选择合适的 ONNX 运行时包。

### GPU 加速 （NVIDIA）

如果您有 NVIDIA GPU 并希望利用 CUDA 进行更快的推理，请安装该软件包。确保您安装了正确的 NVIDIA 驱动程序和 CUDA 工具包。有关兼容性详细信息，请参阅官方 ONNX 运行时 GPU 文档。onnxruntime-gpu

'''
pip install onnxruntime-gpu, onnx
'''

#### 仅 CPU

如果您没有兼容的 NVIDIA GPU 或更喜欢基于 CPU 的推理，请安装标准软件包。有关更多选项，请查看 ONNX 运行时安装指南。onnxruntime

'''
pip install onnxruntime, onnx
'''

---

## 🔁 二、模型导出为 ONNX 格式

### 1. 导出脚本：`onnx_export.py`

确保你已经准备好训练好的模型权重文件（`.pth` 或 `.pt`）后，运行如下命令：

```bash
python ./examples/ONNXRuntime/onnx_export.py -c ./configs/config.yaml
```

> ⚠️ `config.yaml` 需要包含：
- `infer.weights_dir`: 权重路径
- `infer.output_dir`: 输出目录

### 2. 参数说明（导出时）

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `-c`, `--config` | str | 配置文件路径（YAML） |

---

## 🚀 三、ONNX 推理使用指南

### 1. 推理脚本：`main.py`

#### ✅ 不带 prompt 推理

```bash
CUDA_VISIBLE_DEVICES=0 python ./examples/ONNXRuntime/main.py \
  --input_dir ./data/ZK_5 \
  --onnx_model ./examples/ONNXRuntime/weights/segchange.onnx \
  --output_dir ./examples/ONNXRuntime/results/
```

#### ✅ 带 prompt 推理（如支持）

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --input_dir ./data/ZK_5 \
  --onnx_model ./examples/ONNXRuntime/weights/segchange.onnx \
  --output_dir ./examples/ONNXRuntime/results/
  --prompt "Buildings with changes"
```

#### ✅ 分块大图推理（适合遥感图像）

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --input_dir ./data/ZK_5 \
  --onnx_model ./examples/ONNXRuntime/weights/segchange.onnx \
  --output_dir ./examples/ONNXRuntime/results/
  --chunk_size 25600
```

---

### 2. 推理参数说明

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| `--input_dir` | str | ✅ | 输入图像目录（需包含两个 `.tif` 文件） |
| `--onnx_model` | str | ✅ | ONNX 模型路径 |
| `--output_dir` | str | ✅ | 推理结果输出目录 |
| `--prompt` | str | ❌ | 文本提示（如 `"Buildings with changes"`） |
| `--chunk_size` | int | ❌ | 图像分块大小（默认为 0 表示不分块） |

---

## 📚 四、多线程推理

多线程推理可以提高推理速度，但需要根据实际情况调整线程数。

### 1. 推理脚本：`main_parallel.py`

```bash
CUDA_VISIBLE_DEVICES=0 python main_parallel.py \
  --input_dir /path/to/images \
  --onnx_model /path/to/model.onnx \
  --output_dir /path/to/output \
  --chunk_size 25600 \
  --batch_size 2 \ 
  --device 'cuda:1' \
  --crop_threads 4 \ 
  --inference_threads 2 \ 
  --max_queue_size 100 \
  --log_inerval 1 \
  --prompt "Buildings with changes, Mound changes."
```

### 2. 推理参数说明

| 参数名 | 类型 | 必填 | 描述            |
|--------|------|------|---------------|
| `--input_dir` | str | ✅ | 输入图像目录（需包含两个 `.tif` 文件） |
| `--onnx_model` | str | ✅ | ONNX 模型路径     |
| `--output_dir` | str | ✅ | 推理结果输出目录      |
| `--prompt` | str | ❌ | 文本提示（如 `"Buildings with changes"`） |
| `--chunk_size` | int | ❌ | 图像分块大小（默认为 0 表示不分块） |
| `--batch_size` | int | ❌ | 推理批次大小（默认为 1） |
| `--device` | str | ❌ | 使用设备（如 `'cuda'` 或 `'cpu'`，默认为 `'cuda'`） |
| `--crop_threads` | int | ❌ | 裁剪线程数（默认为 4）  |
| `--inference_threads` | int | ❌ | 推理线程数（默认为 2）  |
| `--max_queue_size` | int | ❌ | 最大队列大小（默认为100）|
| `--log_inerval` | int | ❌ | 日志输出间隔（默认为 1s）|

---

## 📁 五、输入输出格式说明

### 输入要求：

- 图像尺寸：任意，但推荐为 512x512 及其倍数
- 图像格式：RGB 格式的 `.tif` 文件（支持多通道）
- Prompt（可选）：文本描述，表示需要识别的变化类型

### 输出内容：

- 推理掩码图像：`result_mask.tif`
- 每个滑窗区域的可视化图像：`results/masks/*.jpg`

---

## 📝 六、注意事项

1. **Prompt 支持**：当前示例中的 prompt 是随机编码的占位符，实际使用中应替换为 CLIP 编码或其他文本编码器。
2. **ONNX 模型结构**：确保模型导出时包含了 prompt 输入节点（如有）。
3. **大图处理**：对于大于显存容量的图像，请使用 `--chunk_size` 启用分块推理。
4. **设备支持**：ONNX Runtime 自动选择 CUDA 或 CPU 执行提供者。