# PDS-Net

<p align="center">
  <img src="Figures/模型结构图.png" width="90%" alt="PDS-Net 模型结构图">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.5.1-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.1-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/Base-YOLOv13n-red.svg" alt="YOLOv13">
  <img src="https://img.shields.io/badge/Params-2.29M-brightgreen.svg" alt="Params">
  <img src="https://img.shields.io/badge/GFLOPs-5.98-blueviolet.svg" alt="GFLOPs">
  <img src="https://img.shields.io/badge/Weight-4.80MB-success.svg" alt="Weight">
  <img src="https://img.shields.io/badge/mAP@50-70.55%25-orange.svg" alt="mAP50">
</p>

> **PDS-Net** 是一个基于 YOLOv13 系列框架构建的轻量级目标检测模型。本项目围绕作物图像检测任务，集成了 `DSC3k2_PPA` 与 `DySample` 等结构改进，在保持较小参数量和模型体积的同时，实现了较好的检测精度与推理效率。

---

## 目录

- [项目概述](#项目概述)
- [主要特点](#主要特点)
- [模型结构](#模型结构)
- [模型权重](#模型权重)
- [数据集](#数据集)
- [实验结果](#实验结果)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [模型训练演示](#模型训练演示)
- [模型验证](#模型验证)
- [批量推理与结果对比](#批量推理与结果对比)
- [训练曲线与指标可视化](#训练曲线与指标可视化)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [致谢](#致谢)

---

## 项目概述

作物图像在自然场景下通常存在光照变化、遮挡、背景复杂、目标尺度差异明显等问题。为了提高检测模型在复杂图像环境中的识别能力，本项目以 YOLOv13n 为基础模型，对骨干网络、特征融合和上采样部分进行改进，形成轻量化检测模型 **PDS-Net**。

当前项目主要包含：

- 自定义 YOLOv13 系列模型配置文件
- 多组基线模型与消融实验结果
- 模型验证脚本
- 多模型批量推理对比脚本
- 训练指标可视化脚本
- P/R-Confidence 曲线绘制脚本

---

## 主要特点

| 特点 | 说明 |
|---|---|
| 轻量化设计 | PDS-Net 参数量约 **2.29M**，权重文件约 **4.80 MB** |
| 较低计算量 | 在 `imgsz=640` 下约 **5.98 GFLOPs** |
| 较高推理速度 | 验证日志中 FPS 约 **309.76** |
| 模块化改进 | 结合 `DSC3k2_PPA` 与 `DySample` 改进特征表达与上采样能力 |
| 对比实验完整 | 包含 YOLOv13n、YOLOv6n、YOLOv8n、YOLOv11n、YOLOv12n、RT-DETR 等对比模型 |
| 可视化完善 | 支持训练曲线、P/R 曲线、检测结果对比图生成 |

---

## 模型结构

本项目主要改进模型位于：

```text
experient results/v13/DSC3k2_PPA+DySample.yaml
```

对应最佳权重路径：

```text
experient results/ablation/DSC3k2_PPA+DySample/weights/best.pt
```

核心思路：

1. **DSC3k2_PPA**：增强多尺度特征提取与注意力表达能力。
2. **DySample**：采用动态上采样方式，提升特征融合阶段的空间细节恢复能力。
3. **YOLOv13 检测框架**：保持端到端检测流程，兼顾精度、速度和部署友好性。

---

## 模型权重

本项目训练得到的模型权重文件未直接包含在仓库中，模型权重从这里获取：

```text
通过网盘分享的文件：weight
链接：https://pan.baidu.com/s/1Tanc-zAQbbhc8tZU6nWe6Q?pwd=neau
提取码：neau
--来自百度网盘超级会员v3的分享
```

下载后请将 `weight` 文件夹放置在项目根目录下，例如：

```text
PDS-Net/
└── weight/
    ├── DSC3k2_PPA+DySample_best.pt
    ├── DSC3k2_PPA+DySample_last.pt
    └── ...
```

---

## 数据集

数据集可通过百度网盘获取：

```text
通过网盘分享的文件：rice_leaf_diseasev1
链接：https://pan.baidu.com/s/1ualoDg5uzVxUZq5jbIfA-g?pwd=neau
提取码：neau
--来自百度网盘超级会员v3的分享
```

下载后请将数据集整理为下方 YOLO 格式目录，并放置在项目根目录的 `dataset/` 文件夹下。

数据集配置文件位于：

```text
dataset/data.yaml
```

当前数据集配置：

```yaml
train: H:\Project\cropper\dataset\train\images
val: H:\Project\cropper\dataset\val\images
test: H:\Project\cropper\dataset\test\images

nc: 2
names: [white, black]
```

推荐数据组织格式：

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 实验结果

以下结果来自项目中的 `validation_model_log.txt`，验证设置为 `imgsz=640`、`batch=16`、`device=0`。

| Model | P | R | mAP@50 | mAP@50:95 | Parameters | GFLOPs | Weight(MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DSC3k2_PPA+DySample / PDS-Net | 0.8274 | 0.6103 | 0.7055 | 0.4830 | 2,294,653 | 5.98 | 4.80 | 309.76 |
| DSC3k2_PPA | 0.8406 | 0.6241 | 0.7212 | 0.4893 | 2,282,301 | 5.96 | 4.77 | 328.53 |
| DySample | 0.8613 | 0.6310 | 0.7289 | 0.5189 | 2,464,765 | 6.23 | 5.19 | 295.22 |
| YOLOv11n | 0.7525 | 0.5245 | 0.6143 | 0.3965 | 2,582,542 | 6.31 | 5.22 | 621.89 |

> 注：不同显卡、CUDA 版本、batch size、输入尺寸会导致 FPS 存在差异。

---

## 环境配置

推荐环境：

| 项目 | 版本 |
|---|---|
| OS | Windows 11 |
| Python | 3.10.19 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| CUDA | 12.1 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU 8GB |

安装依赖：

```bash
cd H:/Project/cropper/yolov13-main
pip install -r requirements.txt
```

如果使用 Conda：

```bash
conda create -n yolov11 python=3.10 -y
conda activate yolov11
pip install -r H:/Project/cropper/yolov13-main/requirements.txt
```

---

## 快速开始

进入项目目录：

```bash
cd H:/Project/cropper
```

---

## 模型训练演示

下面示例演示如何使用项目中的 PDS-Net 配置文件进行训练。训练配置对应当前项目数据集 `dataset/data.yaml`，模型结构配置为 `experient results/v13/DSC3k2_PPA+DySample.yaml`。

```python
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"H:/Project/cropper")

if __name__ == "__main__":
    model = YOLO(str(ROOT / r"experient results/v13/DSC3k2_PPA+DySample.yaml"))

    results = model.train(
        data=str(ROOT / "dataset/data.yaml"),
        epochs=250,
        batch=16,
        imgsz=640,
        device=0,
        workers=0,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        patience=60,
        seed=42,
        deterministic=True,
        cache="ram",
        amp=False,
        project=str(ROOT / r"experient results/ablation"),
        name="DSC3k2_PPA+DySample_train_demo",
    )
```

也可以保存为 `train_pdsnet.py` 后运行：

```bash
& F:/Anaconda/envs/yolov11/python.exe H:/Project/cropper/train_pdsnet.py
```

如需恢复训练，可将 `resume=True` 或传入已有权重路径；如需降低显存占用，可减小 `batch` 或使用 `cache=False`。

---

## 模型验证

脚本：

```text
validate_models.py
```

功能：

- 批量验证多个模型权重
- 输出 `P`、`R`、`mAP@50`、`mAP@50:95`
- 输出参数量、GFLOPs、权重大小和 FPS
- 生成 `validation_model_log.txt` 与 `validation_model_log.csv`

运行：

```bash
python validate_models.py
```

使用指定 Conda 环境运行：

```bash
& F:/Anaconda/envs/yolov11/python.exe H:/Project/cropper/validate_models.py
```

默认验证模型包括：

```text
experient results/ablation/DSC3k2_PPA+DySample/weights/best.pt
experient results/ablation/DSC3k2_PPA/weights/best.pt
experient results/ablation/DYSample/weights/best.pt
experient results/baseline/11n/weights/best.pt
```

---

## 批量推理与结果对比

脚本：

```text
predict.py
```

输入图片目录：

```text
example figures/
```

输出目录：

```text
detect_comparision/
```

运行：

```bash
python predict.py
```

或：

```bash
& F:/Anaconda/envs/yolov11/python.exe H:/Project/cropper/predict.py
```

输出结构示例：

```text
detect_comparision/
├── Day3/
│   ├── PDS-Net.png
│   ├── YOLOv13n.png
│   ├── YOLOv6n.png
│   ├── YOLOv8n.png
│   ├── YOLOv11n.png
│   ├── YOLOv12n.png
│   ├── RT-DETR-l.png
│   ├── RT-DETR-ResNet50.png
│   ├── RT-DETR-ResNet101.png
│   └── RT-DETR-x.png
└── Day4/
    └── ...
```

---

## 训练曲线与指标可视化

### 多模型训练曲线对比

脚本：

```text
visualization.py
```

功能：

- 绘制训练损失曲线
- 绘制验证损失曲线
- 绘制 Precision、Recall、mAP@50、mAP@50:95 曲线
- 生成独立图例

输出目录：

```text
comparison_charts/
```

运行：

```bash
python visualization.py
```

### P/R-Confidence 曲线

脚本：

```text
visualization2.py
```

功能：

- 使用 PDS-Net 权重重新验证数据集
- 绘制 `P-Confidence` 与 `R-Confidence` 曲线
- 生成类别图例

输出目录：

```text
trian_img/
```

运行：

```bash
python visualization2.py
```

---

## 项目结构

```text
cropper/
├── dataset/                         # 数据集与 data.yaml
├── example figures/                 # 示例推理图片
├── experient results/               # 实验结果、模型配置、权重、结果 CSV
│   ├── ablation/                    # 消融实验
│   ├── baseline/                    # 基线模型实验
│   ├── nn/                          # 自定义网络模块
│   └── v13/                         # YOLOv13 系列模型配置
├── comparison_charts/               # 训练曲线与指标曲线输出
├── trian_img/                       # P/R-Confidence 曲线输出
├── detect_comparision/              # 多模型检测结果对比输出
├── yolov13-main/                    # YOLOv13 / Ultralytics 框架代码
├── validate_models.py               # 批量验证脚本
├── predict.py                       # 多模型批量推理脚本
├── visualization.py                 # 多模型训练结果可视化脚本
└── visualization2.py                # P/R-Confidence 曲线脚本
```

---

## 常见问题

### 1. `wandb` 或 NumPy 2.0 报错怎么办？

项目脚本中已加入兼容处理，会禁用不必要的 `wandb` 导入，并兼容 `np.float_`、`np.complex_` 在 NumPy 2.0 中被移除的问题。

### 2. RT-DETR 加载时报 `isinstance()` 类型错误怎么办？

这是当前仓库中自定义检测头类型注册导致的问题。`predict.py` 中已加入运行时补丁，避免该问题中断执行。

### 3. 数据路径为什么会跑到旧的 Linux 路径？

部分训练记录来自 Linux 环境，`args.yaml` 中保留了旧路径。当前脚本会自动将数据路径映射到本机项目目录下的 `dataset/`。

---

## 致谢

本项目基于以下开源项目与工具构建：

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- YOLOv13 相关检测框架
- PyTorch 深度学习框架

---

## 引用

如果本项目对你的研究有帮助，请考虑引用你的论文或项目：

```bibtex
None
```

---

## 联系方式

如有问题或建议，可通过 GitHub Issue、邮件或论文中提供的联系方式联系作者。
