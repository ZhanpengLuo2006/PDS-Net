# PDS-Net

<p align="center">
  <img src="Figures/模型结构图.png" width="90%" alt="PDS-Net Architecture">
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

> **PDS-Net** is a lightweight object detection model built on the YOLOv13 framework. This project focuses on crop image object detection and integrates structural improvements such as `DSC3k2_PPA` and `DySample` to achieve a favorable balance among detection accuracy, model size, computational cost, and inference speed.

---

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Model Architecture](#model-architecture)
- [Model Weights](#model-weights)
- [Dataset](#dataset)
- [Experimental Results](#experimental-results)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Training Demo](#training-demo)
- [Validation](#validation)
- [Batch Inference and Detection Comparison](#batch-inference-and-detection-comparison)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Overview

Crop images collected under natural conditions often suffer from illumination variation, occlusion, complex background, and scale changes. To improve detection robustness under such conditions, this project builds an improved lightweight detector, **PDS-Net**, based on the YOLOv13n detection framework.

The repository includes:

- Custom YOLOv13 model configurations
- Baseline and ablation experiment results
- Multi-model validation script
- Multi-model inference comparison script
- Training-curve visualization script
- P/R-Confidence curve visualization script

---

## Highlights

| Highlight | Description |
|---|---|
| Lightweight design | PDS-Net has about **2.29M** parameters and a **4.80 MB** weight file |
| Low computational cost | About **5.98 GFLOPs** at `imgsz=640` |
| Fast inference | About **309.76 FPS** in the validation log |
| Modular improvements | Combines `DSC3k2_PPA` and `DySample` for stronger feature representation and upsampling |
| Comprehensive comparisons | Includes YOLOv13n, YOLOv6n, YOLOv8n, YOLOv11n, YOLOv12n, and RT-DETR baselines |
| Visualization utilities | Supports training curves, confidence curves, and detection result comparison |

---

## Model Architecture

The main improved model configuration is located at:

```text
experient results/v13/DSC3k2_PPA+DySample.yaml
```

The corresponding best checkpoint is located at:

```text
experient results/ablation/DSC3k2_PPA+DySample/weights/best.pt
```

Main ideas:

1. **DSC3k2_PPA**: Enhances multi-scale feature extraction and attention-aware representation.
2. **DySample**: Introduces dynamic upsampling to better recover spatial details during feature fusion.
3. **YOLOv13 detection framework**: Maintains an end-to-end object detection pipeline with a good balance between accuracy and deployment efficiency.

---

## Model Weights

The trained model weights are not included directly in this repository. The model weights can be obtained from the following Baidu Netdisk share:

```text
Shared file: weight
Link: https://pan.baidu.com/s/1Tanc-zAQbbhc8tZU6nWe6Q?pwd=neau
Extraction code: neau
Source note: shared from Baidu Netdisk Super Member v3
```

After downloading, place the `weight` folder in the project root, for example:

```text
PDS-Net/
└── weight/
    ├── DSC3k2_PPA+DySample_best.pt
    ├── DSC3k2_PPA+DySample_last.pt
    └── ...
```

---

## Dataset

The dataset can be obtained from the following Baidu Netdisk share:

```text
Shared file: rice_leaf_diseasev1
Link: https://pan.baidu.com/s/1ualoDg5uzVxUZq5jbIfA-g?pwd=neau
Extraction code: neau
Source note: shared from Baidu Netdisk Super Member v3
```

After downloading, organize the dataset in the YOLO format shown below and place it under the `dataset/` folder in the project root.

Dataset configuration file:

```text
dataset/data.yaml
```

Current dataset configuration:

```yaml
train: H:\Project\cropper\dataset\train\images
val: H:\Project\cropper\dataset\val\images
test: H:\Project\cropper\dataset\test\images

nc: 2
names: [white, black]
```

Recommended YOLO-format dataset structure:

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

## Experimental Results

The following results are taken from `validation_model_log.txt`. The validation setting is `imgsz=640`, `batch=16`, and `device=0`.

| Model | P | R | mAP@50 | mAP@50:95 | Parameters | GFLOPs | Weight(MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DSC3k2_PPA+DySample / PDS-Net | 0.8274 | 0.6103 | 0.7055 | 0.4830 | 2,294,653 | 5.98 | 4.80 | 309.76 |
| DSC3k2_PPA | 0.8406 | 0.6241 | 0.7212 | 0.4893 | 2,282,301 | 5.96 | 4.77 | 328.53 |
| DySample | 0.8613 | 0.6310 | 0.7289 | 0.5189 | 2,464,765 | 6.23 | 5.19 | 295.22 |
| YOLOv11n | 0.7525 | 0.5245 | 0.6143 | 0.3965 | 2,582,542 | 6.31 | 5.22 | 621.89 |

> Note: FPS may vary with GPU type, CUDA version, batch size, and input image size.

---

## Environment Setup

Recommended environment:

| Item | Version |
|---|---|
| OS | Windows 11 |
| Python | 3.10.19 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| CUDA | 12.1 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU 8GB |

Install dependencies:

```bash
cd H:/Project/cropper/yolov13-main
pip install -r requirements.txt
```

Using Conda:

```bash
conda create -n yolov11 python=3.10 -y
conda activate yolov11
pip install -r H:/Project/cropper/yolov13-main/requirements.txt
```

---

## Quick Start

Enter the project directory:

```bash
cd H:/Project/cropper
```

---

## Training Demo

The following example shows how to train PDS-Net with the project dataset `dataset/data.yaml` and the model configuration `experient results/v13/DSC3k2_PPA+DySample.yaml`.

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

You can save the example as `train_pdsnet.py` and run:

```bash
& F:/Anaconda/envs/yolov11/python.exe H:/Project/cropper/train_pdsnet.py
```

To resume training, set `resume=True` or provide an existing checkpoint path. To reduce GPU memory usage, decrease `batch` or set `cache=False`.

---

## Validation

Script:

```text
validate_models.py
```

This script validates multiple checkpoints and logs:

- Precision
- Recall
- mAP@50
- mAP@50:95
- Parameters
- GFLOPs
- Weight size
- FPS

Run:

```bash
python validate_models.py
```

Run with the specified Conda environment:

```bash
& F:/Anaconda/envs/yolov11/python.exe H:/Project/cropper/validate_models.py
```

Default checkpoints:

```text
experient results/ablation/DSC3k2_PPA+DySample/weights/best.pt
experient results/ablation/DSC3k2_PPA/weights/best.pt
experient results/ablation/DYSample/weights/best.pt
experient results/baseline/11n/weights/best.pt
```

Outputs:

```text
validation_model_log.txt
validation_model_log.csv
```

---

## Batch Inference and Detection Comparison

Script:

```text
predict.py
```

Input image folder:

```text
example figures/
```

Output folder:

```text
detect_comparision/
```

Run:

```bash
python predict.py
```

Or:

```bash
& F:/Anaconda/envs/yolov11/python.exe H:/Project/cropper/predict.py
```

Example output structure:

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

## Visualization

### Multi-model Training Curve Comparison

Script:

```text
visualization.py
```

This script draws:

- Training loss curves
- Validation loss curves
- Precision curve
- Recall curve
- mAP@50 curve
- mAP@50:95 curve
- A standalone legend image

Output folder:

```text
comparison_charts/
```

Run:

```bash
python visualization.py
```

### P/R-Confidence Curves

Script:

```text
visualization2.py
```

This script validates the PDS-Net checkpoint and draws:

- P-Confidence curve
- R-Confidence curve
- Class legend

Output folder:

```text
trian_img/
```

Run:

```bash
python visualization2.py
```

---

## Project Structure

```text
cropper/
├── dataset/                         # Dataset and data.yaml
├── example figures/                 # Example images for inference
├── experient results/               # Experiment results, model configs, weights, CSV logs
│   ├── ablation/                    # Ablation experiments
│   ├── baseline/                    # Baseline experiments
│   ├── nn/                          # Custom network modules
│   └── v13/                         # YOLOv13 model configs
├── comparison_charts/               # Training and metric curves
├── trian_img/                       # P/R-Confidence curves
├── detect_comparision/              # Multi-model detection comparison outputs
├── yolov13-main/                    # YOLOv13 / Ultralytics framework
├── validate_models.py               # Multi-model validation script
├── predict.py                       # Multi-model inference script
├── visualization.py                 # Training-result visualization script
└── visualization2.py                # P/R-Confidence curve script
```

---

## FAQ

### 1. What if `wandb` or NumPy 2.0 raises errors?

The project scripts include compatibility handling to disable unnecessary `wandb` imports and to support removed NumPy aliases such as `np.float_` and `np.complex_`.

### 2. What if RT-DETR raises an `isinstance()` type error?

This is caused by custom detection-head registration in the local YOLOv13 codebase. Runtime patches have been added to `predict.py` to avoid this issue.

### 3. Why does the dataset path sometimes point to an old Linux path?

Some training logs were generated in a Linux environment and still contain old paths in `args.yaml`. The current scripts remap dataset paths to the local `dataset/` folder automatically.

---

## Acknowledgements

This project is built upon the following open-source frameworks and tools:

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- YOLOv13 detection framework
- PyTorch deep learning framework

---

## Citation

If this project is useful for your research, please consider citing your paper or repository:

```bibtex
None
```

---

## Contact

For questions or suggestions, please contact the authors through GitHub Issues, email, or the contact information provided in the paper.
