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

- [English README](README_EN.md)
- [中文 README](README_CN.md)

---

## Notice

This source release includes code, configuration files, README files, and example figures only. The dataset and large model weights are not included in this repository.

---

## Quick Start

```bash
git clone https://github.com/ZhanpengLuo2006/PDS-Net.git
cd PDS-Net
pip install -r yolov13-main/requirements.txt
```

Please refer to [README_EN.md](README_EN.md) or [README_CN.md](README_CN.md) for training, validation, inference, and visualization instructions.
