from __future__ import annotations

import argparse
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TIMM_FX_DISABLED", "1")

if "wandb" not in sys.modules:
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.__dict__.update(
        {
            "init": lambda *args, **kwargs: None,
            "log": lambda *args, **kwargs: None,
            "finish": lambda *args, **kwargs: None,
            "login": lambda *args, **kwargs: None,
            "watch": lambda *args, **kwargs: None,
            "run": None,
            "config": {},
            "__all__": [],
        }
    )
    sys.modules["wandb"] = wandb_stub

ROOT = Path(__file__).resolve().parent
YOLO_ROOT = ROOT / "yolov13-main"
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from ultralytics import RTDETR, YOLO  # noqa: E402
import ultralytics.nn.tasks as yolo_tasks  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_SOURCE = ROOT / "example figures"
DEFAULT_OUTPUT = ROOT / "detect_comparision"

MODELS: dict[str, Path] = {
    "PDS-Net": ROOT / r"experient results\ablation\DSC3k2_PPA+DySample\weights\best.pt",
    "YOLOv13n": ROOT / r"experient results\baseline\13n2\weights\best.pt",
    "YOLOv6n": ROOT / r"experient results\baseline\6n3\weights\best.pt",
    "YOLOv8n": ROOT / r"experient results\baseline\8n\weights\best.pt",
    "YOLOv11n": ROOT / r"experient results\baseline\11n\weights\best.pt",
    "YOLOv12n": ROOT / r"experient results\baseline\12n\weights\best.pt",
    "RT-DETR-l": ROOT / r"experient results\baseline\rtdetr-l\weights\best.pt",
    "RT-DETR-ResNet50": ROOT / r"experient results\baseline\rtdetr-resnet50\weights\best.pt",
    "RT-DETR-ResNet101": ROOT / r"experient results\baseline\rtdetr-resnet101\weights\best.pt",
    "RT-DETR-x": ROOT / r"experient results\baseline\rtdetr-x\weights\best.pt",
}


def patch_yolo_apply() -> None:
    def patched_apply(self, fn):
        self = super(yolo_tasks.BaseModel, self)._apply(fn)
        if not hasattr(self, "model") or not self.model:
            return self

        head = self.model[-1]
        detect_types = tuple(
            t
            for t in (
                getattr(yolo_tasks, "Detect", None),
                getattr(yolo_tasks, "Detect_AFPN3", None),
                getattr(yolo_tasks, "Detect_AFPN4", None),
                getattr(yolo_tasks, "Detect_Dyhead", None),
                getattr(yolo_tasks, "Detect_MBConv", None),
            )
            if isinstance(t, type)
        )

        if detect_types and isinstance(head, detect_types):
            for attr in ("stride", "anchors", "strides"):
                if hasattr(head, attr):
                    value = getattr(head, attr)
                    if value is not None:
                        setattr(head, attr, fn(value))
        return self

    yolo_tasks.BaseModel._apply = patched_apply


patch_yolo_apply()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection comparison on example images with multiple models.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Input image folder or image path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", type=str, default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width.")
    return parser.parse_args()


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTS:
            return [source]
        raise ValueError(f"Unsupported image file: {source}")

    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found in: {source}")
    return images


def existing_models() -> dict[str, Path]:
    models = {}
    for name, weight in MODELS.items():
        if weight.exists():
            models[name] = weight
        else:
            print(f"[WARN] Missing model weight, skipped: {name} -> {weight}")
    if not models:
        raise FileNotFoundError("No model weights found.")
    return models


def save_prediction(result: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(output_path))


def load_model(model_name: str, weight_path: Path) -> Any:
    if model_name.lower().startswith("rt-detr"):
        return RTDETR(str(weight_path))
    return YOLO(str(weight_path))


def run_inference(args: argparse.Namespace) -> None:
    images = collect_images(args.source)
    models = existing_models()
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Detection comparison inference")
    print(f"Images: {len(images)} from {args.source}")
    print(f"Models: {len(models)}")
    print(f"Output: {args.output}")
    print("=" * 70)

    loaded_models: dict[str, Any] = {}
    failed_models: dict[str, str] = {}
    for model_name, weight_path in models.items():
        print(f"[INFO] Loading model: {model_name} -> {weight_path}")
        try:
            loaded_models[model_name] = load_model(model_name, weight_path)
        except Exception as exc:
            failed_models[model_name] = str(exc)
            print(f"[ERROR] Failed to load {model_name}: {exc}")

    if not loaded_models:
        raise RuntimeError("All models failed to load.")

    for image_path in images:
        image_dir = args.output / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Image: {image_path.name}")

        for model_name, model in loaded_models.items():
            out_path = image_dir / f"{safe_name(model_name)}{image_path.suffix.lower()}"
            print(f"  [RUN] {model_name}")
            try:
                results = model.predict(
                    source=str(image_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    save=False,
                    verbose=False,
                    line_width=args.line_width,
                )
            except Exception as exc:
                print(f"  [ERROR] Failed to predict with {model_name}: {exc}")
                continue
            if not results:
                print(f"  [WARN] No result returned: {model_name}")
                continue
            save_prediction(results[0], out_path)
            print(f"  [OK]  {out_path}")

    if failed_models:
        print("\nFailed models:")
        for model_name, error in failed_models.items():
            print(f"  {model_name}: {error}")

    print("\nDone.")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
