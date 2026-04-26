from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402

DEFAULT_WEIGHTS = [
    ROOT / r"experient results\ablation\DSC3k2_PPA+DySample\weights\best.pt",
    ROOT / r"experient results\ablation\DSC3k2_PPA\weights\best.pt",
    ROOT / r"experient results\ablation\DYSample\weights\best.pt",
    ROOT / r"experient results\baseline\11n\weights\best.pt",
]
DEFAULT_DATA = 'H:\Project\cropper\dataset\data.yaml'
DEFAULT_OUTPUT_TXT = ROOT / "validation_model_log.txt"
DEFAULT_OUTPUT_CSV = ROOT / "validation_model_log.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multiple YOLO models and log key metrics.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to dataset yaml.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Validation device, e.g. 0 or cpu. Use cpu if no GPU is available.",
    )
    parser.add_argument("--split", type=str, default="val", help="Dataset split for validation.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "multi_val", help="Validation output dir.")
    parser.add_argument("--name", type=str, default="compare_models", help="Validation run name.")
    parser.add_argument("--txt", type=Path, default=DEFAULT_OUTPUT_TXT, help="Output txt log path.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output csv path.")
    parser.add_argument(
        "--weights",
        nargs="*",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Optional custom weight paths. Defaults to the four requested checkpoints.",
    )
    return parser.parse_args()


def build_local_data_yaml(src_yaml: Path) -> Path:
    with src_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = ROOT / "dataset"
    images_root = {
        "train": dataset_root / "train" / "images",
        "val": dataset_root / "val" / "images",
        "test": dataset_root / "test" / "images",
    }

    for key, path in images_root.items():
        if key in data and path.exists():
            data[key] = str(path)

    fd, temp_path = tempfile.mkstemp(prefix="cropper_data_", suffix=".yaml")
    os.close(fd)
    temp_yaml = Path(temp_path)
    temp_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return temp_yaml


def safe_float(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def extract_detection_metrics(metrics: Any) -> tuple[float | None, float | None, float | None, float | None]:
    results_dict = getattr(metrics, "results_dict", {}) or {}
    p = results_dict.get("metrics/precision(B)")
    r = results_dict.get("metrics/recall(B)")
    map50 = results_dict.get("metrics/mAP50(B)")
    map5095 = results_dict.get("metrics/mAP50-95(B)")

    if None not in (p, r, map50, map5095):
        return tuple(safe_float(x) for x in (p, r, map50, map5095))

    mean_results = None
    if hasattr(metrics, "box") and hasattr(metrics.box, "mean_results"):
        mean_results = metrics.box.mean_results()
    elif hasattr(metrics, "mean_results"):
        mean_results = metrics.mean_results()

    if mean_results is not None and len(mean_results) >= 4:
        return tuple(safe_float(x) for x in mean_results[:4])

    return None, None, None, None


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def validate_one_model(weight_path: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight not found: {weight_path}")

    model = YOLO(str(weight_path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        project=str(args.project),
        name=f"{args.name}_{weight_path.parent.parent.name}",
        exist_ok=True,
        verbose=True,
        plots=False,
    )

    p, r, map50, map5095 = extract_detection_metrics(metrics)
    inference_ms = None
    fps = None
    speed = getattr(metrics, "speed", {}) or {}
    if isinstance(speed, dict):
        inference_ms = safe_float(speed.get("inference"), digits=4)
        if inference_ms and inference_ms > 0:
            fps = round(1000.0 / inference_ms, 4)

    model_core = getattr(model, "model", None)
    params = int(get_num_params(model_core)) if model_core is not None else None
    gflops = safe_float(get_flops(model_core, imgsz=args.imgsz), digits=2) if model_core is not None else None
    weight_mb = round(weight_path.stat().st_size / (1024 * 1024), 2)

    return {
        "Model": weight_path.parent.parent.name,
        "WeightPath": str(weight_path),
        "P": p,
        "R": r,
        "mAP@50": map50,
        "mAP@50:95": map5095,
        "Parameters": params,
        "GFLOPS": gflops,
        "Weight(MB)": weight_mb,
        "FPS": fps,
        "Inference(ms/img)": inference_ms,
    }


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Model",
        "P",
        "R",
        "mAP@50",
        "mAP@50:95",
        "Parameters",
        "GFLOPS",
        "Weight(MB)",
        "FPS",
        "Inference(ms/img)",
        "WeightPath",
    ]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_txt_log(rows: list[dict[str, Any]], output_txt: Path, data_yaml: Path, args: argparse.Namespace) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "P", "R", "mAP@50", "mAP@50:95", "Parameters", "GFLOPS", "Weight(MB)", "FPS"]
    widths = {
        "Model": 22,
        "P": 10,
        "R": 10,
        "mAP@50": 10,
        "mAP@50:95": 12,
        "Parameters": 14,
        "GFLOPS": 10,
        "Weight(MB)": 12,
        "FPS": 10,
    }

    def pad(text: Any, key: str) -> str:
        return str(text).ljust(widths[key])

    lines = [
        "Model validation summary",
        f"Data YAML: {data_yaml}",
        f"Image Size: {args.imgsz}",
        f"Batch Size: {args.batch}",
        f"Device: {args.device}",
        "",
        " | ".join(pad(h, h) for h in headers),
        "-" * (sum(widths.values()) + 3 * (len(headers) - 1)),
    ]

    for row in rows:
        lines.append(
            " | ".join(
                [
                    pad(row["Model"], "Model"),
                    pad(format_number(row["P"]), "P"),
                    pad(format_number(row["R"]), "R"),
                    pad(format_number(row["mAP@50"]), "mAP@50"),
                    pad(format_number(row["mAP@50:95"]), "mAP@50:95"),
                    pad(row["Parameters"] if row["Parameters"] is not None else "N/A", "Parameters"),
                    pad(format_number(row["GFLOPS"], 2), "GFLOPS"),
                    pad(format_number(row["Weight(MB)"], 2), "Weight(MB)"),
                    pad(format_number(row["FPS"]), "FPS"),
                ]
            )
        )

    lines.append("")
    lines.append("Weight paths:")
    for row in rows:
        lines.append(f"- {row['Model']}: {row['WeightPath']}")

    output_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)

    temp_data_yaml = build_local_data_yaml(args.data)
    rows: list[dict[str, Any]] = []

    try:
        for weight in args.weights:
            print(f"\n[INFO] Validating: {weight}")
            row = validate_one_model(weight, temp_data_yaml, args)
            rows.append(row)
            print(
                "[INFO] Done: "
                f"{row['Model']} | P={format_number(row['P'])}, R={format_number(row['R'])}, "
                f"mAP50={format_number(row['mAP@50'])}, mAP50-95={format_number(row['mAP@50:95'])}, "
                f"FPS={format_number(row['FPS'])}"
            )

        write_csv(rows, args.csv)
        write_txt_log(rows, args.txt, temp_data_yaml, args)
        print(f"\n[INFO] CSV saved to: {args.csv}")
        print(f"[INFO] TXT log saved to: {args.txt}")
    finally:
        temp_data_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
