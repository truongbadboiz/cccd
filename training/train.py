"""
training/train.py — Train YOLOv8 nano cho corner và field detector

Chạy:
    # Train corner model
    python -m training.train --detector corner --data data/labels/corner/data.yaml

    # Train field model
    python -m training.train --detector field  --data data/labels/field/data.yaml

Kết quả lưu tại:
    runs/detect/corner_cccd/weights/best.pt   → copy vào models/corner_model.pt
    runs/detect/field_cccd/weights/best.pt    → copy vào models/field_model.pt
"""

import argparse
import shutil
import os
from pathlib import Path
from ultralytics import YOLO


# ── Cấu hình train tối ưu cho RTX 3050 4GB ──────────────────────────────────
TRAIN_CONFIG = {
    # Model nano — nhỏ nhất, đủ dùng cho bài toán có ít class
    "model":     "yolov8n.pt",

    # Epochs: 100 là đủ cho fine-tune với ~400 ảnh
    "epochs":    100,

    # Image size: 640 chuẩn YOLOv8, vừa VRAM 3050
    "imgsz":     640,

    # Batch size: 8 an toàn cho 4GB VRAM
    # Nếu CUDA out of memory → giảm xuống 4
    "batch":     8,

    # Workers: 4 cho Windows, 8 cho Linux
    "workers":   4,

    # Augmentation mạnh (bù thiếu data)
    "augment":   True,
    "degrees":   15.0,    # xoay ngẫu nhiên ±15°
    "flipud":    0.0,     # không flip dọc (CCCD có chiều cố định)
    "fliplr":    0.3,     # flip ngang 30%
    "mosaic":    0.5,     # mosaic augmentation
    "mixup":     0.1,     # mixup nhẹ

    # Optimizer
    "optimizer": "AdamW",
    "lr0":       0.001,
    "lrf":       0.01,    # lr cuối = lr0 * lrf

    # Early stopping
    "patience":  30,      # dừng nếu 30 epoch không cải thiện

    # Device
    "device":    "0",     # GPU 0 (RTX 3050)
                          # đổi thành "cpu" nếu muốn train CPU
    # Verbose
    "verbose":   True,
}


def train(detector: str, data_yaml: str, resume: bool = False):
    """
    Fine-tune YOLOv8n từ pretrained COCO checkpoint.

    Args:
        detector: "corner" hoặc "field"
        data_yaml: đường dẫn file data.yaml
        resume:    tiếp tục train từ lần trước
    """
    nc = 3 if detector == "corner" else 8
    project_name = f"{detector}_cccd"

    print(f"\n{'='*50}")
    print(f"Train {detector} model — {nc} classes")
    print(f"Data: {data_yaml}")
    print(f"Config: YOLOv8n | {TRAIN_CONFIG['epochs']} epochs | batch={TRAIN_CONFIG['batch']}")
    print(f"Device: RTX 3050 (GPU {TRAIN_CONFIG['device']})")
    print(f"{'='*50}\n")

    model = YOLO(TRAIN_CONFIG["model"])

    results = model.train(
        data=data_yaml,
        project="runs/detect",
        name=project_name,
        exist_ok=True,
        resume=resume,
        **{k: v for k, v in TRAIN_CONFIG.items() if k != "model"},
    )

    # Copy best.pt vào thư mục models/
    best_pt  = Path(f"runs/detect/{project_name}/weights/best.pt")
    dest_pt  = Path(f"models/{detector}_model.pt")
    os.makedirs("models", exist_ok=True)

    if best_pt.exists():
        shutil.copy2(best_pt, dest_pt)
        print(f"\n[OK] Model saved → {dest_pt}")
    else:
        print(f"\n[WARN] Không tìm thấy {best_pt}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["corner","field"], required=True)
    parser.add_argument("--data",     required=True, help="Đường dẫn data.yaml")
    parser.add_argument("--resume",   action="store_true", help="Tiếp tục train")
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--batch",    type=int, default=None)
    parser.add_argument("--device",   default=None, help="0=GPU, cpu=CPU")
    args = parser.parse_args()

    if args.epochs: TRAIN_CONFIG["epochs"] = args.epochs
    if args.batch:  TRAIN_CONFIG["batch"]  = args.batch
    if args.device: TRAIN_CONFIG["device"] = args.device

    train(args.detector, args.data, args.resume)


if __name__ == "__main__":
    main()