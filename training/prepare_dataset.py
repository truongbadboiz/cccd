"""
training/prepare_dataset.py — Chuyển annotation LabelImg (Pascal VOC XML)
sang định dạng YOLO để train bằng ultralytics YOLOv8.

Chạy:
    python -m training.prepare_dataset --detector corner \
        --input_dir data/annotations/corner \
        --img_dir   data/images \
        --output_dir data/labels/corner

    python -m training.prepare_dataset --detector field \
        --input_dir data/annotations/field \
        --img_dir   data/images \
        --output_dir data/labels/field
"""

import argparse
import glob
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Label maps ──────────────────────────────────────────────────────────────

CORNER_LABEL_MAP = {
    "top_left":     0,
    "bottom_left":  1,
    "bottom_right": 2,
    # top_right KHÔNG có — dùng QR detection
}

FIELD_LABEL_MAP = {
    "id":          0,
    "name":        1,
    "birth":       2,
    "sex":         3,
    "nationality": 4,
    "home":        5,
    "address":     6,
    "expiry":      7,
}


def parse_voc_xml(xml_path: str, label_map: dict):
    """
    Parse file XML LabelImg → list of (class_id, cx, cy, w, h) normalized.
    YOLO format: tọa độ chuẩn hóa về [0,1] theo kích thước ảnh.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.findtext("width",  default="1"))
    img_h = int(size.findtext("height", default="1"))

    annotations = []
    for obj in root.iter("object"):
        label = obj.findtext("name", "").strip()
        if label not in label_map:
            print(f"  [SKIP] class '{label}' không có trong label_map")
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))

        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        annotations.append((label_map[label], cx, cy, bw, bh))

    return annotations, root.findtext("filename", "")


def convert(input_dir: str, img_dir: str, output_dir: str,
            label_map: dict, train_ratio: float = 0.85):
    """
    Chuyển toàn bộ XML → YOLO txt, đồng thời copy ảnh vào
    cấu trúc thư mục YOLOv8:
        output_dir/
            images/train/
            images/val/
            labels/train/
            labels/val/
    """
    xml_files = sorted(glob.glob(os.path.join(input_dir, "*.xml")))
    if not xml_files:
        print(f"[ERROR] Không tìm thấy file XML trong {input_dir}")
        return

    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * train_ratio)
    splits = {
        "train": xml_files[:split_idx],
        "val":   xml_files[split_idx:],
    }

    for split, files in splits.items():
        img_out = os.path.join(output_dir, "images", split)
        lbl_out = os.path.join(output_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        ok = skip = 0
        for xml_path in files:
            annotations, filename = parse_voc_xml(xml_path, label_map)
            if not annotations:
                skip += 1
                continue

            # Tìm file ảnh
            stem    = Path(xml_path).stem
            img_src = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                candidate = os.path.join(img_dir, stem + ext)
                if os.path.exists(candidate):
                    img_src = candidate
                    break

            if img_src is None:
                print(f"  [SKIP] Không tìm thấy ảnh cho: {stem}")
                skip += 1
                continue

            # Copy ảnh
            img_dst = os.path.join(img_out, os.path.basename(img_src))
            shutil.copy2(img_src, img_dst)

            # Ghi file label YOLO
            lbl_dst = os.path.join(lbl_out, stem + ".txt")
            with open(lbl_dst, "w", encoding="utf-8") as f:
                for cls_id, cx, cy, bw, bh in annotations:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            ok += 1

        print(f"[{split:5s}] {ok} files → {img_out}  (skip {skip})")

    # Ghi file data.yaml cho YOLOv8
    yaml_path = os.path.join(output_dir, "data.yaml")
    names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n")
        f.write(f"nc: {len(label_map)}\n")
        f.write(f"names: {names}\n")
    print(f"[YAML] {yaml_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector",   choices=["corner","field"], required=True)
    parser.add_argument("--input_dir",  required=True, help="Thư mục chứa file XML")
    parser.add_argument("--img_dir",    required=True, help="Thư mục chứa ảnh gốc")
    parser.add_argument("--output_dir", required=True, help="Thư mục output YOLO")
    parser.add_argument("--train_ratio",type=float, default=0.85)
    args = parser.parse_args()

    label_map = CORNER_LABEL_MAP if args.detector == "corner" else FIELD_LABEL_MAP
    print(f"[INFO] Detector: {args.detector} | {len(label_map)} classes: {list(label_map.keys())}")
    if args.detector == "corner":
        print("[INFO] top_right KHÔNG có trong label map — dùng QR detection")

    convert(args.input_dir, args.img_dir, args.output_dir,
            label_map, args.train_ratio)


if __name__ == "__main__":
    main()