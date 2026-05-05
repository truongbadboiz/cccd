"""
propagate_labels.py — Copy annotation từ ảnh gốc sang ảnh augment

Chạy:
    python propagate_labels.py --source cccd_01_orig --detector field
    python propagate_labels.py --all --detector field
    python propagate_labels.py --all --detector all
"""

import os
import cv2
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_DIR    = "data/images/cccd_all"
CORNER_DIR = "data/annotations/corner"
FIELD_DIR  = "data/annotations/field"

ORIG_STEMS = [
    "cccd_01_orig",
    "cccd_02_orig",
    "cccd_03_orig",
    "cccd_04_orig",
    "cccd_05_orig",
]

# Mapping: stem gốc → suffix nhận dạng ảnh augment
# cccd_01_orig → augment có hậu tố "cccd00"
# cccd_03_orig → augment bắt đầu bằng "cccd_03"
STEM_TO_PATTERN = {
    "cccd_01_orig": ("endswith", "cccd00"),
    "cccd_02_orig": ("startswith", "cccd_02"),
    "cccd_03_orig": ("startswith", "cccd_03"),
    "cccd_04_orig": ("startswith", "cccd_04"),
    "cccd_05_orig": ("startswith", "cccd_05"),
}


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def parse_xml(xml_path):
    tree  = ET.parse(xml_path)
    root  = tree.getroot()
    sz    = root.find("size")
    img_w = int(sz.findtext("width",  "1"))
    img_h = int(sz.findtext("height", "1"))
    boxes = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip()
        bb   = obj.find("bndbox")
        boxes.append({
            "name": name,
            "xmin": float(bb.findtext("xmin")),
            "ymin": float(bb.findtext("ymin")),
            "xmax": float(bb.findtext("xmax")),
            "ymax": float(bb.findtext("ymax")),
        })
    return img_w, img_h, boxes


def make_xml(filename, img_w, img_h, boxes):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text    = "images"
    ET.SubElement(root, "filename").text  = filename
    ET.SubElement(root, "path").text      = filename
    src = ET.SubElement(root, "source")
    ET.SubElement(src, "database").text   = "Unknown"
    sz  = ET.SubElement(root, "size")
    ET.SubElement(sz, "width").text       = str(img_w)
    ET.SubElement(sz, "height").text      = str(img_h)
    ET.SubElement(sz, "depth").text       = "3"
    ET.SubElement(root, "segmented").text = "0"

    for b in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text       = b["name"]
        ET.SubElement(obj, "pose").text       = "Unspecified"
        ET.SubElement(obj, "truncated").text  = "0"
        ET.SubElement(obj, "difficult").text  = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(max(0, int(b["xmin"])))
        ET.SubElement(bb, "ymin").text = str(max(0, int(b["ymin"])))
        ET.SubElement(bb, "xmax").text = str(min(img_w, int(b["xmax"])))
        ET.SubElement(bb, "ymax").text = str(min(img_h, int(b["ymax"])))

    raw   = ET.tostring(root, encoding="unicode")
    lines = minidom.parseString(raw).toprettyxml(indent="  ").split("\n")
    lines = [l for l in lines if l.strip()]
    if lines[0].startswith("<?xml"):
        lines = lines[1:]
    return "\n".join(lines)


def save_xml(content, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Scale bbox
# ---------------------------------------------------------------------------

def scale_boxes(boxes, src_w, src_h, dst_w, dst_h):
    sx, sy = dst_w / src_w, dst_h / src_h
    return [{
        "name": b["name"],
        "xmin": b["xmin"] * sx,
        "ymin": b["ymin"] * sy,
        "xmax": b["xmax"] * sx,
        "ymax": b["ymax"] * sy,
    } for b in boxes]


# ---------------------------------------------------------------------------
# Tìm ảnh augment theo pattern
# ---------------------------------------------------------------------------

def find_augment_images(img_dir, source_stem):
    """
    Tìm tất cả ảnh augment tương ứng với ảnh gốc source_stem.
    Dùng STEM_TO_PATTERN để xác định pattern tìm kiếm.
    """
    pattern = STEM_TO_PATTERN.get(source_stem)
    if pattern is None:
        # Fallback: lấy 2 phần đầu làm prefix
        prefix = "_".join(source_stem.split("_")[:2])
        pattern = ("startswith", prefix)

    mode, value = pattern

    result = []
    for p in Path(img_dir).iterdir():
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if p.stem == source_stem:
            continue
        if mode == "startswith" and p.stem.startswith(value):
            result.append(p)
        elif mode == "endswith" and p.stem.endswith(value):
            result.append(p)

    return sorted(result)


# ---------------------------------------------------------------------------
# Propagate
# ---------------------------------------------------------------------------

def propagate(source_stem, detector, img_dir, corner_dir, field_dir):
    print(f"\n[INFO] Source: {source_stem}")

    detectors = []
    if detector in ("corner", "all"):
        detectors.append(("corner", corner_dir))
    if detector in ("field", "all"):
        detectors.append(("field", field_dir))

    for det_name, ann_dir in detectors:
        src_xml = os.path.join(ann_dir, source_stem + ".xml")
        if not os.path.exists(src_xml):
            print(f"  [SKIP-{det_name}] Không tìm thấy {src_xml}")
            print(f"                   → Hãy label ảnh gốc trong LabelImg trước")
            continue

        src_w, src_h, boxes = parse_xml(src_xml)
        print(f"  [{det_name}] {len(boxes)} bbox từ {os.path.basename(src_xml)}")

        img_paths = find_augment_images(img_dir, source_stem)
        print(f"  [{det_name}] Tìm thấy {len(img_paths)} ảnh augment")

        ok = skip = 0
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                skip += 1
                continue
            dst_h, dst_w = img.shape[:2]
            scaled   = scale_boxes(boxes, src_w, src_h, dst_w, dst_h)
            xml_str  = make_xml(img_path.name, dst_w, dst_h, scaled)
            out_path = os.path.join(ann_dir, img_path.stem + ".xml")
            save_xml(xml_str, out_path)
            ok += 1

        print(f"           → Propagated {ok} ảnh (skip {skip})")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   default=None,
                        help="Stem ảnh gốc, vd: cccd_01_orig")
    parser.add_argument("--all",      action="store_true",
                        help="Propagate tất cả ảnh gốc có XML")
    parser.add_argument("--detector", choices=["corner","field","all"],
                        default="field")
    parser.add_argument("--img_dir",    default=IMG_DIR)
    parser.add_argument("--corner_dir", default=CORNER_DIR)
    parser.add_argument("--field_dir",  default=FIELD_DIR)
    args = parser.parse_args()

    if not args.source and not args.all:
        print("[ERROR] Cần --source <stem> hoặc --all")
        parser.print_help()
        return

    if args.all:
        ann_dir = args.corner_dir if args.detector == "corner" else args.field_dir
        sources = [s for s in ORIG_STEMS
                   if os.path.exists(os.path.join(ann_dir, s + ".xml"))]
        missing = [s for s in ORIG_STEMS
                   if s not in sources]

        for m in missing:
            print(f"[WARN] Chưa có XML cho {m} — bỏ qua")

        if not sources:
            print("[ERROR] Không tìm thấy XML gốc nào.")
            return

        print(f"[INFO] Propagate {len(sources)} ảnh gốc: {sources}")
        for s in sources:
            propagate(s, args.detector,
                      args.img_dir, args.corner_dir, args.field_dir)
    else:
        propagate(args.source, args.detector,
                  args.img_dir, args.corner_dir, args.field_dir)

    print("\n[DONE] Kiểm tra lại trong LabelImg.")


if __name__ == "__main__":
    main()