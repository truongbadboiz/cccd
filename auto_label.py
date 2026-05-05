"""
auto_label_v2.py — Tự động detect vùng text và sinh XML annotation
Không cần hardcode tỷ lệ — dùng OpenCV MSER + morphology để tìm text lines,
sau đó map vào field dựa trên vị trí tương đối trong thẻ.

Chạy:
    python auto_label_v2.py --img_dir data/images/cccd_all --mode all
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom


# ---------------------------------------------------------------------------
# Field map theo vị trí Y (% chiều cao thẻ từ trên xuống)
# Mỗi field có khoảng Y tương ứng
# ---------------------------------------------------------------------------
# Dựa trên layout chuẩn CCCD chip:
#  0%  - 28% : header (không label)
#  28% - 38% : id
#  38% - 50% : name
#  50% - 58% : birth
#  58% - 66% : sex + nationality (cùng hàng)
#  66% - 76% : home
#  76% - 92% : address (2 dòng)
#  86% - 96% : expiry (góc trái)

FIELD_Y_RANGES = [
    # (y_start, y_end, field_name, x_split)
    # x_split: None = toàn chiều ngang | float = tách trái/phải
    (0.28, 0.38, "id",          None),
    (0.38, 0.50, "name",        None),
    (0.50, 0.58, "birth",       None),
    (0.58, 0.66, "sex",         0.45),   # trái = sex, phải = nationality
    (0.66, 0.76, "home",        None),
    (0.76, 0.93, "address",     None),
    (0.86, 0.96, "expiry",      0.35),   # chỉ phần trái = expiry
]

CORNER_SIZE_RATIO = 0.08


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def make_xml(filename, img_w, img_h, boxes):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text   = "images"
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text     = filename
    src = ET.SubElement(root, "source")
    ET.SubElement(src, "database").text  = "Unknown"
    sz = ET.SubElement(root, "size")
    ET.SubElement(sz, "width").text      = str(img_w)
    ET.SubElement(sz, "height").text     = str(img_h)
    ET.SubElement(sz, "depth").text      = "3"
    ET.SubElement(root, "segmented").text = "0"

    for b in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text      = b["name"]
        ET.SubElement(obj, "pose").text      = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(max(0, int(b["xmin"])))
        ET.SubElement(bb, "ymin").text = str(max(0, int(b["ymin"])))
        ET.SubElement(bb, "xmax").text = str(min(img_w, int(b["xmax"])))
        ET.SubElement(bb, "ymax").text = str(min(img_h, int(b["ymax"])))

    raw = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    lines = [l for l in pretty.split("\n") if l.strip()]
    if lines[0].startswith("<?xml"):
        lines = lines[1:]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detect card region
# ---------------------------------------------------------------------------

def detect_card(img):
    """Tìm bounding rect của thẻ CCCD trong ảnh."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu + Adaptive kết hợp
    _, t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t2     = cv2.adaptiveThreshold(gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)
    thresh = cv2.bitwise_and(t1, t2)

    # Morphology close để lấp lỗ hổng
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # Fallback: dùng toàn ảnh bỏ 5% margin
        mx, my = int(w*0.05), int(h*0.05)
        return mx, my, w-2*mx, h-2*my

    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < w * h * 0.15:
        mx, my = int(w*0.05), int(h*0.05)
        return mx, my, w-2*mx, h-2*my

    x, y, cw, ch = cv2.boundingRect(largest)
    pad = 4
    return (max(0, x-pad), max(0, y-pad),
            min(w-x, cw+2*pad), min(h-y, ch+2*pad))


# ---------------------------------------------------------------------------
# Detect text lines trong card đã warp (hoặc crop)
# ---------------------------------------------------------------------------

def detect_text_lines(card_img):
    """
    Dùng morphological operations để tìm các dòng text trong thẻ.
    Trả về list (x,y,w,h) của từng text line, sorted top-to-bottom.
    """
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

    # Denoise + threshold
    blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate theo chiều ngang để nối các ký tự thành dòng
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(bw, kh, iterations=2)

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    ch, cw = card_img.shape[:2]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Lọc: dòng text hợp lý (không quá nhỏ, không quá cao)
        if w < cw * 0.05 or h < 5 or h > ch * 0.15:
            continue
        lines.append((x, y, w, h))

    # Sort từ trên xuống
    lines.sort(key=lambda l: l[1])
    return lines


# ---------------------------------------------------------------------------
# Map text lines → field names dựa trên vị trí Y
# ---------------------------------------------------------------------------

def map_lines_to_fields(text_lines, card_h, card_w, card_x, card_y):
    """
    Với mỗi text line (tính trong hệ tọa độ card),
    xác định nó thuộc field nào dựa trên FIELD_Y_RANGES.
    Trả về list bbox dict với field name.
    """
    boxes = []
    used_fields = set()

    for (lx, ly, lw, lh) in text_lines:
        y_ratio = ly / card_h
        x_ratio = lx / card_w

        for (ys, ye, fname, x_split) in FIELD_Y_RANGES:
            if not (ys <= y_ratio <= ye):
                continue

            # Xử lý field chia trái/phải (sex/nationality, expiry)
            if x_split is not None:
                if fname == "sex":
                    if x_ratio < x_split:
                        actual_field = "sex"
                    else:
                        actual_field = "nationality"
                elif fname == "expiry":
                    if x_ratio < x_split:
                        actual_field = "expiry"
                    else:
                        continue   # bỏ phần phải của hàng expiry
                else:
                    actual_field = fname
            else:
                actual_field = fname

            # address có thể có nhiều dòng → gộp lại
            if actual_field == "address":
                # Tìm bbox address đã có rồi mở rộng
                existing = next((b for b in boxes
                                 if b["name"] == "address"), None)
                abs_xmin = card_x + lx
                abs_ymin = card_y + ly
                abs_xmax = card_x + lx + lw
                abs_ymax = card_y + ly + lh

                if existing:
                    existing["xmin"] = min(existing["xmin"], abs_xmin)
                    existing["ymin"] = min(existing["ymin"], abs_ymin)
                    existing["xmax"] = max(existing["xmax"], abs_xmax)
                    existing["ymax"] = max(existing["ymax"], abs_ymax)
                else:
                    boxes.append({"name": "address",
                                  "xmin": abs_xmin, "ymin": abs_ymin,
                                  "xmax": abs_xmax, "ymax": abs_ymax})
                break

            # Các field 1 dòng: lấy line có y_center gần nhất
            # Nếu đã có rồi → giữ lại dòng nào rộng hơn
            abs_xmin = card_x + lx
            abs_ymin = card_y + ly
            abs_xmax = card_x + lx + lw
            abs_ymax = card_y + ly + lh

            existing = next((b for b in boxes
                             if b["name"] == actual_field), None)
            if existing:
                # Giữ dòng rộng hơn
                if lw > (existing["xmax"] - existing["xmin"]):
                    existing.update({"xmin": abs_xmin, "ymin": abs_ymin,
                                     "xmax": abs_xmax, "ymax": abs_ymax})
            else:
                boxes.append({"name": actual_field,
                              "xmin": abs_xmin, "ymin": abs_ymin,
                              "xmax": abs_xmax, "ymax": abs_ymax})
            break

    # Padding nhỏ cho mỗi bbox
    pad = 4
    for b in boxes:
        b["xmin"] -= pad
        b["ymin"] -= pad
        b["xmax"] += pad
        b["ymax"] += pad

    return boxes


# ---------------------------------------------------------------------------
# Fallback: dùng tỷ lệ cố định nếu detect không được đủ field
# ---------------------------------------------------------------------------

FIELD_LAYOUT_FALLBACK = {
    "id":          (0.33, 0.30, 0.48, 0.08),
    "name":        (0.20, 0.40, 0.60, 0.09),
    "birth":       (0.46, 0.50, 0.32, 0.07),
    "sex":         (0.36, 0.58, 0.13, 0.07),
    "nationality": (0.65, 0.58, 0.28, 0.07),
    "home":        (0.20, 0.67, 0.68, 0.08),
    "address":     (0.20, 0.76, 0.68, 0.14),
    "expiry":      (0.02, 0.87, 0.28, 0.07),
}

REQUIRED_FIELDS = {"id", "name", "birth", "sex",
                   "nationality", "home", "address", "expiry"}


def fallback_boxes(card_x, card_y, card_w, card_h):
    boxes = []
    for field, (xr, yr, wr, hr) in FIELD_LAYOUT_FALLBACK.items():
        boxes.append({
            "name": field,
            "xmin": card_x + card_w * xr,
            "ymin": card_y + card_h * yr,
            "xmax": card_x + card_w * (xr + wr),
            "ymax": card_y + card_h * (yr + hr),
        })
    return boxes


# ---------------------------------------------------------------------------
# Corner boxes
# ---------------------------------------------------------------------------

def corner_boxes(card_x, card_y, card_w, card_h):
    size = max(20, int(min(card_w, card_h) * CORNER_SIZE_RATIO))
    return [
        {"name": "top_left",
         "xmin": card_x,          "ymin": card_y,
         "xmax": card_x+size*2,   "ymax": card_y+size},
        {"name": "bottom_left",
         "xmin": card_x,          "ymin": card_y+card_h-size,
         "xmax": card_x+size*2,   "ymax": card_y+card_h},
        {"name": "bottom_right",
         "xmin": card_x+card_w-size*2, "ymin": card_y+card_h-size,
         "xmax": card_x+card_w,        "ymax": card_y+card_h},
    ]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process(img_path, mode, corner_dir, field_dir):
    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    fname = Path(img_path).name
    stem  = Path(img_path).stem

    # Detect card region
    cx, cy, cw, ch = detect_card(img)
    card_crop = img[cy:cy+ch, cx:cx+cw]

    if mode in ("corner", "all"):
        boxes = corner_boxes(cx, cy, cw, ch)
        with open(os.path.join(corner_dir, stem+".xml"),
                  "w", encoding="utf-8") as f:
            f.write(make_xml(fname, w, h, boxes))

    if mode in ("field", "all"):
        # Bước 1: detect text lines trong card
        lines = detect_text_lines(card_crop)

        # Bước 2: map text lines → fields
        boxes = map_lines_to_fields(lines, ch, cw, cx, cy)

        # Bước 3: kiểm tra đủ field chưa — fallback nếu thiếu
        detected = {b["name"] for b in boxes}
        missing  = REQUIRED_FIELDS - detected
        if missing:
            # Bổ sung field còn thiếu từ fallback
            fb = {b["name"]: b for b in fallback_boxes(cx, cy, cw, ch)}
            for f in missing:
                boxes.append(fb[f])

        with open(os.path.join(field_dir, stem+".xml"),
                  "w", encoding="utf-8") as f:
            f.write(make_xml(fname, w, h, boxes))

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",    default="data/images/cccd_all")
    parser.add_argument("--corner_dir", default="data/annotations/corner")
    parser.add_argument("--field_dir",  default="data/annotations/field")
    parser.add_argument("--mode",
        choices=["corner","field","all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.corner_dir, exist_ok=True)
    os.makedirs(args.field_dir,  exist_ok=True)

    imgs = sorted([
        str(p) for p in Path(args.img_dir).iterdir()
        if p.suffix.lower() in (".jpg",".jpeg",".png")
    ])
    if args.limit:
        imgs = imgs[:args.limit]

    print(f"[INFO] {len(imgs)} ảnh | mode={args.mode}")
    ok = fail = 0
    for i, p in enumerate(imgs, 1):
        print(f"  [{i:3d}/{len(imgs)}] {Path(p).name}", end=" ")
        if process(p, args.mode, args.corner_dir, args.field_dir):
            print("✓"); ok += 1
        else:
            print("✗"); fail += 1

    print(f"\n[DONE] OK={ok} | FAIL={fail}")
    print("Mở LabelImg kiểm tra lại, sửa bbox lệch nếu có.")


if __name__ == "__main__":
    main()