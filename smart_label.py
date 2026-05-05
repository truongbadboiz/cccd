"""
smart_label.py — Tự động sinh XML annotation chính xác
Dùng corner_model.pt đã train để warp thẻ về 856×540,
sau đó apply layout CCCD chuẩn → bbox chính xác hơn nhiều.

Chạy:
    python smart_label.py               # label field cho 5 ảnh gốc
    python smart_label.py --all         # label field cho toàn bộ ảnh
    python smart_label.py --mode corner # label corner
"""

import os
import cv2
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_DIR      = "data/images/cccd_all"
FIELD_DIR    = "data/annotations/field"
CORNER_DIR   = "data/annotations/corner"
CORNER_MODEL = "models/corner_model.pt"

CCCD_W, CCCD_H = 856, 540   # kích thước chuẩn sau warp

# Layout CCCD chuẩn (tính trên ảnh đã warp 856×540)
# (x_start%, y_start%, width%, height%) — tính theo CCCD_W, CCCD_H
FIELD_LAYOUT = {
    "id":          (0.355, 0.295, 0.460, 0.100),
    "name":        (0.195, 0.400, 0.620, 0.095),
    "birth":       (0.470, 0.500, 0.325, 0.080),
    "sex":         (0.340, 0.580, 0.140, 0.080),
    "nationality": (0.630, 0.580, 0.310, 0.080),
    "home":        (0.195, 0.665, 0.690, 0.085),
    "address":     (0.195, 0.755, 0.690, 0.155),  # 2 dòng → cao hơn
    "expiry":      (0.020, 0.855, 0.275, 0.080),
}

CORNER_LABELS = ["top_left", "bottom_left", "bottom_right"]
CORNER_SIZE   = 0.08   # % cạnh ngắn thẻ


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def write_xml(out_path, filename, img_w, img_h, boxes):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text    = "images"
    ET.SubElement(root, "filename").text  = filename
    ET.SubElement(root, "path").text      = filename
    src = ET.SubElement(root, "source")
    ET.SubElement(src, "database").text   = "Unknown"
    sz = ET.SubElement(root, "size")
    ET.SubElement(sz, "width").text       = str(img_w)
    ET.SubElement(sz, "height").text      = str(img_h)
    ET.SubElement(sz, "depth").text       = "3"
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

    raw    = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    lines  = [l for l in pretty.split("\n") if l.strip()]
    if lines[0].startswith("<?xml"):
        lines = lines[1:]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Detect corners bằng YOLO + QR
# ---------------------------------------------------------------------------

def detect_corners_yolo(img, model):
    """Dùng corner_model detect 3 góc."""
    results = model.predict(
        source=img, conf=0.35, verbose=False, imgsz=640
    )[0]
    corners = {}
    if results.boxes is not None:
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls >= len(CORNER_LABELS):
                continue
            name = CORNER_LABELS[cls]
            if name not in corners or conf > corners[name][2]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                corners[name] = ((x1+x2)/2, (y1+y2)/2, conf)
    return {k: v[:2] for k, v in corners.items()}


def detect_qr_top_right(img):
    """Detect QR code, lấy đỉnh top-right."""
    qr = cv2.QRCodeDetector()
    try:
        ret = qr.detectAndDecodeMulti(img)
        ok, _, pts, _ = ret if len(ret) == 4 else (*ret, None)
    except Exception:
        ok, pts = qr.detect(img)
        pts = [pts] if ok and pts is not None else None

    if not ok or pts is None or len(pts) == 0:
        return None

    best, best_score = None, float("-inf")
    for p in pts:
        p = p.reshape(-1, 2)
        if len(p) < 4:
            continue
        scores = p[:, 0] - p[:, 1]
        tr = p[np.argmax(scores)]
        s  = float(tr[0] - tr[1])
        if s > best_score:
            best_score, best = s, tr

    return (float(best[0]), float(best[1])) if best is not None else None


def infer_top_right(corners):
    """Nội suy top_right từ 3 góc (hình bình hành)."""
    tl = corners["top_left"]
    bl = corners["bottom_left"]
    br = corners["bottom_right"]
    return (tl[0] + br[0] - bl[0], tl[1] + br[1] - bl[1])


# ---------------------------------------------------------------------------
# Warp perspective
# ---------------------------------------------------------------------------

def warp_card(img, model):
    """
    Warp ảnh về 856×540 dùng corner model + QR.
    Trả về (warped_img, M_inv) để map bbox ngược lại.
    M_inv: ma trận inverse perspective transform.
    """
    h, w = img.shape[:2]

    # Resize nếu ảnh quá nhỏ
    if max(h, w) < 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        h, w = img.shape[:2]

    corners = detect_corners_yolo(img, model)
    missing = [c for c in CORNER_LABELS if c not in corners]
    if missing:
        return None, None, None

    qr_tr = detect_qr_top_right(img)
    if qr_tr is None:
        qr_tr = infer_top_right(corners)

    src = np.float32([
        corners["top_left"],
        qr_tr,
        corners["bottom_right"],
        corners["bottom_left"],
    ])
    dst = np.float32([
        [0,        0],
        [CCCD_W,   0],
        [CCCD_W,   CCCD_H],
        [0,        CCCD_H],
    ])

    M     = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (CCCD_W, CCCD_H))

    return warped, M_inv, img.shape[:2]


# ---------------------------------------------------------------------------
# Sinh bbox field trên ảnh gốc (map từ warped → original)
# ---------------------------------------------------------------------------

def layout_boxes_on_original(M_inv, orig_h, orig_w):
    """
    Tính bbox của từng field trên ảnh gốc bằng cách map
    4 điểm góc của bbox trong warped space → original space.
    """
    boxes = []
    for field, (xr, yr, wr, hr) in FIELD_LAYOUT.items():
        # Bbox trong warped space
        x1_w = CCCD_W * xr
        y1_w = CCCD_H * yr
        x2_w = CCCD_W * (xr + wr)
        y2_w = CCCD_H * (yr + hr)

        # 4 điểm góc
        pts_warped = np.float32([
            [x1_w, y1_w],
            [x2_w, y1_w],
            [x2_w, y2_w],
            [x1_w, y2_w],
        ]).reshape(-1, 1, 2)

        # Map ngược về original
        pts_orig = cv2.perspectiveTransform(pts_warped, M_inv)
        pts_orig = pts_orig.reshape(-1, 2)

        xmin = max(0,       float(pts_orig[:, 0].min()) - 5)
        ymin = max(0,       float(pts_orig[:, 1].min()) - 5)
        xmax = min(orig_w,  float(pts_orig[:, 0].max()) + 5)
        ymax = min(orig_h,  float(pts_orig[:, 1].max()) + 5)

        boxes.append({
            "name": field,
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax,
        })

    return boxes


# ---------------------------------------------------------------------------
# Fallback khi không warp được (dùng tỷ lệ trên ảnh gốc)
# ---------------------------------------------------------------------------

def fallback_boxes_orig(img_w, img_h):
    """
    Estimate vùng thẻ trong ảnh → apply layout cố định.
    """
    # Giả sử thẻ bắt đầu từ 5% mép, chiếm 90% ảnh
    mx = int(img_w * 0.05)
    my = int(img_h * 0.05)
    cw = int(img_w * 0.90)
    ch = int(img_h * 0.90)

    boxes = []
    for field, (xr, yr, wr, hr) in FIELD_LAYOUT.items():
        boxes.append({
            "name": field,
            "xmin": mx + cw * xr,
            "ymin": my + ch * yr,
            "xmax": mx + cw * (xr + wr),
            "ymax": my + ch * (yr + hr),
        })
    return boxes


# ---------------------------------------------------------------------------
# Sinh corner boxes (cho label corner)
# ---------------------------------------------------------------------------

def corner_boxes_orig(img_w, img_h):
    mx = int(img_w * 0.05)
    my = int(img_h * 0.05)
    cw = int(img_w * 0.90)
    ch = int(img_h * 0.90)
    size = max(20, int(min(cw, ch) * CORNER_SIZE))
    return [
        {"name": "top_left",
         "xmin": mx,       "ymin": my,
         "xmax": mx+size*2,"ymax": my+size},
        {"name": "bottom_left",
         "xmin": mx,           "ymin": my+ch-size,
         "xmax": mx+size*2,    "ymax": my+ch},
        {"name": "bottom_right",
         "xmin": mx+cw-size*2, "ymin": my+ch-size,
         "xmax": mx+cw,        "ymax": my+ch},
    ]


# ---------------------------------------------------------------------------
# Process 1 ảnh
# ---------------------------------------------------------------------------

def process_image(img_path, mode, model, corner_dir, field_dir):
    img = cv2.imread(img_path)
    if img is None:
        return False, "Không đọc được ảnh"

    h, w   = img.shape[:2]
    fname  = Path(img_path).name
    stem   = Path(img_path).stem

    if mode in ("corner", "all"):
        boxes = corner_boxes_orig(w, h)
        write_xml(str(Path(corner_dir) / f"{stem}.xml"),
                  fname, w, h, boxes)

    if mode in ("field", "all"):
        method = "warp"
        warped, M_inv, orig_shape = warp_card(img, model)

        if warped is not None and M_inv is not None:
            orig_h, orig_w = orig_shape
            boxes = layout_boxes_on_original(M_inv, orig_h, orig_w)
        else:
            # Fallback nếu không detect được góc
            method = "fallback"
            boxes  = fallback_boxes_orig(w, h)

        write_xml(str(Path(field_dir) / f"{stem}.xml"),
                  fname, w, h, boxes)
        return True, method

    return True, "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["field", "corner", "all"],
                        default="field")
    parser.add_argument("--all", action="store_true",
                        help="Label toàn bộ ảnh (default: chỉ 5 ảnh gốc)")
    parser.add_argument("--img_dir",    default=IMG_DIR)
    parser.add_argument("--field_dir",  default=FIELD_DIR)
    parser.add_argument("--corner_dir", default=CORNER_DIR)
    parser.add_argument("--model",      default=CORNER_MODEL)
    args = parser.parse_args()

    # Load corner model
    if args.mode in ("field", "all"):
        if not Path(args.model).exists():
            print(f"[ERR] Không tìm thấy corner model: {args.model}")
            print("      Hãy train corner model trước.")
            return
        from ultralytics import YOLO
        print(f"[INFO] Loading corner model: {args.model}")
        model = YOLO(args.model)
    else:
        model = None

    os.makedirs(args.field_dir,  exist_ok=True)
    os.makedirs(args.corner_dir, exist_ok=True)

    # Chọn ảnh cần label
    if args.all:
        imgs = sorted([
            str(p) for p in Path(args.img_dir).iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        print(f"[INFO] Label toàn bộ {len(imgs)} ảnh")
    else:
        # Chỉ 5 ảnh gốc
        orig_names = [
            "cccd_01_orig.jpg", "cccd_02_orig.jpg", "cccd_03_orig.jpg",
            "cccd_04_orig.jpg", "cccd_05_orig.jpg",
        ]
        imgs = [str(Path(args.img_dir) / n) for n in orig_names
                if (Path(args.img_dir) / n).exists()]
        print(f"[INFO] Label {len(imgs)} ảnh gốc")

    if not imgs:
        print("[ERR] Không tìm thấy ảnh nào.")
        return

    ok = fail = warp_ok = fallback_count = 0
    for i, img_path in enumerate(imgs, 1):
        name = Path(img_path).name
        print(f"  [{i:3d}/{len(imgs)}] {name}", end=" ... ")

        success, method = process_image(
            img_path, args.mode, model,
            args.corner_dir, args.field_dir,
        )
        if success:
            ok += 1
            if method == "warp":
                warp_ok += 1
                print(f"✓ (warp)")
            elif method == "fallback":
                fallback_count += 1
                print(f"~ (fallback)")
            else:
                print(f"✓")
        else:
            fail += 1
            print(f"✗ {method}")

    print(f"\n{'='*50}")
    print(f"DONE: {ok} OK | {fail} FAIL")
    if args.mode in ("field", "all"):
        print(f"  Warp chính xác : {warp_ok}")
        print(f"  Fallback layout: {fallback_count}")
    print()
    if not args.all:
        print("Bước tiếp theo:")
        print("  1. Mở LabelImg kiểm tra nhanh 5 ảnh gốc")
        print("  2. python propagate_labels_v2.py --detector field")
        print("  3. python -m training.prepare_dataset ...")
        print("  4. python -m training.train ...")
    else:
        print("Bước tiếp theo:")
        print("  1. python -m training.prepare_dataset ...")
        print("  2. python -m training.train ...")


if __name__ == "__main__":
    main()