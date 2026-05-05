"""
core/extractor.py — CCCD Extractor
Model:  YOLOv8 nano (ultralytics) — nhẹ, chạy tốt trên RTX 3050 4GB
OCR:    Tesseract + pytesseract  — hỗ trợ tiếng Việt (gói vie)

Pipeline:
  1. QR detect  → top_right anchor (cv2.QRCodeDetector)
  2. YOLO nano  → 3 góc còn lại   (corner_model.pt)
  3. Perspective warp → 856×540
  4. YOLO nano  → 8 field region  (field_model.pt)
  5. NMS
  6. Tesseract OCR từng crop (config tối ưu cho từng field)
  7. Post-process
"""

import os
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_CONFIG_ROOT = PROJECT_ROOT / ".cache"
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_ROOT))
YOLO_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

import pytesseract
from ultralytics import YOLO

from core.utils import (
    preprocess_for_detection,
    preprocess_crop_for_ocr,
    sharpen,
    infer_top_right_from_3_corners,
    perspective_transform,
    non_max_suppression,
    sort_boxes_top_to_bottom,
    validate_cccd_id,
    normalize_date,
    normalize_sex,
    clean_address_text,
    strip_field_label,
    cross_validate_id_province,
    CCCD_FIELDS,
    CCCD_WIDTH,
    CCCD_HEIGHT,
    CORNER_LABELS,
)

# ---------------------------------------------------------------------------
# Tesseract config
# ---------------------------------------------------------------------------
_TESS_BASE = "--oem 1 -l vie"

TESS_CONFIG = {
    "id":          f"{_TESS_BASE} --psm 7 -c tessedit_char_whitelist=0123456789",
    "name":        f"{_TESS_BASE} --psm 6",
    "birth":       f"{_TESS_BASE} --psm 7 -c tessedit_char_whitelist=0123456789/",
    "sex":         f"{_TESS_BASE} --psm 8",
    "nationality": f"{_TESS_BASE} --psm 7",
    "home":        f"{_TESS_BASE} --psm 6",
    "address":     f"{_TESS_BASE} --psm 6",
    "expiry":      f"{_TESS_BASE} --psm 7 -c tessedit_char_whitelist=0123456789/",
}

_MULTILINE_FIELDS = {"address", "home"}


def _save_debug_crop(label, raw, processed, debug_dir):
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{label}_raw.png"), raw)
    cv2.imwrite(str(debug_dir / f"{label}_processed.png"), processed)


class CCCDExtractor:
    """
    Trích xuất thông tin CCCD chip-based.

    Corner strategy:
        top_right    ← cv2.QRCodeDetector (cố định, fallback nội suy)
        top_left     ← YOLO class 0
        bottom_left  ← YOLO class 1
        bottom_right ← YOLO class 2
    """

    CORNER_CONF = 0.25   # hạ xuống để detect góc khó hơn
    FIELD_CONF  = 0.35

    def __init__(
        self,
        corner_model_path: str = "models/corner_model.pt",
        field_model_path:  str = "models/field_model.pt",
        tesseract_cmd:     Optional[str] = None,
        device:            str = "0",
        debug_dir:         Optional[str] = None,
    ):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif os.name == "nt":
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )

        print("[INFO] Loading corner model (YOLOv8)...")
        self.corner_model = YOLO(corner_model_path)

        print("[INFO] Loading field model (YOLOv8)...")
        self.field_model  = YOLO(field_model_path)

        self._device    = device
        self._qr        = cv2.QRCodeDetector()
        self._debug_dir = Path(debug_dir) if debug_dir else None
        print("[INFO] Ready. OCR: Tesseract")

    # ------------------------------------------------------------------
    # Bước 1 — QR → top_right
    # ------------------------------------------------------------------

    def _detect_qr_top_right(self, image: np.ndarray) -> tuple | None:
        """
        Quét QR, lấy đỉnh có x-y lớn nhất trong polygon.
        Thử thêm với ảnh resize nhỏ hơn và sharpen nếu thất bại.
        """
        # Thử 1: ảnh gốc
        pts = self._run_qr(image)

        # Thử 2: sharpen
        if pts is None:
            pts = self._run_qr(sharpen(image))

        # Thử 3: resize xuống để QR detector nhận tốt hơn với ảnh lớn
        if pts is None:
            h, w = image.shape[:2]
            scale = min(1.0, 1280 / max(h, w))
            if scale < 1.0:
                small = cv2.resize(image, (int(w * scale), int(h * scale)))
                pts_small = self._run_qr(small) or self._run_qr(sharpen(small))
                if pts_small is not None:
                    # Scale điểm trở lại kích thước gốc
                    pts = [p / scale for p in pts_small]

        if pts is None:
            return None

        best, best_score = None, float("-inf")
        for p in pts:
            p = np.array(p).reshape(-1, 2)
            if len(p) < 4:
                continue
            scores = p[:, 0] - p[:, 1]
            tr = p[np.argmax(scores)]
            s  = float(tr[0] - tr[1])
            if s > best_score:
                best_score, best = s, tr

        return (float(best[0]), float(best[1])) if best is not None else None

    def _run_qr(self, image: np.ndarray):
        try:
            ret = self._qr.detectAndDecodeMulti(image)
            ok, _, pts, _ = ret if len(ret) == 4 else (*ret, None)
        except Exception:
            ok, pts = self._qr.detect(image)
            pts = [pts] if ok and pts is not None else None
        return pts if ok and pts is not None and len(pts) > 0 else None

    # ------------------------------------------------------------------
    # Bước 2 — YOLO → 3 góc
    # ------------------------------------------------------------------

    def _detect_3_corners(self, image: np.ndarray) -> dict:
        results = self.corner_model.predict(
            source=image,
            conf=self.CORNER_CONF,
            device=self._device,
            verbose=False,
            imgsz=640,
        )[0]

        best = {}
        if results.boxes is not None:
            for box in results.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls < 0 or cls >= len(CORNER_LABELS):
                    continue
                name = CORNER_LABELS[cls]
                if name not in best or conf > best[name][2]:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    best[name] = (cx, cy, conf)

        return {k: v[:2] for k, v in best.items()}

    # ------------------------------------------------------------------
    # Bước 3 — Warp perspective
    # ------------------------------------------------------------------

    def _warp_card(self, image: np.ndarray) -> np.ndarray:
        image   = preprocess_for_detection(image)
        qr_tr   = self._detect_qr_top_right(image)
        corners = self._detect_3_corners(image)

        # Bước 3a: Set top_right TRƯỚC từ QR hoặc nội suy
        if qr_tr is not None:
            corners["top_right"] = qr_tr
            print(f"[INFO] QR detected → top_right={qr_tr}")
        else:
            # Cần đủ 3 góc để nội suy top_right
            if all(c in corners for c in ("top_left", "bottom_left", "bottom_right")):
                corners["top_right"] = infer_top_right_from_3_corners(corners)
                print("[WARN] QR không detect — nội suy top_right từ 3 góc")
            else:
                # Chưa đủ góc để nội suy, xử lý ở bước 3b
                corners["top_right"] = None

        # Bước 3b: Kiểm tra 3 góc còn lại, nội suy nếu thiếu 1
        missing = [c for c in ("top_left", "bottom_left", "bottom_right")
                   if c not in corners]

        if len(missing) >= 2:
            raise ValueError(
                f"Không detect được góc {missing}. "
                "Ảnh quá mờ hoặc CCCD bị che khuất."
            )
        elif len(missing) == 1:
            print(f"[WARN] Thiếu góc {missing[0]} → nội suy")
            tl = corners.get("top_left")
            bl = corners.get("bottom_left")
            br = corners.get("bottom_right")
            tr = corners.get("top_right")

            if missing[0] == "top_left" and bl and br and tr:
                corners["top_left"] = (
                    tr[0] + bl[0] - br[0],
                    tr[1] + bl[1] - br[1]
                )
            elif missing[0] == "bottom_left" and tl and br and tr:
                corners["bottom_left"] = (
                    tl[0] + br[0] - tr[0],
                    tl[1] + br[1] - tr[1]
                )
            elif missing[0] == "bottom_right" and tl and bl and tr:
                corners["bottom_right"] = (
                    tr[0] + bl[0] - tl[0],
                    tr[1] + bl[1] - tl[1]
                )
            else:
                raise ValueError(f"Không thể nội suy góc {missing[0]}.")

            # Nếu top_right vẫn None (QR không detect được + thiếu 1 góc)
            # → nội suy lại top_right bây giờ khi đã đủ 3 góc
            if corners.get("top_right") is None:
                corners["top_right"] = infer_top_right_from_3_corners(corners)
                print("[WARN] Nội suy lại top_right sau khi bổ sung góc thiếu")

        # Bước 3c: Warp
        src = np.float32([
            corners["top_left"],
            corners["top_right"],
            corners["bottom_right"],
            corners["bottom_left"],
        ])
        return perspective_transform(image, src)

    # ------------------------------------------------------------------
    # Bước 4+5 — YOLO field detect + NMS
    # ------------------------------------------------------------------

    def _detect_fields(self, card: np.ndarray) -> tuple:
        results = self.field_model.predict(
            source=card,
            conf=self.FIELD_CONF,
            device=self._device,
            verbose=False,
            imgsz=640,
        )[0]

        raw_boxes, raw_labels = [], []
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls < 0 or cls >= len(CCCD_FIELDS):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw_boxes.append([int(y1), int(x1), int(y2), int(x2)])
                raw_labels.append(CCCD_FIELDS[cls])

        if not raw_boxes:
            return np.array([]), []

        boxes, labels = non_max_suppression(
            np.array(raw_boxes), raw_labels, overlap_thresh=0.3)
        return sort_boxes_top_to_bottom(boxes, labels)

    # ------------------------------------------------------------------
    # Bước 6 — Tesseract OCR
    # ------------------------------------------------------------------

    def _ocr_fields(self, card: np.ndarray,
                    boxes: np.ndarray, labels: list) -> dict:
        field_texts: dict[str, list[str]] = {f: [] for f in CCCD_FIELDS}
        pad = 5

        for box, label in zip(boxes, labels):
            ymin, xmin, ymax, xmax = box
            ymin = max(0, ymin - pad)
            xmin = max(0, xmin - pad)
            ymax = min(card.shape[0], ymax + pad)
            xmax = min(card.shape[1], xmax + pad)

            crop_bgr = card[ymin:ymax, xmin:xmax]
            if crop_bgr.size == 0:
                continue

            processed = preprocess_crop_for_ocr(
                crop_bgr,
                field=label,
                large_block=(label in _MULTILINE_FIELDS),
            )

            _save_debug_crop(label, crop_bgr, processed, self._debug_dir)

            pil_img = Image.fromarray(processed)
            cfg     = TESS_CONFIG.get(label, f"{_TESS_BASE} --psm 6")

            try:
                text = pytesseract.image_to_string(pil_img, config=cfg)
                text = text.strip().replace("\n", " ").replace("\f", "")
                field_texts[label].append(text)
            except Exception as e:
                print(f"[WARN] Tesseract lỗi field={label}: {e}")

        return {
            f: " ".join(t).strip() for f, t in field_texts.items()
        }

    # ------------------------------------------------------------------
    # Bước 7 — Post-process
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess(raw: dict) -> dict:
        out = dict(raw)

        if out.get("id"):
            out["id"] = validate_cccd_id(out["id"])

        if out.get("birth"):
            out["birth"] = normalize_date(out["birth"], field="birth")

        if out.get("expiry"):
            out["expiry"] = normalize_date(out["expiry"], field="expiry")

        if out.get("sex"):
            out["sex"] = normalize_sex(out["sex"])

        if not out.get("nationality"):
            out["nationality"] = "Việt Nam"

        if out.get("name"):
            out["name"] = re.sub(r"[\s\-_.,/\\|]+$", "", out["name"]).strip()

        if out.get("address"):
            out["address"] = clean_address_text(out["address"])
        if out.get("home"):
            out["home"] = clean_address_text(out["home"])

        if not out.get("home") and out.get("address"):
            lines = [ln.strip() for ln in out["address"].split("\n") if ln.strip()]
            if len(lines) >= 2:
                out["home"]    = lines[0]
                out["address"] = " ".join(lines[1:])

        if out.get("id") and len(out["id"]) == 12:
            out["id"] = cross_validate_id_province(
                out["id"],
                out.get("home", ""),
                out.get("address", ""),
            )

        return {
            k: v.strip() if isinstance(v, str) else v
            for k, v in out.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Không đọc được: {image_path}")
        return self.extract_from_array(img)

    def extract_from_array(self, image: np.ndarray) -> dict:
        card          = self._warp_card(image)
        boxes, labels = self._detect_fields(card)
        if len(boxes) == 0:
            raise ValueError("Không detect được vùng text.")
        raw = self._ocr_fields(card, boxes, labels)
        return self._postprocess(raw)