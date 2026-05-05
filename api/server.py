"""
api/server.py — FastAPI CCCD Extractor
Model: YOLOv8 nano | OCR: Tesseract
Port:  8002
"""

import base64
import os
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.extractor import CCCDExtractor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CORNER_MODEL = os.getenv("CORNER_MODEL", "models/corner_model.pt")
FIELD_MODEL  = os.getenv("FIELD_MODEL",  "models/field_model.pt")
TESS_CMD     = os.getenv("TESSERACT_CMD", None)   # None = auto detect
PORT         = int(os.getenv("PORT", 8002))
DEVICE       = os.getenv("DEVICE", "0")           # "0"=GPU, "cpu"=CPU

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CCCD Extractor — YOLOv8 + Tesseract",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

extractor: CCCDExtractor = None
_ready = False
_error = ""


@app.on_event("startup")
async def startup():
    global extractor, _ready, _error

    missing = []
    if not os.path.isfile(CORNER_MODEL): missing.append(f"corner: {CORNER_MODEL}")
    if not os.path.isfile(FIELD_MODEL):  missing.append(f"field:  {FIELD_MODEL}")

    if missing:
        _error = "Chưa có model: " + " | ".join(missing)
        print(f"[WARN]   {_error}")
        print(f"[SERVER] Chạy training trước, sau đó restart.")
        print(f"[SERVER] Port {PORT} — chỉ /health hoạt động.")
        return

    try:
        extractor = CCCDExtractor(
            corner_model_path=CORNER_MODEL,
            field_model_path=FIELD_MODEL,
            tesseract_cmd=TESS_CMD,
            device=DEVICE,
        )
        _ready = True
        print(f"[SERVER] Sẵn sàng — port {PORT}.")
    except Exception as e:
        _error = str(e)
        print(f"[ERROR] {e}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CCCDResult(BaseModel):
    """8 field thuần String — Java parse Map<String,String> không lỗi."""
    id:          str = ""
    name:        str = ""
    birth:       str = ""
    sex:         str = ""
    nationality: str = ""
    home:        str = ""
    address:     str = ""
    expiry:      str = ""


class CCCDResultFull(CCCDResult):
    elapsed_ms: float = 0.0


class CCCDResultVi(BaseModel):
    so_cccd:        str = ""
    ho_va_ten:      str = ""
    ngay_sinh:      str = ""
    gioi_tinh:      str = ""
    quoc_tich:      str = ""
    que_quan:       str = ""
    noi_thuong_tru: str = ""
    co_gia_tri_den: str = ""
    elapsed_ms:     float = 0.0


class Base64Request(BaseModel):
    image_base64: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check():
    if not _ready:
        raise HTTPException(503, detail=f"Model chưa sẵn sàng. {_error}")


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không decode được ảnh.")
    return img


def _run(data: bytes):
    import cv2
    img  = _decode(data)
    t0   = time.time()

    # Debug warp
    card = extractor._warp_card(img)
    cv2.imwrite("debug_warp.jpg", card)

    # Debug field detection
    boxes, labels = extractor._detect_fields(card)
    print(f"[DEBUG] Field boxes: {len(boxes)}")
    print(f"[DEBUG] Labels: {labels}")

    r = extractor.extract_from_array(img)
    return r, time.time() - t0


def _validate_ext(filename):
    if not filename: return
    ext = filename.rsplit(".",1)[-1].lower() if "." in filename else ""
    if ext not in ("jpg","jpeg","png","bmp","webp"):
        raise HTTPException(400, f"Định dạng không hỗ trợ: .{ext}")


def _vi(r, e):
    return CCCDResultVi(
        so_cccd=r.get("id",""), ho_va_ten=r.get("name",""),
        ngay_sinh=r.get("birth",""), gioi_tinh=r.get("sex",""),
        quoc_tich=r.get("nationality",""), que_quan=r.get("home",""),
        noi_thuong_tru=r.get("address",""), co_gia_tri_den=r.get("expiry",""),
        elapsed_ms=round(e*1000,1))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok" if _ready else "no_model",
            "model_ready": _ready, "error": _error,
            "corner_model": CORNER_MODEL, "field_model": FIELD_MODEL,
            "ocr": "Tesseract", "detector": "YOLOv8n", "port": PORT}


@app.post("/extract", response_model=CCCDResult,
          summary="[Java] Upload ảnh — 8 field thuần string")
async def extract(file: UploadFile = File(...)):
    _check(); _validate_ext(file.filename)
    try:
        r, _ = _run(await file.read())
    except ValueError as e: raise HTTPException(422, str(e))
    except Exception as e:  raise HTTPException(500, str(e))
    return CCCDResult(**r)


@app.post("/extract/full", response_model=CCCDResultFull,
          summary="[Debug] Upload ảnh — có elapsed_ms")
async def extract_full(file: UploadFile = File(...)):
    _check(); _validate_ext(file.filename)
    try:
        r, e = _run(await file.read())
    except ValueError as e: raise HTTPException(422, str(e))
    except Exception as e:  raise HTTPException(500, str(e))
    return CCCDResultFull(**r, elapsed_ms=round(e*1000,1))


@app.post("/extract/vi", response_model=CCCDResultVi,
          summary="Upload ảnh — label tiếng Việt")
async def extract_vi(file: UploadFile = File(...)):
    _check(); _validate_ext(file.filename)
    try:
        r, e = _run(await file.read())
    except ValueError as e: raise HTTPException(422, str(e))
    except Exception as e:  raise HTTPException(500, str(e))
    return _vi(r, e)


@app.post("/extract/base64", response_model=CCCDResult,
          summary="Gửi ảnh base64")
async def extract_b64(req: Base64Request):
    _check()
    try:
        b64 = req.image_base64
        if "," in b64: b64 = b64.split(",",1)[1]
        r, _ = _run(base64.b64decode(b64))
    except ValueError as e: raise HTTPException(422, str(e))
    except Exception as e:  raise HTTPException(500, str(e))
    return CCCDResult(**r)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=PORT, reload=False)