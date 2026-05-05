# CCCD Information Extractor

## Cấu trúc thư mục

```
cccd_python/
│
├── main.py                         ← Entry point (chạy từ đây)
├── requirements.txt
│
├── core/                           ← Logic xử lý ảnh & model
│   ├── __init__.py
│   ├── utils.py                    ← Constants, NMS, warp, post-process
│   └── extractor.py                ← CCCDExtractor class (pipeline chính)
│
├── api/                            ← FastAPI server
│   ├── __init__.py
│   └── server.py                   ← Endpoints: /extract, /extract/vi, /extract/base64
│
├── training/                       ← Chuẩn bị dữ liệu & train model
│   └── prepare_dataset.py          ← VOC XML → TFRecord
│
├── models/                         ← Model weights (không commit git)
│   ├── corner_detector/
│   │   └── saved_model/            ← TF SavedModel 3 class góc
│   └── field_detector/
│       └── saved_model/            ← TF SavedModel 8 class field
│
└── data/                           ← Dữ liệu training (không commit git)
    ├── annotations/
    │   ├── corner/                 ← VOC XML label 3 góc
    │   └── field/                  ← VOC XML label 8 field
    └── tfrecords/                  ← TFRecord output sau prepare_dataset.py
```

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy server

```bash
# Development
python main.py

# Production
uvicorn api.server:app --host 0.0.0.0 --port 8002 --workers 2
```

## Chuẩn bị dữ liệu training

```bash
# Corner model (3 class: top_left, bottom_left, bottom_right)
python -m training.prepare_dataset \
  --detector corner \
  --input_dir  data/annotations/corner \
  --output_dir data/tfrecords/corner

# Field model (8 class: id, name, birth, sex, nationality, home, address, expiry)
python -m training.prepare_dataset \
  --detector field \
  --input_dir  data/annotations/field \
  --output_dir data/tfrecords/field
```

## Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `CORNER_MODEL_PATH` | `models/corner_detector/saved_model` | Đường dẫn TF SavedModel góc |
| `FIELD_MODEL_PATH` | `models/field_detector/saved_model` | Đường dẫn TF SavedModel field |
| `PORT` | `8002` | Port server |
