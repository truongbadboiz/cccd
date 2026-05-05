"""
core/utils.py — CCCD Extractor utilities
Model: YOLOv8 nano  |  OCR: Tesseract
"""

import re
import cv2
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CCCD_WIDTH  = 856
CCCD_HEIGHT = 540

CCCD_FIELDS = ["id", "name", "birth", "sex", "nationality", "home", "address", "expiry"]

FIELD_NAMES_VI = {
    "id":          "Số CCCD",
    "name":        "Họ và tên",
    "birth":       "Ngày sinh",
    "sex":         "Giới tính",
    "nationality": "Quốc tịch",
    "home":        "Quê quán",
    "address":     "Nơi thường trú",
    "expiry":      "Có giá trị đến",
}

# 3 class corner — top_right dùng QR
CORNER_LABELS = ["top_left", "bottom_left", "bottom_right"]

# Năm hợp lệ cho ngày sinh / ngày hết hạn
_CURRENT_YEAR = datetime.now().year
_BIRTH_YEAR_MIN  = 1900
_BIRTH_YEAR_MAX  = _CURRENT_YEAR
_EXPIRY_YEAR_MIN = _CURRENT_YEAR - 5   # CCCD tối đa 15 năm
_EXPIRY_YEAR_MAX = _CURRENT_YEAR + 20

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_detection(image: np.ndarray) -> np.ndarray:
    """Resize nếu cạnh dài < 640px (min input YOLOv8)."""
    h, w = image.shape[:2]
    if max(h, w) < 640:
        scale = 640 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LINEAR)
    return image


def sharpen(image: np.ndarray) -> np.ndarray:
    k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, k)


def preprocess_crop_for_ocr(
    crop: np.ndarray,
    field: str = "",
    large_block: bool = False,
) -> np.ndarray:
    """
    Tiền xử lý crop trước khi đưa vào Tesseract.
    Thứ tự: resize → grayscale → denoise → threshold adaptive.

    Args:
        crop:        ảnh BGR crop từ card đã warp.
        field:       tên field (dùng để tuỳ chỉnh adaptive block size).
        large_block: True cho các field nhiều dòng (address, home) →
                     dùng block size lớn hơn để xử lý tốt hơn.
    """
    if crop is None or crop.size == 0:
        return np.zeros((40, 200), dtype=np.uint8)

    # 1. Resize lên nếu crop quá nhỏ
    h, w = crop.shape[:2]
    if h < 40:
        scale = 40 / h
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # 3. Denoise nhẹ
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 4. Threshold adaptive — block size tuỳ field
    # address / home: nền phức tạp hơn → block lớn
    block_size = 31 if (large_block or field in ("address", "home")) else 21
    c_value    = 10 if (large_block or field in ("address", "home")) else 8

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c_value,
    )

    # 5. Dilation nhẹ để nối nét chữ bị đứt
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    return thresh


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def infer_top_right_from_3_corners(corners: dict) -> tuple:
    """Nội suy top_right: TR = TL + BR - BL (hình bình hành)."""
    tl = corners["top_left"]
    bl = corners["bottom_left"]
    br = corners["bottom_right"]
    return (tl[0] + br[0] - bl[0], tl[1] + br[1] - bl[1])


def perspective_transform(image: np.ndarray, source_points: np.ndarray) -> np.ndarray:
    """Warp về kích thước chuẩn CCCD_WIDTH × CCCD_HEIGHT."""
    dst = np.float32([
        [0,          0],
        [CCCD_WIDTH, 0],
        [CCCD_WIDTH, CCCD_HEIGHT],
        [0,          CCCD_HEIGHT],
    ])
    M = cv2.getPerspectiveTransform(source_points, dst)
    return cv2.warpPerspective(image, M, (CCCD_WIDTH, CCCD_HEIGHT))


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def non_max_suppression(boxes: np.ndarray, labels: list,
                         overlap_thresh: float = 0.3):
    if len(boxes) == 0:
        return np.array([]), []

    boxes = boxes.astype("float")
    pick  = []
    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area  = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs  = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i    = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w_i = np.maximum(0, xx2 - xx1 + 1)
        h_i = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w_i * h_i) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int"), [labels[i] for i in pick]


def sort_boxes_top_to_bottom(boxes: np.ndarray, labels: list):
    if len(boxes) == 0:
        return boxes, labels
    order = np.argsort([(boxes[i][0] + boxes[i][2]) / 2 for i in range(len(boxes))])
    return boxes[order], [labels[i] for i in order]


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def validate_cccd_id(s: str) -> str:
    """
    Trả về chuỗi 12 chữ số hoặc giữ nguyên nếu không sửa được.

    Các lỗi OCR phổ biến được xử lý:
      O/Q → 0,  I/L → 1,  S → 5,  B → 8,  Z → 2

    Lỗi 6↔8 trong mã tỉnh được xử lý riêng trong
    cross_validate_id_province() sau khi có thông tin địa chỉ.
    """
    s_strip = s.strip()
    digits = re.sub(r"\D", "", s_strip)
    if len(digits) == 12:
        return digits

    # Bước 1: thay thế ký tự chữ hay bị OCR nhầm thành chữ số
    corrected = (s_strip.upper()
                 .replace("O", "0").replace("Q", "0")
                 .replace("I", "1").replace("L", "1")
                 .replace("S", "5").replace("B", "8").replace("Z", "2"))
    digits = re.sub(r"\D", "", corrected)
    return digits if len(digits) == 12 else s_strip


# Danh sách mã tỉnh CCCD hợp lệ (Thông tư 66/2020/TT-BCA)
_VALID_PROVINCES = {
    "001","002","004","006","008","010","011","012","014","015",
    "017","019","020","022","024","025","026","027","030","031",
    "033","034","035","036","037","038","040","042","044","045",
    "046","048","049","051","052","054","056","058","060","062",
    "064","066","067","068","070","072","074","075","077","079",
    "080","082","083","084","086","087","089","091","092","093",
    "094","095","096",
}


def _is_valid_province(code3: str) -> bool:
    return code3 in _VALID_PROVINCES


def _try_fix_province(digits: str) -> str | None:
    """
    Thử swap 6↔8 ở 3 chữ số đầu (mã tỉnh) để ra mã tỉnh hợp lệ.
    Trả về chuỗi 12 số đã sửa, hoặc None nếu không sửa được.
    """
    prefix = digits[:3]
    suffix = digits[3:]
    for i, ch in enumerate(prefix):
        swap = {"6": "8", "8": "6"}.get(ch)
        if swap:
            candidate = prefix[:i] + swap + prefix[i+1:] + suffix
            if _is_valid_province(candidate[:3]):
                print(f"[INFO] ID province fix: {digits[:3]} → {candidate[:3]}")
                return candidate
    return None


# Bảng map từ chuỗi tên tỉnh trong địa chỉ → mã CCCD (3 số đầu)
# Bao gồm cả các cặp 6↔8 hay bị nhầm
_PROVINCE_NAME_TO_CODE: dict[str, str] = {
    # Tên chuẩn → mã
    "an giang":          "089", "bà rịa":            "077",
    "bắc giang":         "024", "bắc kạn":           "006",
    "bạc liêu":          "095", "bắc ninh":          "027",
    "bến tre":           "083", "bình định":         "052",
    "bình dương":        "074", "bình phước":        "070",
    "bình thuận":        "060", "cà mau":            "096",
    "cần thơ":           "092", "cao bằng":          "004",
    "đà nẵng":           "048", "đắk lắk":           "066",
    "đắk nông":          "067", "điện biên":         "011",
    "đồng nai":          "075", "đồng tháp":         "087",
    "gia lai":           "064", "hà giang":          "002",
    "hà nam":            "035", "hà nội":            "001",
    "hà tĩnh":           "042", "hải dương":         "030",
    "hải phòng":         "031", "hậu giang":         "093",
    "hòa bình":          "017", "hưng yên":          "033",
    "khánh hòa":         "056", "kiên giang":        "091",
    "kon tum":           "062", "lai châu":          "012",
    "lâm đồng":          "068", "lạng sơn":          "020",
    "lào cai":           "010", "long an":           "080",
    "nam định":          "036", "nghệ an":           "037",
    "ninh bình":         "037", "ninh thuận":        "058",  # 037=Nghệ An, xem bên dưới
    "phú thọ":           "025", "phú yên":           "054",
    "quảng bình":        "044", "quảng nam":         "049",
    "quảng ngãi":        "051", "quảng ninh":        "022",
    "quảng trị":         "045", "sóc trăng":         "094",
    "sơn la":            "014", "tây ninh":          "072",
    "thái bình":         "038", "thái nguyên":       "019",
    "thanh hóa":         "040", "thừa thiên huế":    "046",
    "tiền giang":        "082", "tp hồ chí minh":    "079",
    "hồ chí minh":       "079", "trà vinh":          "084",
    "tuyên quang":       "008", "vĩnh long":         "086",
    "vĩnh phúc":         "026", "yên bái":           "015",
    # Alias
    "nam dinh":          "036", "ninh binh":         "037",
    "thai binh":         "038", "ha noi":            "001",
}
# Sửa: ninh bình thật ra là 034, ko phải 037 (037=Nghệ An)
_PROVINCE_NAME_TO_CODE["ninh bình"]  = "034"
_PROVINCE_NAME_TO_CODE["ninh binh"]  = "034"
# Thêm
_PROVINCE_NAME_TO_CODE["bắc giang"]  = "024"
_PROVINCE_NAME_TO_CODE["ha tinh"]    = "042"


def cross_validate_id_province(cccd_id: str, home: str, address: str) -> str:
    """
    Cross-validate 3 số đầu của CCCD ID với tỉnh trong địa chỉ.
    Nếu phát hiện lỗi 6↔8 trong mã tỉnh, tự động sửa.

    Chỉ sửa khi: tỉnh detect được, và swap 6↔8 cho ra mã tỉnh khớp.
    """
    if not cccd_id or len(cccd_id) != 12:
        return cccd_id

    # Tìm tên tỉnh trong home/address
    combined = (home + " " + address).lower()
    detected_code = None
    for province_name, code in _PROVINCE_NAME_TO_CODE.items():
        if province_name in combined:
            detected_code = code
            break

    if not detected_code:
        return cccd_id

    current_prefix = cccd_id[:3]
    if current_prefix == detected_code:
        return cccd_id  # đã đúng

    # Kiểm tra swap 6↔8 ở prefix
    for i, ch in enumerate(current_prefix):
        swap = {"6": "8", "8": "6"}.get(ch)
        if swap:
            candidate_prefix = current_prefix[:i] + swap + current_prefix[i+1:]
            if candidate_prefix == detected_code:
                fixed = candidate_prefix + cccd_id[3:]
                print(f"[INFO] ID cross-validated: {cccd_id} → {fixed} "
                      f"(province '{detected_code}' from address)")
                return fixed

    return cccd_id


def _correct_year(yyyy: str, year_min: int, year_max: int) -> str:
    """
    Sửa năm 4 chữ số bị OCR nhầm khi nằm ngoài khoảng hợp lệ.

    Chiến lược (theo thứ tự):
      1. Nếu ký tự đầu bị nhầm thành '2' thay vì '1'  → thử đổi lại
      2. Nếu ký tự thứ 3 bị nhầm (vd 2094 → 2004, 1094 → 1994) → thử sửa
      3. Fallback: giữ nguyên
    """
    try:
        year_int = int(yyyy)
    except ValueError:
        return yyyy

    if year_min <= year_int <= year_max:
        return yyyy  # hợp lệ, giữ nguyên

    # ----------------------------------------------------------------
    # Chiến lược sửa: ưu tiên theo độ tin cậy giảm dần
    #
    # Tier 0 (CCCD-specific, ưu tiên cao nhất):
    #   OCR hay nhầm '1' thành '2' ở chữ số đầu (nét thẳng vs cong).
    #   Nếu yyyy bắt đầu bằng '2' và '1'+yyyy[1:] hợp lệ → chọn ngay.
    #
    # Tier 1 — sửa đúng 1 chữ số, kết quả trong khoảng hợp lệ
    # Tier 2 — fallback gần midpoint thực tế nhất
    # ----------------------------------------------------------------

    # Tier 0: '2xxx' → '1xxx' (lỗi OCR phổ biến nhất cho năm sinh)
    if yyyy.startswith("2"):
        candidate_1x = "1" + yyyy[1:]
        try:
            y = int(candidate_1x)
            if year_min <= y <= year_max:
                print(f"[WARN] Năm '{yyyy}' ngoài khoảng [{year_min},{year_max}] "
                      f"→ sửa thành '{candidate_1x}' (OCR 1→2 correction)")
                return candidate_1x
        except ValueError:
            pass

    # midpoint thực tế: birth → 1985, expiry → năm hiện tại + 7
    mid = 1985 if year_max <= _CURRENT_YEAR else _CURRENT_YEAR + 7

    tier1: list[tuple[int, str]] = []   # (dist_from_mid, candidate)

    def _try(candidate: str) -> None:
        try:
            y = int(candidate)
            if year_min <= y <= year_max:
                tier1.append((abs(y - mid), candidate))
        except ValueError:
            pass

    for d in "0123456789":
        _try(d         + yyyy[1:])       # sửa vị trí 0: thay yyyy[0]
        _try(yyyy[0]   + d + yyyy[2:])   # sửa vị trí 1: thay yyyy[1]
        _try(yyyy[:2]  + d + yyyy[3:])   # sửa vị trí 2: thay yyyy[2]
        _try(yyyy[:3]  + d)              # sửa vị trí 3: thay yyyy[3]

    if tier1:
        tier1.sort()
        corrected = tier1[0][1]
        print(f"[WARN] Năm '{yyyy}' ngoài khoảng [{year_min},{year_max}] "
              f"→ sửa thành '{corrected}'")
        return corrected

    print(f"[WARN] Không thể sửa năm '{yyyy}' — giữ nguyên")
    return yyyy


def normalize_date(s: str, field: str = "birth") -> str:
    """
    Chuẩn hoá ngày tháng về dạng DD/MM/YYYY.
    Tự động sửa năm bị OCR sai nằm ngoài khoảng hợp lệ.
    """
    s = s.strip().replace("-", "/").replace(".", "/")
    parts = s.split("/")

    year_min = _BIRTH_YEAR_MIN  if field == "birth" else _EXPIRY_YEAR_MIN
    year_max = _BIRTH_YEAR_MAX  if field == "birth" else _EXPIRY_YEAR_MAX

    if len(parts) == 3:
        dd, mm, yyyy = parts[0].strip(), parts[1].strip(), parts[2].strip()

        # Năm 2 chữ số → mở rộng
        # Ngưỡng: năm 2 chữ số <= (năm hiện tại % 100) + 5 → thế kỷ 21
        # Ví dụ: năm hiện tại 2026 → ngưỡng 31; "04" ≤ 31 → 2004; "94" > 31 → 1994
        # Phù hợp CCCD VN: sinh từ 1924–2010, hết hạn 2020–2040
        if len(yyyy) == 2:
            try:
                yy = int(yyyy)
                threshold = (_CURRENT_YEAR % 100) + 5   # vd 2026 → threshold=31
                yyyy = ("20" if yy <= threshold else "19") + yyyy
            except ValueError:
                pass

        # Năm 4 chữ số → sanity check & tự sửa
        if len(yyyy) == 4:
            yyyy = _correct_year(yyyy, year_min, year_max)

        return f"{dd.zfill(2)}/{mm.zfill(2)}/{yyyy}"

    # Thử parse từ chuỗi thuần chữ số
    digits = re.sub(r"\D", "", s)
    if len(digits) == 8:
        yyyy = _correct_year(digits[4:], year_min, year_max)
        return f"{digits[:2]}/{digits[2:4]}/{yyyy}"

    return s


# ---------------------------------------------------------------------------
# FIX: normalize_sex — loại bỏ mọi ký tự không phải chữ cái (kể cả _)
# ---------------------------------------------------------------------------
# Bộ ký tự tiếng Việt đầy đủ để whitelist
_VI_CHARS = (
    "a-zA-Z"
    "àáâãèéêìíòóôõùúýăđơư"
    "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ"
    "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ"
)
_RE_NOT_VI_ALPHA = re.compile(rf"[^{_VI_CHARS}\s]")


def normalize_sex(s: str) -> str:
    """
    Chuẩn hoá giới tính.

    Bug cũ: regex [^a-zA-ZÀ-ỹ\\s] không loại được '_' vì '_' (U+005F)
    nằm trước 'À' (U+00C0) trong bảng Unicode nhưng lại nằm giữa khoảng
    [...] nên không khớp → dùng whitelist ký tự tiếng Việt tường minh.
    """
    # Loại toàn bộ ký tự không phải chữ cái Việt / khoảng trắng
    s = _RE_NOT_VI_ALPHA.sub("", s).strip()
    lo = s.lower()

    if re.search(r"\bnam\b",    lo): return "Nam"
    if re.search(r"\bmale\b",   lo): return "Nam"
    if re.search(r"\bn[uữ]\b",  lo): return "Nữ"
    if re.search(r"\bfemale\b", lo): return "Nữ"
    if lo.strip() == "m":            return "Nam"
    if lo.strip() == "f":            return "Nữ"
    return s.strip()


# ---------------------------------------------------------------------------
# Clean text helpers
# ---------------------------------------------------------------------------

# Các cụm label tiếng Việt / tiếng Anh thường bị OCR lẫn vào đầu field
# Sắp xếp từ dài → ngắn để tránh partial match
_LABEL_PREFIXES = [
    # address — "Nơi thường trú / Place of residence"
    # Dùng \S* để khoan dung với dấu tiếng Việt bị OCR sai (trú→trủ, ơ→o...)
    r"\S*[Nn]\S*\s+\S*th\S*\s+\S*tr\S*\s*/?\s*\S*[Pp]\S*\s+of\s+\S*res\S*\s*[:\-]?",
    r"\S*[Nn]\S*\s+\S*th\S*\s+\S*tr\S*\s*[:\-]?",
    r"\S*[Pp]\S*\s+of\s+\S*res\S*\s*[:\-]?",
    # home — "Quê quán / Place of origin"
    r"\S*[Qq]u\S*\s+\S*qu\S*\s*/?\s*\S*[Pp]\S*\s+of\s+\S*ori\S*\s*[:\-]?",
    r"\S*[Qq]u\S*\s+\S*qu\S*\s*[:\-]?",
    r"\S*[Pp]\S*\s+of\s+\S*ori\S*\s*[:\-]?",
]
_RE_LABEL_PREFIX = re.compile(
    r"^\s*(?:" + "|".join(_LABEL_PREFIXES) + r")\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Label prefix stripping
# ---------------------------------------------------------------------------
# Các từ khoá của label, dùng \w+ để match cả ký tự Unicode tiếng Việt
# với flag re.UNICODE. Mỗi "token" trong label được match bằng \S+
# (bất kỳ non-whitespace) để chịu được OCR sai dấu.
#
# Pattern: khớp phần đầu chuỗi gồm N từ mà trong đó có các từ khoá
# đặc trưng của label, tiếp theo là dấu / hay : tuỳ chọn.
# Dùng look-ahead để không nuốt phần địa chỉ thật.
#
# Chiến lược an toàn: chỉ strip nếu 3 token đầu của chuỗi chứa
# các từ khoá nhận dạng được (không strip nếu nghi ngờ là nội dung thật).

_ADDR_KW   = re.compile(r"th[uư\w]*ng|tr[uúủụ\w]*", re.IGNORECASE | re.UNICODE)
_HOME_KW   = re.compile(r"qu[aáạ\w]*n|origin",       re.IGNORECASE | re.UNICODE)
_PLACE_KW  = re.compile(r"place|piace|pIace",         re.IGNORECASE | re.UNICODE)
_RESID_KW  = re.compile(r"res[il]?d?\w*",             re.IGNORECASE | re.UNICODE)
_NOI_KW    = re.compile(r"n[o\w][i\w]",               re.IGNORECASE | re.UNICODE)
_QUE_KW    = re.compile(r"qu[eê\w]+",                 re.IGNORECASE | re.UNICODE)

# Regex strip label theo cách dễ duy trì:
# - Một "cụm label" gồm các từ liên quan đến nhau, cách nhau bằng space
# - Hỗ trợ prefix số rác (ví dụ "129 Nơi thường...") bằng ^\d+\s*
# - Dùng re.sub để xoá phần khớp, giữ phần sau
#
# Pattern: bắt đầu chuỗi (tuỳ chọn số + khoảng trắng rác),
#   rồi đến các token label, kết thúc bằng ký tự phân cách tuỳ chọn.
#
# Thay vì hardcode từng biến thể dấu, dùng [\S]+ để khớp bất kỳ
# non-whitespace token, nhưng chỉ khi token đó khớp với từ khoá kỳ vọng
# theo ngữ nghĩa (được kiểm tra bằng look-ahead/look-behind riêng).
#
# Cách đơn giản nhất: dùng re với named group, match pattern lỏng
# cho từng từ khoá, ghép lại.

def _build_label_re() -> re.Pattern:
    """
    Xây pattern match cụm label "Nơi thường trú / Place of residence"
    và "Quê quán / Place of origin" ở đầu chuỗi, khoan dung với OCR.

    Mỗi từ khoá được match bằng: ký tự đầu (cố định) + S* (đuôi linh hoạt).
    """
    # Token patterns (bắt đầu bằng chữ cái đặc trưng + bất kỳ non-space)
    noi   = r"[Nn]\S*"          # Nơi / noi
    thuong= r"[Tt]h\S*"        # thường / thuong / thưòng
    tru   = r"[Tt]r\S*"        # trú / tru / trủ
    place = r"[Pp][il]\S*"     # Place / Piace / pIace
    of    = r"[Oo][Ff]"        # of / OF
    res   = r"[Rr][Ee][Ss]\S*" # residence / resdence / residencs
    que   = r"[Qq]u[eêếềệểễ]\S*"  # Quê / Que
    quan  = r"[Qq]u[aáạ]\S*"  # quán / quan / quản
    ori   = r"[Oo][Rr][Ii]\S*" # origin / Origin

    # Phân cách giữa phần VI và phần EN: khoảng trắng, /, :, -
    # và tối đa 1 token rác đơn lẻ (1 ký tự, thường là 'í','Í','I','l',...)
    sep   = r"[\s/:\-]*(?:[^\w\s]{0,3}\s*)?"

    # Khoảng trắng 1+
    sp    = r"\s+"

    # Prefix số rác tuỳ chọn (vd "129 ")
    num_prefix = r"(?:\d[\d\s]*)?"

    # Pattern địa chỉ: "Nơi thường trú [/ Place of residence]"
    addr_vi  = rf"{noi}{sp}{thuong}{sp}{tru}"
    addr_en  = rf"{place}{sp}{of}{sp}{res}"
    addr_pat = rf"(?:{addr_vi}{sep}(?:{addr_en}{sep})?|{addr_en}{sep})"

    # Pattern quê quán: "Quê quán [/ Place of origin]"
    home_vi  = rf"{que}{sp}{quan}"
    home_en  = rf"{place}{sp}{of}{sp}{ori}"
    home_pat = rf"(?:{home_vi}{sep}(?:{home_en}{sep})?|{home_en}{sep})"

    full = rf"^\s*{num_prefix}\s*(?:{addr_pat}|{home_pat})"
    return re.compile(full, re.IGNORECASE | re.UNICODE)


_RE_LABEL_STRIP = _build_label_re()


def strip_field_label(s: str) -> str:
    """
    Xoá phần label bị OCR lẫn vào đầu giá trị field.

    Dùng vòng lặp:
      1. Strip cụm label (VI + EN) bằng regex
      2. Strip token rác đơn lẻ 1-2 ký tự ở đầu (vd 'í', 'Í', 'I', 'l')
      Lặp lại đến khi chuỗi ổn định (thường 1-3 vòng).
    """
    if not s:
        return s

    result = s
    prev   = None
    for _ in range(5):
        # Strip label pattern
        r2 = _RE_LABEL_STRIP.sub("", result).strip().lstrip(":/.-,")
        # Strip token rác ngắn ở đầu:
        # Token bị coi là rác nếu: ≤ 2 ký tự VÀ không phải chữ cái thuần
        # (tức là chứa ký tự không phải alpha, như số, ký tự đặc biệt, hoặc
        #  là ký tự đơn lẻ không phải âm tiết tiếng Việt có nghĩa).
        # Ví dụ rác: 'í' (1 char), 'Í' (1 char), 'I' (1 char), '129' (số)
        # Không strip: 'Tổ' (từ tiếng Việt 2 char có nghĩa), 'Xã'
        def _is_noise_token(tok: str) -> bool:
            if len(tok) <= 1:
                return True   # token 1 ký tự luôn là rác trong context này
            if len(tok) <= 2 and not tok[0].isupper():
                return True   # token 2 char không viết hoa → rác
            return False

        first_tok_m = re.match(r"^(\S+)\s+", r2)
        if first_tok_m and _is_noise_token(first_tok_m.group(1)):
            r2 = r2[first_tok_m.end():].strip()
        if r2 == prev or r2 == result:
            result = r2
            break
        prev   = result
        result = r2

    return result.strip() if result.strip() else s.strip()


# Ký tự hợp lệ trong địa chỉ / quê quán: chữ Việt, số, khoảng trắng,
# dấu phẩy, dấu chấm, gạch ngang, dấu gạch chéo
_RE_NOISE_ADDR = re.compile(
    rf"[^{_VI_CHARS}0-9\s,.\-/]"
)


def clean_address_text(s: str) -> str:
    """
    Loại ký tự rác khỏi địa chỉ / quê quán sau OCR.

    Thứ tự:
      1. Strip label prefix (Nơi thường trú / Quê quán...)
      2. Loại ký tự đặc biệt không hợp lệ
      3. Strip cụm noise đầu chuỗi trước tên địa danh thật
      4. Chuẩn hoá whitespace
    """
    s = strip_field_label(s)
    s = _RE_NOISE_ADDR.sub(" ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()

    # Bước 3: strip cụm noise trước địa danh thật.
    # Địa danh Việt Nam thường: "Từ, Từ, Từ" — mỗi thành phần ≥ 2 từ,
    # cách nhau bằng dấu phẩy.
    # Tìm vị trí đầu tiên có pattern "Chữhoa+\S+, Chữhoa+":
    #   → đây là nơi địa danh thật bắt đầu
    # Chỉ strip nếu phần bị bỏ phía trước ngắn hơn phần giữ lại
    # (tránh strip sai như "Tổ 4 Tân Bình, ...")
    m = re.search(
        r"(?<!\S)"                  # bắt đầu sau khoảng trắng hoặc đầu chuỗi
        r"([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ"
        r"ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]"
        r"[a-zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]"  # chữ thường sau chữ hoa
        r"\S*"                      # phần còn lại của từ đầu
        r"(?:\s+\S+)*"             # thêm từ
        r",)",                      # dấu phẩy xác nhận đây là địa danh
        s,
    )
    if m and m.start() > 0:
        prefix_len  = m.start()
        content_len = len(s) - prefix_len
        # Chỉ strip nếu noise prefix ngắn hơn nội dung thật
        # và prefix không phải là nội dung có nghĩa dài
        if prefix_len < content_len and prefix_len <= 20:
            s = s[m.start():].strip()

    return s