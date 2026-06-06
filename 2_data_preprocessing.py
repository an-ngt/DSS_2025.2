# data_preprocessing.py
# Bước 3: Làm sạch và chuẩn hóa tiếng Việt

import pandas as pd
import re
import unicodedata
from underthesea import word_tokenize
from tqdm import tqdm

tqdm.pandas()  # Hiện thanh tiến trình khi xử lý

# ============================================================
# 1. LOAD DỮ LIỆU THÔ
# ============================================================
df = pd.read_csv("reviews_raw.csv", encoding="utf-8-sig")
print(f"✅ Đã load {len(df)} reviews")
print(df.head(3))

# ============================================================
# 2. TỪ ĐIỂN TEENCODE (đặc thù tiếng Việt TMĐT)
# ============================================================
teencode_dict = {
    "sp": "sản phẩm",
    "ung ho": "ủng hộ",
    "ok": "ổn",
    "okie": "ổn",
    "đc": "được",
    "dc": "được",
    "ms": "mới",
    "vs": "với",
    "nx": "nữa",
    "k": "không",
    "ko": "không",
    "kh": "không",
    "hàg": "hàng",
    "hag": "hàng",
    "chx": "chưa",
    "ik": "đi",
    "bt": "bình thường",
    "tl": "trả lời",
    "ship": "giao hàng",
    "r": "rồi",
    "đẹp": "đẹp",
}

def normalize_teencode(text):
    """Thay thế teencode bằng từ chuẩn"""
    words = text.split()
    words = [teencode_dict.get(w, w) for w in words]
    return " ".join(words)

# ============================================================
# 3. CÁC HÀM LÀM SẠCH
# ============================================================

def normalize_unicode(text):
    """Chuẩn hóa Unicode tiếng Việt về NFC (tránh lỗi font)"""
    return unicodedata.normalize("NFC", text)

def to_lowercase(text):
    """Chuyển về chữ thường"""
    return text.lower()

def remove_noise(text):
    """Loại bỏ emoji, ký tự đặc biệt, URL, số điện thoại"""
    # Loại emoji
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    # Loại URL
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # Loại số điện thoại
    text = re.sub(r'\b\d{9,11}\b', ' ', text)
    # Loại ký tự lặp quá nhiều (vd: "đẹpppppp" → "đẹp")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Loại khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_vietnamese(text):
    """Tách từ tiếng Việt bằng underthesea"""
    try:
        tokens = word_tokenize(text, format="text")
        return tokens
    except:
        return text

# Stop words tiếng Việt (tùy chỉnh cho ngành thời trang)
STOP_WORDS = {
    "và", "thì", "là", "của", "có", "trong", "được", "cho",
    "với", "này", "đó", "các", "những", "một", "cũng", "đã",
    "sẽ", "rất", "thế", "vậy", "nên", "khi", "đến", "từ",
    "tôi", "mình", "bạn", "ạ", "ơi", "nhé", "nha", "ý",
    "cái", "cái này", "lắm", "quá", "hơn", "hết", "lại",
}

def remove_stopwords(text):
    """Loại bỏ stop words"""
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)

# ============================================================
# 4. HÀM TỔNG HỢP: ÁP DỤNG TẤT CẢ CÁC BƯỚC
# ============================================================

def preprocess(text):
    """Pipeline làm sạch hoàn chỉnh"""
    if pd.isna(text) or text == "":
        return ""
    text = normalize_unicode(text)      # Bước 1: Chuẩn hóa Unicode
    text = to_lowercase(text)           # Bước 2: Lowercase
    text = remove_noise(text)           # Bước 3: Loại ký tự rác
    text = normalize_teencode(text)     # Bước 4: Chuẩn hóa teencode
    text = tokenize_vietnamese(text)    # Bước 5: Tách từ
    text = remove_stopwords(text)       # Bước 6: Loại stop words
    return text

# ============================================================
# 5. CHẠY PIPELINE & LƯU KẾT QUẢ
# ============================================================
print("\n⏳ Đang xử lý văn bản (có thể mất vài phút)...")
df["review_clean"] = df["Nội dung review"].progress_apply(preprocess)

# Loại bỏ các dòng rỗng sau khi làm sạch
df_clean = df[df["review_clean"].str.strip() != ""].copy()
df_clean = df_clean.dropna(subset=["review_clean"])

print(f"\n✅ Hoàn thành! Còn lại {len(df_clean)} reviews sau làm sạch")

# So sánh trước/sau để kiểm tra
print("\n--- VÍ DỤ TRƯỚC/SAU LÀM SẠCH ---")
for i in range(3):
    print(f"\n[Gốc]  : {df['Nội dung review'].iloc[i]}")
    print(f"[Sạch] : {df_clean['review_clean'].iloc[i]}")

# Lưu file
df_clean.to_csv("reviews_clean.csv", index=False, encoding="utf-8-sig")
print("\n✅ Đã lưu file reviews_clean.csv")