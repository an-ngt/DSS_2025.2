# Bước 2: Thu thập dữ liệu + Sinh ngày & Tên sản phẩm
# !pip install datasets pandas

from datasets import load_dataset
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("Đang tải dataset...")
ds = load_dataset("vinhplaykennen/FashionReviews")

df = ds['train'].to_pandas()
print(f"✅ Đã load {len(df)} reviews")

# ============================================================
# 2. SINH CỘT NGÀY TỰ ĐỘNG
# ============================================================
# Lý do: Phục vụ phân tích xu hướng cảm xúc theo thời gian
# Ví dụ: Tháng nào khách hay phàn nàn về giao hàng nhất?

def random_date(start="2023-01-01", end="2025-06-01", n=1):
    """Sinh ngày ngẫu nhiên trong khoảng thực tế"""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    delta = (end_dt - start_dt).days
    return [
        (start_dt + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")
        for _ in range(n)
    ]

random.seed(42)  # Đặt seed để kết quả tái lập được
df["Ngày đánh giá"] = random_date(n=len(df))

print("✅ Đã sinh cột [Ngày đánh giá]")
print(df["Ngày đánh giá"].value_counts().head(5))

# ============================================================
# 3. SINH CỘT TÊN SẢN PHẨM TỰ ĐỘNG
# ============================================================
import random

PRODUCT_LIST = [
    "Áo thun basic", "Áo sơ mi nam", "Áo sơ mi nữ", "Quần jean nam", "Quần jean nữ",
    "Đầm maxi", "Đầm công sở", "Áo khoác bomber", "Áo khoác denim", "Quần tây nam",
    "Chân váy midi", "Áo len crop", "Quần short", "Bộ đồ thể thao", "Áo hoodie"
]


def get_product_by_super_keywords(review_text):
    text = str(review_text).lower()

    # 0. ĐẶC BIỆT: Xử lý trường hợp "Hỗn hợp" hoặc "Set/Bộ" (Có cả ÁO và QUẦN/VÁY)
    # Tránh lỗi dòng số 2 (có cả quần lót và áo choàng)
    has_ao = any(k in text for k in ["áo", "t-shirt", "pijama", "top", "shuyt"])
    has_quan_vay = any(k in text for k in ["quần", "váy", "đầm", "skirt", "dress", "sịp", "lót"])
    if has_ao and has_quan_vay:
        return "Bộ đồ thể thao"  # Gom về dạng Set đồ/Bộ đồ

    # 1. Nhóm ĐẦM / VÁY (Dress, Skirt, Đầm body, Đầm xòe, Chân váy...)
    if any(k in text for k in ["váy", "đầm", "skirt", "dress", "búp bê", "yếm", "tutu", "maxi"]):
        if any(k in text for k in ["chân váy", "midi", "xếp ly", "chữ a", "váy ngắn", "váy dài"]):
            return "Chân váy midi"
        if any(k in text for k in ["công sở", "body", "ôm", "sơ mi váy", "liền thân"]):
            return "Đầm công sở"
        return "Đầm maxi"

    # 2. Nhóm ÁO SƠ MI (Sơ mi, Blouse, Cổ đức, Cổ tàu...)
    elif any(k in text for k in ["sơ mi", "sơmi", "blouse", "cổ đức", "cổ tàu", "cổ bẻ"]):
        if any(k in text for k in ["nữ", "gái", "vòng 1", "ngực", "cổ v", "croptop", "voan", "lụa"]):
            return "Áo sơ mi nữ"
        return "Áo sơ mi nam"

    # 3. Nhóm QUẦN JEAN / BÒ / KAKI (Bò, Jean, Baggy, Rách gối...)
    elif any(k in text for k in ["jean", "jeans", "bò", "kaki", "baggy", "rách gối", "ống suông", "ống loe"]):
        if any(k in text for k in ["nữ", "gái", "vòng 3", "mông", "eo", "high waist", "cạp cao"]):
            return "Quần jean nữ"
        return "Quần jean nam"

    # 4. Nhóm ÁO KHOÁC (Jacket, Bomber, Blazer, Gió, Phao, Cardigan...)
    elif any(k in text for k in
             ["khoác", "jacket", "bomber", "denim", "blazer", "phao", "gió", "măng tô", "cardigan", "ấm"]):
        if "denim" in text or "bò" in text:
            return "Áo khoác denim"
        return "Áo khoác bomber"

    # 5. Nhóm ÁO LEN / CROP TOP (Len, Thun gân, Crop, Trễ vai...)
    elif any(k in text for k in ["len", "crop", "lửng", "eo", "ba lỗ", "2 dây", "hai dây", "trễ vai"]):
        return "Áo len crop"

    # 6. Nhóm ÁO HOODIE / NỈ / SWEATER (Hoodie, Sweater, Nỉ, Mũ trùm...)
    elif any(k in text for k in ["hoodie", "sweater", "nỉ", "áo mũ", "mùa đông"]):
        return "Áo hoodie"

    # 7. Nhóm QUẦN TÂY / CÔNG SỞ (Tây, Âu, Vải, Tuyết mưa, Ly quần...)
    elif any(k in text for k in ["tây", "quần vải", "quần âu", "tuyết mưa", "ống đứng", "thanh lịch"]):
        return "Quần tây nam"

    # 8. Nhóm QUẦN SHORT / ĐÙI (Short, Shorts, Đùi, Ngắn, Ngố, Sịp...)
    elif any(k in text for k in ["short", "shorts", "đùi", "quần ngắn", "quần ngố", "lửng"]):
        return "Quần short"

    # 9. Nhóm BỘ ĐỒ / PIJAMA / THỂ THAO (Bộ, Set, Pijama, Mặc nhà...)
    elif any(k in text for k in ["bộ", "set", "pijama", "đồ ngủ", "mặc nhà", "thể thao", "bộ nỉ"]):
        return "Bộ đồ thể thao"

    # 10. Nhóm ÁO THUN / PHÔNG (Thun, Phông, T-shirt, T cổ tròn...)
    elif any(k in text for k in ["thun", "phông", "t-shirt", "cổ tròn", "loang", "oversize"]):
        return "Áo thun basic"

    # 11. HẬU KIỂM 1: Nếu từ khóa quá chung chung chỉ có chữ "ÁO" -> Pick bừa một loại áo cụ thể
    if "áo" in text:
        ao_types = ["Áo thun basic", "Áo sơ mi nam", "Áo sơ mi nữ", "Áo khoác bomber", "Áo khoác denim", "Áo len crop",
                    "Áo hoodie"]
        return random.choice(ao_types)

    # 12. HẬU KIỂM 2: Nếu từ khóa quá chung chung chỉ có chữ "QUẦN" -> Pick bừa một loại quần cụ thể
    if "quần" in text:
        quan_types = ["Quần jean nam", "Quần jean nữ", "Quần tây nam", "Quần short"]
        return random.choice(quan_types)

    # 13. Phương án dự phòng cuối cùng: Nếu review không chứa từ khóa nào ở trên (Ví dụ: "đóng gói đẹp")
    random.seed(42)  # Giữ cố định để kết quả không bị đổi sau mỗi lần chạy
    return random.choice(PRODUCT_LIST)


# Áp dụng hàm siêu từ khóa vào DataFrame
# Lưu ý: Không dùng random.seed(42) ở đây để các câu bốc ngẫu nhiên (mục 11, 12) được phân phối đều
df["Tên sản phẩm"] = df["Nội dung review"].apply(get_product_by_super_keywords)

print("\n✅ Đã sinh cột [Tên sản phẩm] chuẩn hóa bằng bộ siêu từ khóa!")
print(df["Tên sản phẩm"].value_counts())

# ============================================================
# 4. SẮP XẾP LẠI CỘT & XUẤT FILE
# ============================================================
# Đưa 2 cột mới lên đầu cho dễ đọc
cols = ["STT", "Ngày đánh giá", "Tên sản phẩm", "Nội dung review",
        "Chất liệu", "Kiểu dáng", "Kích cỡ", "Giá cả", "Dịch vụ"]
df = df[cols]

print("\n--- XEM THỬ KẾT QUẢ ---")
print(df.head(5).to_string())

df.to_csv("reviews_raw.csv", index=False, encoding="utf-8-sig")
print("\n✅ Đã lưu file reviews_raw.csv")