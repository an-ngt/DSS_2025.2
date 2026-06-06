# labeling.py
# Bước 4: Gán nhãn, kiểm tra, cân bằng và chia dữ liệu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from collections import Counter

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# 1. LOAD DỮ LIỆU SẠCH
# ============================================================
df = pd.read_csv("reviews_clean.csv", encoding="utf-8-sig")
print(f"✅ Đã load {len(df)} reviews sạch")

# Các cột nhãn cần xử lý
LABEL_COLS = ["Chất liệu", "Kiểu dáng", "Kích cỡ", "Giá cả", "Dịch vụ"]

# Mapping nhãn để dễ đọc
LABEL_MAP = {0: "None", 1: "Tiêu cực", 2: "Trung tính", 3: "Tích cực"}

# ============================================================
# 2. KIỂM TRA CHẤT LƯỢNG NHÃN
# ============================================================
print("\n" + "="*50)
print("KIỂM TRA CHẤT LƯỢNG NHÃN")
print("="*50)

# 2a. Kiểm tra giá trị hợp lệ (chỉ được là 0, 1, 2, 3)
print("\n📌 Kiểm tra giá trị hợp lệ:")
for col in LABEL_COLS:
    unique_vals = df[col].unique()
    invalid = [v for v in unique_vals if v not in [0, 1, 2, 3]]
    if invalid:
        print(f"  ⚠️  [{col}] có giá trị lạ: {invalid}")
    else:
        print(f"  ✅ [{col}] hợp lệ — các giá trị: {sorted(unique_vals)}")

# 2b. Kiểm tra giá trị thiếu (NaN)
print("\n📌 Kiểm tra giá trị thiếu (NaN):")
missing = df[LABEL_COLS].isnull().sum()
if missing.sum() == 0:
    print("  ✅ Không có giá trị thiếu")
else:
    print(missing[missing > 0])
    # Điền NaN bằng 0 (None - không đề cập)
    df[LABEL_COLS] = df[LABEL_COLS].fillna(0).astype(int)
    print("  → Đã điền NaN = 0 (None)")

# 2c. Kiểm tra review rỗng sau làm sạch
print("\n📌 Kiểm tra review rỗng:")
empty_reviews = df["review_clean"].isna() | (df["review_clean"].str.strip() == "")
print(f"  Số review rỗng: {empty_reviews.sum()}")
df = df[~empty_reviews].copy()
print(f"  → Còn lại: {len(df)} reviews")

# ============================================================
# 3. PHÂN TÍCH PHÂN BỐ NHÃN
# ============================================================
print("\n" + "="*50)
print("PHÂN BỐ NHÃN CẢM XÚC THEO TỪNG KHÍA CẠNH")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, col in enumerate(LABEL_COLS):
    counts = df[col].value_counts().sort_index()
    labels = [LABEL_MAP.get(i, str(i)) for i in counts.index]
    colors = ["#95a5a6", "#e74c3c", "#f39c12", "#2ecc71"]
    color_list = [colors[i] for i in counts.index]

    axes[idx].bar(labels, counts.values, color=color_list, edgecolor="white")
    axes[idx].set_title(f"Phân bố nhãn: {col}", fontsize=12, fontweight="bold")
    axes[idx].set_ylabel("Số lượng")

    # Hiện % trên mỗi cột
    total = counts.sum()
    for i, (label, val) in enumerate(zip(labels, counts.values)):
        axes[idx].text(i, val + total*0.01, f"{val/total*100:.1f}%",
                      ha="center", fontsize=9)

    print(f"\n[{col}]")
    for label_id, count in zip(counts.index, counts.values):
        print(f"  {LABEL_MAP[label_id]:10s}: {count:5d} ({count/total*100:.1f}%)")

# Ẩn subplot thừa
axes[-1].set_visible(False)
plt.suptitle("Phân bố nhãn cảm xúc theo từng khía cạnh", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("label_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Đã lưu biểu đồ label_distribution.png")

# ============================================================
# 4. TẠO NHÃN TỔNG HỢP (Overall Sentiment)
# ============================================================
# Mục đích: Tóm tắt cảm xúc chung của toàn bộ review
# Quy tắc: Lấy nhãn xuất hiện nhiều nhất trong 5 khía cạnh
#          (bỏ qua các khía cạnh = 0 vì đó là "không đề cập")

def get_overall_sentiment(row):
    """Tính cảm xúc tổng hợp từ 5 khía cạnh bằng cơ chế logic chặt chẽ"""
    # Chỉ lấy các nhãn có cảm xúc thực sự (1, 2, 3), bỏ qua 0
    sentiments = [row[col] for col in LABEL_COLS if row[col] != 0]

    if not sentiments:
        return 0  # Không khía cạnh nào được nhắc đến -> Tổng hợp là None

    # Đếm số lần xuất hiện của từng nhãn
    counts = Counter(sentiments)

    # Tìm số lần xuất hiện lớn nhất (ví dụ: nhãn xuất hiện nhiều nhất là 2 lần)
    max_freq = max(counts.values())

    # Lấy ra danh sách các nhãn đạt số lần xuất hiện lớn nhất đó
    most_common_labels = [label for label, freq in counts.items() if freq == max_freq]

    # XỬ LÝ TRƯỜNG HỢP ĐỒNG HẠNG (Xung đột cảm xúc)
    if len(most_common_labels) > 1:
        # Nếu có cả 1 (Tiêu cực) và 3 (Tích cực) cạnh tranh nhau
        if 1 in most_common_labels and 3 in most_common_labels:
            return 2  # Coi như cảm xúc Hỗn hợp/Trung tính
        # Nếu đồng hạng giữa (1 và 2) hoặc (2 và 3), ưu tiên nhãn cảm xúc mạnh hơn (1 hoặc 3)
        elif 1 in most_common_labels:
            return 1
        elif 3 in most_common_labels:
            return 3

    # Trường hợp bình thường: Có 1 nhãn chiến thắng tuyệt đối
    return most_common_labels[0]

df["Cảm xúc tổng hợp"] = df.apply(get_overall_sentiment, axis=1)

print("\n📌 Phân bố Cảm xúc tổng hợp:")
overall_counts = df["Cảm xúc tổng hợp"].value_counts().sort_index()
for label_id, count in overall_counts.items():
    print(f"  {LABEL_MAP[label_id]:10s}: {count:5d} ({count/len(df)*100:.1f}%)")

# ============================================================
# 5. CHIA TRAIN / TEST (80/20)
# ============================================================
print("\n" + "="*50)
print("CHIA DỮ LIỆU TRAIN / TEST")
print("="*50)

# stratify: đảm bảo tỷ lệ nhãn tổng hợp giống nhau ở cả train và test
df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Cảm xúc tổng hợp"]  # Quan trọng! Giữ tỷ lệ nhãn cân bằng
)

print(f"  Train: {len(df_train)} reviews ({len(df_train)/len(df)*100:.0f}%)")
print(f"  Test : {len(df_test)} reviews  ({len(df_test)/len(df)*100:.0f}%)")

# Đánh dấu split vào dataframe
df_train = df_train.copy()
df_test  = df_test.copy()
df_train["split"] = "train"
df_test["split"]  = "test"

# Gộp lại thành 1 file duy nhất
df_labeled = pd.concat([df_train, df_test], ignore_index=True)

# ============================================================
# 6. XUẤT FILE reviews_labeled.csv
# ============================================================
df_labeled.to_csv("reviews_labeled.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ Đã lưu reviews_labeled.csv ({len(df_labeled)} dòng)")

# Xem thử kết quả cuối
print("\n--- MẪU KẾT QUẢ ---")
sample_cols = ["Nội dung review", "review_clean",
               "Chất liệu", "Kiểu dáng", "Cảm xúc tổng hợp", "split"]
print(df_labeled[sample_cols].head(5).to_string())



# ============================================================
# PHẦN BỔ SUNG: XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU
# Thêm vào cuối file labeling.py
# ============================================================

# Chỉ làm việc trên tập TRAIN, không đụng Test
df_train_only = df_labeled[df_labeled["split"] == "train"].copy()
df_test_only  = df_labeled[df_labeled["split"] == "test"].copy()

# Bỏ nhãn None (0) ra khỏi quá trình cân bằng
# vì chỉ có 112 dòng, không đủ đại diện
df_train_valid = df_train_only[df_train_only["Cảm xúc tổng hợp"] != 0].copy()
df_train_none  = df_train_only[df_train_only["Cảm xúc tổng hợp"] == 0].copy()

print("="*50)
print("TRƯỚC KHI CÂN BẰNG (tập Train)")
print("="*50)
counts_before = df_train_valid["Cảm xúc tổng hợp"].value_counts().sort_index()
for label_id, count in counts_before.items():
    print(f"  {LABEL_MAP[label_id]:10s}: {count:6d} ({count/len(df_train_valid)*100:.1f}%)")

# ============================================================
# XÁC ĐỊNH MỤC TIÊU CÂN BẰNG
# ============================================================
# Lấy mức trung bình của 3 nhóm làm mục tiêu
target_count = int(df_train_valid["Cảm xúc tổng hợp"].value_counts().mean())
print(f"\n🎯 Mục tiêu mỗi nhóm: ~{target_count} mẫu")

# ============================================================
# THỰC HIỆN CÂN BẰNG
# ============================================================
balanced_parts = []

for label_id in [1, 2, 3]:  # Tiêu cực, Trung tính, Tích cực
    group = df_train_valid[df_train_valid["Cảm xúc tổng hợp"] == label_id]
    current_count = len(group)

    if current_count < target_count:
        # OVERSAMPLING: nhóm thiểu số → lấy thêm mẫu (có lặp lại)
        extra = group.sample(
            n=target_count - current_count,
            replace=True,        # Cho phép lấy lặp lại
            random_state=42
        )
        group_balanced = pd.concat([group, extra], ignore_index=True)
        action = f"Oversampling ({current_count} → {len(group_balanced)})"

    elif current_count > target_count:
        # UNDERSAMPLING: nhóm đa số → bỏ bớt mẫu
        group_balanced = group.sample(
            n=target_count,
            replace=False,       # Không lấy lặp lại
            random_state=42
        )
        action = f"Undersampling ({current_count} → {len(group_balanced)})"

    else:
        group_balanced = group
        action = f"Giữ nguyên ({current_count})"

    print(f"  [{LABEL_MAP[label_id]:10s}]: {action}")
    balanced_parts.append(group_balanced)

# Gộp lại + thêm lại nhóm None
df_train_balanced = pd.concat(balanced_parts + [df_train_none], ignore_index=True)

# Xáo trộn thứ tự để tránh mô hình học theo thứ tự
df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================
# KIỂM TRA SAU CÂN BẰNG
# ============================================================
print("\n" + "="*50)
print("SAU KHI CÂN BẰNG (tập Train)")
print("="*50)
counts_after = df_train_balanced[
    df_train_balanced["Cảm xúc tổng hợp"] != 0
]["Cảm xúc tổng hợp"].value_counts().sort_index()

for label_id, count in counts_after.items():
    print(f"  {LABEL_MAP[label_id]:10s}: {count:6d} ({count/len(df_train_balanced)*100:.1f}%)")

# ============================================================
# VẼ BIỂU ĐỒ SO SÁNH TRƯỚC/SAU
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels_plot  = ["Tiêu cực", "Trung tính", "Tích cực"]
colors_plot  = ["#e74c3c", "#f39c12", "#2ecc71"]

# Trước
vals_before = [counts_before.get(i, 0) for i in [1, 2, 3]]
ax1.bar(labels_plot, vals_before, color=colors_plot, edgecolor="white")
ax1.set_title("Trước khi cân bằng", fontsize=13, fontweight="bold")
ax1.set_ylabel("Số lượng (tập Train)")
total_before = sum(vals_before)
for i, v in enumerate(vals_before):
    ax1.text(i, v + total_before*0.01, f"{v:,}\n({v/total_before*100:.1f}%)",
             ha="center", fontsize=10)

# Sau
vals_after = [counts_after.get(i, 0) for i in [1, 2, 3]]
ax2.bar(labels_plot, vals_after, color=colors_plot, edgecolor="white")
ax2.set_title("Sau khi cân bằng", fontsize=13, fontweight="bold")
ax2.set_ylabel("Số lượng (tập Train)")
total_after = sum(vals_after)
for i, v in enumerate(vals_after):
    ax2.text(i, v + total_after*0.01, f"{v:,}\n({v/total_after*100:.1f}%)",
             ha="center", fontsize=10)

plt.suptitle("So sánh phân bố nhãn Trước/Sau khi cân bằng (tập Train)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("balance_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Đã lưu biểu đồ balance_comparison.png")

# ============================================================
# GỘP LẠI VÀ LƯU FILE CUỐI
# ============================================================
# Gộp train đã cân bằng + test giữ nguyên
df_final = pd.concat([df_train_balanced, df_test_only], ignore_index=True)

df_final.to_csv("reviews_labeled.csv", index=False, encoding="utf-8-sig")

print("\n" + "="*50)
print("TỔNG KẾT FILE reviews_labeled.csv")
print("="*50)
print(f"  Tổng cộng : {len(df_final):,} dòng")
print(f"  Tập Train : {len(df_train_balanced):,} dòng (đã cân bằng)")
print(f"  Tập Test  : {len(df_test_only):,} dòng (giữ nguyên thực tế)")
print("\n✅ Hoàn tất! File reviews_labeled.csv đã sẵn sàng cho bước tiếp theo.")