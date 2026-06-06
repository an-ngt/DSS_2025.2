# create_demo.py
# Bước 5: Tạo file demo và mô tả dữ liệu cho báo cáo

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

LABEL_MAP = {0: "None", 1: "Tiêu cực", 2: "Trung tính", 3: "Tích cực"}
LABEL_COLS = ["Chất liệu", "Kiểu dáng", "Kích cỡ", "Giá cả", "Dịch vụ"]

# ============================================================
# LOAD DỮ LIỆU GỐC (dùng reviews_raw để demo có text gốc đẹp)
# ============================================================
df_raw     = pd.read_csv("reviews_raw.csv",     encoding="utf-8-sig")
df_labeled = pd.read_csv("reviews_labeled.csv", encoding="utf-8-sig")

# Lấy các cột nhãn từ labeled ghép vào raw theo STT
df = df_raw.copy()
label_info = df_labeled[["STT", "Cảm xúc tổng hợp", "split"]].drop_duplicates("STT")
df = df.merge(label_info, on="STT", how="left")

print(f"✅ Đã load {len(df)} reviews")

# ============================================================
# PHẦN A: TẠO FILE DEMO (reviews_demo.csv)
# ============================================================
# Mục tiêu: ~100 dòng, đủ 4 tình huống đặc trưng

print("\n" + "="*50)
print("PHẦN A: TẠO FILE DEMO")
print("="*50)

demo_parts = []

# Tình huống 1: Review tích cực rõ ràng (30 mẫu)
pos = df[df["Cảm xúc tổng hợp"] == 3].sample(30, random_state=42)
demo_parts.append(pos)
print(f"  ✅ Tích cực rõ ràng    : {len(pos)} mẫu")

# Tình huống 2: Review tiêu cực rõ ràng (30 mẫu)
neg = df[df["Cảm xúc tổng hợp"] == 1].sample(30, random_state=42)
demo_parts.append(neg)
print(f"  ✅ Tiêu cực rõ ràng   : {len(neg)} mẫu")

# Tình huống 3: Review trung tính / khen chê lẫn lộn (20 mẫu)
neu = df[df["Cảm xúc tổng hợp"] == 2].sample(20, random_state=42)
demo_parts.append(neu)
print(f"  ✅ Trung tính          : {len(neu)} mẫu")

# Tình huống 4: Review đề cập nhiều khía cạnh cùng lúc (20 mẫu)
# (có ít nhất 3 cột nhãn khác 0)
df["so_khia_canh"] = (df[LABEL_COLS] != 0).sum(axis=1)
multi = df[df["so_khia_canh"] >= 3].sample(20, random_state=42)
demo_parts.append(multi)
print(f"  ✅ Đa khía cạnh (≥3)  : {len(multi)} mẫu")

# Gộp, loại trùng, sắp xếp
df_demo = pd.concat(demo_parts).drop_duplicates(subset=["STT"])
df_demo = df_demo.sort_values("Cảm xúc tổng hợp").reset_index(drop=True)

# Chọn cột cần thiết cho demo
demo_cols = ["STT", "Ngày đánh giá", "Tên sản phẩm",
             "Nội dung review"] + LABEL_COLS + ["Cảm xúc tổng hợp"]
df_demo = df_demo[demo_cols]

# Thêm cột nhãn dạng chữ để dễ đọc khi demo
df_demo["Cảm xúc (chữ)"] = df_demo["Cảm xúc tổng hợp"].map(LABEL_MAP)

df_demo.to_csv("reviews_demo.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ Đã lưu reviews_demo.csv — {len(df_demo)} dòng")

# Xem thử
print("\n--- MẪU FILE DEMO ---")
print(df_demo[["Tên sản phẩm", "Nội dung review",
               "Cảm xúc (chữ)"]].head(5).to_string())

# ============================================================
# PHẦN B: MÔ TẢ DỮ LIỆU CHO BÁO CÁO
# ============================================================
print("\n" + "="*50)
print("PHẦN B: MÔ TẢ DỮ LIỆU CHO BÁO CÁO")
print("="*50)

# B1. Thống kê tổng quan
print("\n📌 THỐNG KÊ TỔNG QUAN:")
total = len(df_labeled)
train = len(df_labeled[df_labeled["split"] == "train"])
test  = len(df_labeled[df_labeled["split"] == "test"])
print(f"  Tổng số review       : {total:,}")
print(f"  Tập Train            : {train:,} ({train/total*100:.0f}%)")
print(f"  Tập Test             : {test:,}  ({test/total*100:.0f}%)")
print(f"  Số khía cạnh phân tích: {len(LABEL_COLS)}")
print(f"  Số nhãn cảm xúc      : 4 (None, Tiêu cực, Trung tính, Tích cực)")

# B2. Độ dài review
df_labeled["do_dai"] = df_labeled["Nội dung review"].astype(str).apply(
    lambda x: len(x.split())
)
print(f"\n📌 ĐỘ DÀI REVIEW (số từ):")
print(f"  Trung bình : {df_labeled['do_dai'].mean():.1f} từ")
print(f"  Ngắn nhất  : {df_labeled['do_dai'].min()} từ")
print(f"  Dài nhất   : {df_labeled['do_dai'].max()} từ")
print(f"  Trung vị   : {df_labeled['do_dai'].median():.0f} từ")

# B3. Phân bố theo sản phẩm
print(f"\n📌 TOP 5 SẢN PHẨM ĐƯỢC ĐÁNH GIÁ NHIỀU NHẤT:")
top_products = df_raw["Tên sản phẩm"].value_counts().head(5)
for name, count in top_products.items():
    print(f"  {name:20s}: {count:,} reviews")

# B4. Phân bố theo thời gian (theo tháng)
df_raw["Ngày đánh giá"] = pd.to_datetime(df_raw["Ngày đánh giá"])
df_raw["Tháng"] = df_raw["Ngày đánh giá"].dt.to_period("M")
monthly = df_raw.groupby("Tháng").size()

# ============================================================
# VẼ BIỂU ĐỒ MÔ TẢ DỮ LIỆU (cho báo cáo)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Biểu đồ 1: Phân bố cảm xúc tổng hợp
ax1 = axes[0, 0]
sentiment_counts = df_labeled[
    df_labeled["Cảm xúc tổng hợp"] != 0
]["Cảm xúc tổng hợp"].value_counts().sort_index()
labels  = [LABEL_MAP[i] for i in sentiment_counts.index]
colors  = ["#e74c3c", "#f39c12", "#2ecc71"]
wedges, texts, autotexts = ax1.pie(
    sentiment_counts.values,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 11}
)
ax1.set_title("Phân bố cảm xúc tổng hợp", fontsize=12, fontweight="bold")

# Biểu đồ 2: Số review theo sản phẩm
ax2 = axes[0, 1]
product_counts = df_raw["Tên sản phẩm"].value_counts()
ax2.barh(product_counts.index[:10][::-1],
         product_counts.values[:10][::-1],
         color="#3498db", edgecolor="white")
ax2.set_title("Top 10 sản phẩm được đánh giá nhiều nhất",
              fontsize=12, fontweight="bold")
ax2.set_xlabel("Số lượng review")

# Biểu đồ 3: Phân bố độ dài review
ax3 = axes[1, 0]
ax3.hist(df_labeled["do_dai"].clip(upper=100),
         bins=40, color="#9b59b6", edgecolor="white", alpha=0.8)
ax3.axvline(df_labeled["do_dai"].mean(), color="red",
            linestyle="--", label=f"Trung bình: {df_labeled['do_dai'].mean():.0f} từ")
ax3.set_title("Phân bố độ dài review (số từ)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Số từ (cắt tại 100)")
ax3.set_ylabel("Số lượng review")
ax3.legend()

# Biểu đồ 4: Xu hướng review theo tháng
ax4 = axes[1, 1]
monthly_vals  = monthly.values
monthly_index = [str(m) for m in monthly.index]
# Chỉ hiện mỗi 3 tháng để không bị chật
step = max(1, len(monthly_index) // 10)
ax4.plot(range(len(monthly_vals)), monthly_vals,
         color="#1abc9c", linewidth=2, marker="o", markersize=3)
ax4.set_xticks(range(0, len(monthly_index), step))
ax4.set_xticklabels(monthly_index[::step], rotation=45, ha="right", fontsize=8)
ax4.set_title("Xu hướng số lượng review theo tháng",
              fontsize=12, fontweight="bold")
ax4.set_ylabel("Số lượng review")
ax4.grid(axis="y", alpha=0.3)

plt.suptitle("Mô tả tổng quan dữ liệu — FashionReviews ABSA",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data_description.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Đã lưu biểu đồ data_description.png")

# ============================================================
# TỔNG KẾT TOÀN BỘ CÁC FILE ĐÃ TẠO
# ============================================================
print("\n" + "="*50)
print("TỔNG KẾT CÁC FILE ĐÃ TẠO")
print("="*50)
print("  📄 reviews_raw.csv         — Dữ liệu thô gốc")
print("  📄 reviews_clean.csv       — Dữ liệu đã làm sạch")
print("  📄 reviews_labeled.csv     — Dữ liệu có nhãn, đã chia Train/Test")
print("  📄 reviews_demo.csv        — File demo ~100 dòng cho Streamlit")
print("  🐍 data_preprocessing.py   — Script làm sạch dữ liệu")
print("  🖼️  label_distribution.png  — Biểu đồ phân bố nhãn gốc")
print("  🖼️  balance_comparison.png  — Biểu đồ trước/sau cân bằng")
print("  🖼️  data_description.png    — Biểu đồ mô tả dữ liệu cho báo cáo")
print("\n🎉 Hoàn tất toàn bộ pipeline dữ liệu!")