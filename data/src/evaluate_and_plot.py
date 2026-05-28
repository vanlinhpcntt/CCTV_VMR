import os
import json
import matplotlib.pyplot as plt

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG
# ==========================================
# Đường dẫn file log JSON mà file train xuất ra
HISTORY_FILE = r'D:\CCTV_VMR\AI\data\train_history.json'
# Đường dẫn lưu file ảnh đồ thị đầu ra
OUTPUT_IMAGE = r'D:\CCTV_VMR\AI\data\crac_advanced_metrics.png'

# Kịch bản kiểm tra tệp tin log đầu vào
if not os.path.exists(HISTORY_FILE):
    print(f"❌ LỖI: Không tìm thấy file lịch sử tại: {HISTORY_FILE}")
    print("👉 Hãy chắc chắn anh đang chạy file 'train_rl_fast.py' phiên bản mới để sinh ra dữ liệu log này.")
    exit()

# ==========================================
# 2. ĐỌC DỮ LIỆU TỪ FILE LOG JSON
# ==========================================
with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
    history = json.load(f)

if len(history) == 0:
    print("⚠️ CẢNH BÁO: File log JSON hiện tại đang trống. Hãy chờ file train chạy xong ít nhất 1 Epoch!")
    exit()

# Bóc tách dữ liệu các trục tọa độ
epochs = [x['epoch'] for x in history]
losses = [x['loss'] for x in history]
rewards = [x['reward'] for x in history]
mious = [x['miou'] for x in history]

print(f"📊 Tìm thấy dữ liệu huấn luyện của {len(epochs)} Epochs. Đang tiến hành dựng biểu đồ...")

# ==========================================
# 3. KHỞI TẠO ĐỒ HỌA MẠNG LƯỚI (PHÂN HỆ 3 SUBPLOTS)
# ==========================================
# Thiết lập kích thước khung hình chuẩn tỉ lệ vàng (18x5 inches) nằm ngang
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('QUY ĐẠO HỘI TỤ VÀ TỐI ƯU HÓA MÔ HÌNH CRAC ADVANCED', fontsize=14, fontweight='bold', y=0.98)

# ----------------------------------------------------------------------
# ĐỒ THỊ 1: QUY ĐẠO TĂNG TRƯỞNG MEAN IOU (CHỈ SỐ ĐÁNH GIÁ CHÍNH)
# ----------------------------------------------------------------------
axs[0].plot(epochs, mious, color='#2ca02c', linestyle='-', linewidth=2, marker='o', label='Mean IoU')
axs[0].axhline(y=0.70, color='r', linestyle='--', alpha=0.7, label='Mục tiêu IoU >= 0.7')
axs[0].set_title('Quỹ Đạo Tăng Trưởng mIoU', fontsize=12, fontweight='bold')
axs[0].set_xlabel('Epoch', fontsize=10)
axs[0].set_ylabel('mIoU Score', fontsize=10)
axs[0].grid(True, linestyle=':', alpha=0.6)
axs[0].legend(loc='lower right')

# ----------------------------------------------------------------------
# ĐỒ THỊ 2: ĐƯỜNG CONG HỘI TỤ LOSS (ĐÁNH GIÁ TOÁN HỌC)
# ----------------------------------------------------------------------
axs[1].plot(epochs, losses, color='#d62728', linestyle='-', linewidth=2, label='Total Loss')
axs[1].set_title('Đường Cong Hội Tụ Loss', fontsize=12, fontweight='bold')
axs[1].set_xlabel('Epoch', fontsize=10)
axs[1].set_ylabel('Loss Value', fontsize=10)
axs[1].grid(True, linestyle=':', alpha=0.6)
axs[1].legend(loc='upper right')

# ----------------------------------------------------------------------
# ĐỒ THỊ 3: BIẾN THIÊN PHẦN THƯỞNG TÍCH LŨY TRUNGBÌNH (REWARD)
# ----------------------------------------------------------------------
axs[2].plot(epochs, rewards, color='#1f77b4', linestyle='-', linewidth=2, label='Avg Reward')
axs[2].set_title('Biến Thiên Phan Thưởng Tích Lũy', fontsize=12, fontweight='bold')
axs[2].set_xlabel('Epoch', fontsize=10)
axs[2].set_ylabel('Reward Points', fontsize=10)
axs[2].grid(True, linestyle=':', alpha=0.6)
axs[2].legend(loc='lower right')

# Căn chỉnh khoảng cách tự động để chống đè chữ giữa các phân hệ đồ thị
plt.tight_layout()

# ==========================================
# 4. LƯU ẢNH CHẤT LƯỢNG IN ẤN & HIỂN THỊ LÊN MÀN HÌNH
# ==========================================
# Xuất file ảnh với độ phân giải cao 300 DPI chống vỡ hình khi chèn Word/In luận văn
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
print(f"✅ Đã xuất file ảnh đồ thị thành công tại: {OUTPUT_IMAGE}")

# Ép hệ thống mở cửa sổ pop-up hiển thị biểu đồ trực tiếp lên màn hình
print("🖥️ Đang mở cửa sổ hiển thị đồ thị...")
plt.show()