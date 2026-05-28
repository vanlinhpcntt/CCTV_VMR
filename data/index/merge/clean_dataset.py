import os
import json

print(f"\n{'='*50}\n🧹 KHỞI ĐỘNG CÔNG CỤ DỌN DẸP DỮ LIỆU (DATA CLEANER)\n{'='*50}")

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN 
# ==========================================
ANN_FILE_OLD = r"D:\CCTV_VMR\data\ann\cam02\225\annotations.jsonl"
ANN_FILE_NEW = r"D:\CCTV_VMR\data\ann\cam02\225\clean_annotations.jsonl"
VIDEO_DIR = r"D:\CCTV_VMR\data\event_clips\cam02\225" 

if not os.path.exists(ANN_FILE_OLD):
    print(f"❌ LỖI: Không tìm thấy file nhãn gốc tại:\n   {ANN_FILE_OLD}")
    exit(1)
if not os.path.exists(VIDEO_DIR):
    print(f"❌ LỖI: Không tìm thấy thư mục video tại:\n   {VIDEO_DIR}")
    exit(1)

# ==========================================
# 2. QUÉT TRƯỚC TẤT CẢ VIDEO THỰC TẾ
# ==========================================
print("⏳ Đang quét sâu các thư mục để gom danh sách video thực tế...")
actual_videos = set()
for root, dirs, files in os.walk(VIDEO_DIR):
    for file in files:
        if file.endswith('.mp4'):
            actual_videos.add(file)

print(f"   -> Đã tìm thấy {len(actual_videos)} file video (.mp4) trong ổ cứng.\n")

# ==========================================
# 3. TIẾN HÀNH LỌC VÀ XUẤT FILE MỚI
# ==========================================
print("⏳ Đang tiến hành lọc nhãn lỗi và xuất file mới...")

valid_count = 0
removed_json_error = 0
removed_missing_video = 0

# Mở file cũ để đọc (dùng utf-8-sig để chống lỗi BOM), mở file mới để ghi
with open(ANN_FILE_OLD, 'r', encoding='utf-8-sig') as infile, \
     open(ANN_FILE_NEW, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        clean_line = line.strip()
        if not clean_line: 
            continue
            
        try:
            # Thử parse JSON xem có bị lỗi cú pháp không
            data = json.loads(clean_line)
            clip_id = data.get('clip_id', '')
            
            base_name = os.path.basename(clip_id)
            if base_name and not base_name.endswith('.mp4'):
                base_name += '.mp4'
            
            # KIỂM TRA ĐIỀU KIỆN SỐNG CÒN: Video có tồn tại không?
            if base_name in actual_videos:
                # Nếu mọi thứ đều OK, ghi nguyên dòng gốc vào file mới
                outfile.write(clean_line + "\n")
                valid_count += 1
            else:
                # Nhãn có nhưng video không có -> Bỏ qua và đếm
                removed_missing_video += 1
                
        except json.JSONDecodeError:
            # Lỗi cú pháp JSON (dòng rác, thiếu ngoặc...) -> Bỏ qua và đếm
            removed_json_error += 1

# ==========================================
# 4. IN BÁO CÁO NGHIỆM THU
# ==========================================
print(f"\n✅ ĐÃ HOÀN TẤT VIỆC DỌN DẸP!")
print(f"📊 BÁO CÁO KẾT QUẢ:")
print(f"   🟢 Giữ lại thành công: {valid_count} nhãn hợp lệ (Chuẩn 100% có video).")
print(f"   🔴 Đã xóa bỏ vĩnh viễn:")
print(f"      - {removed_json_error} dòng bị lỗi định dạng JSON/rác.")
print(f"      - {removed_missing_video} dòng trỏ vào video không tồn tại.")

print("-" * 50)
print(f"👉 FILE NHÃN SẠCH ĐÃ ĐƯỢC LƯU TẠI:\n   {ANN_FILE_NEW}")
print(f"{'='*50}")