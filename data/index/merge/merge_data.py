import os
import json

print(f"\n{'='*50}\n🔗 CÔNG CỤ GỘP DỮ LIỆU NHÃN (MERGE ANNOTATIONS)\n{'='*50}")

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ TÊN FILE
# ==========================================
ANN_ROOT_DIR = r"D:\CCTV_VMR\data\ann"
OUTPUT_FILE = r"D:\CCTV_VMR\data\index\merge\annotations_new.jsonl"

# ⚠️ QUAN TRỌNG: Xác định tên file nhãn bạn muốn tìm để gộp.
# Nếu bạn đã chạy tool dọn rác trước đó, hãy để là 'clean_annotations.jsonl'
# Nếu muốn gộp file gốc, đổi thành 'annotations.jsonl'
TARGET_FILENAME = "clean_annotations.jsonl" 

if not os.path.exists(ANN_ROOT_DIR):
    print(f"❌ LỖI: Không tìm thấy thư mục gốc tại:\n   {ANN_ROOT_DIR}")
    exit(1)

# ==========================================
# 2. QUÉT SÂU TÌM CÁC FILE NHÃN (DEEP SCAN)
# ==========================================
print(f"⏳ Đang quét sâu vào thư mục {ANN_ROOT_DIR}\n   để tìm các file '{TARGET_FILENAME}'...")
files_to_merge = []

for root, dirs, files in os.walk(ANN_ROOT_DIR):
    for file in files:
        if file == TARGET_FILENAME:
            full_path = os.path.join(root, file)
            files_to_merge.append(full_path)

if not files_to_merge:
    print(f"⚠️ KHÔNG TÌM THẤY file '{TARGET_FILENAME}' nào trong các thư mục con!")
    print("   -> Vui lòng kiểm tra lại biến TARGET_FILENAME trong code.")
    exit(0)

print(f"   -> Đã tìm thấy {len(files_to_merge)} file nhãn sẵn sàng để gộp.\n")

# ==========================================
# 3. TIẾN HÀNH GỘP FILE (KÈM KIỂM TRA LỖI)
# ==========================================
print("⏳ Đang tiến hành gộp dữ liệu...")
total_lines_merged = 0
bad_lines = 0

with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    for file_path in files_to_merge:
        # Lấy tên thư mục chứa file (VD: cam01, cam02) để in log cho dễ nhìn
        camera_name = os.path.basename(os.path.dirname(file_path)) 
        print(f"   ➕ Đang hút dữ liệu từ: [{camera_name}]")
        
        with open(file_path, 'r', encoding='utf-8-sig') as infile:
            for line in infile:
                clean_line = line.strip()
                if not clean_line:
                    continue
                
                try:
                    # Parse JSON để đảm bảo dòng chữ này chuẩn xác 100%
                    data = json.loads(clean_line)
                    
                    # (Tính năng phụ) Đánh dấu camera_id vào data để sau này truy xuất nguồn gốc dễ hơn
                    if 'camera_id' not in data:
                        data['camera_id'] = camera_name
                        
                    # Ghi dòng dữ liệu sạch vào file tổng
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                    total_lines_merged += 1
                except json.JSONDecodeError:
                    bad_lines += 1

# ==========================================
# 4. BÁO CÁO KẾT QUẢ
# ==========================================
print(f"\n✅ ĐÃ GỘP THÀNH CÔNG!")
print(f"📊 BÁO CÁO:")
print(f"   - Số lượng thư mục/file đã quét qua: {len(files_to_merge)}")
print(f"   - TỔNG SỐ NHÃN TRONG DATASET MỚI:  {total_lines_merged} nhãn.")
if bad_lines > 0:
    print(f"   - ⚠️ Đã chặn đứng {bad_lines} dòng rác/lỗi trong quá trình gộp.")
print("-" * 50)
print(f"👉 FILE NHÃN TỔNG ĐÃ LƯU TẠI:\n   {OUTPUT_FILE}")
print(f"{'='*50}")