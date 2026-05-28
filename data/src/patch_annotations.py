import json
import os

OLD_ANN = r"D:\CCTV_VMR\data\index\merge\annotations_new.jsonl"
NEW_ANN = r"D:\CCTV_VMR\data\index\merge\train_ready_annotations.jsonl"

if not os.path.exists(OLD_ANN):
    print(f"❌ Không tìm thấy file nhãn gốc tại: {OLD_ANN}")
    exit()

ready_records = []

with open(OLD_ANN, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        data = json.loads(line)
        
        # Lấy tọa độ gốc [x, y, w, h] từ tool cũ của anh
        bbox = data.get("bbox", [])
        if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # 1. Tự động quy đổi sang định dạng hình hộp chuẩn tuyệt đối
            xmin = int(x)
            ymin = int(y)
            xmax = int(x + w)
            ymax = int(y + h)
            real_bbox = [xmin, ymin, xmax, ymax]
            
            # 2. Tự động giả lập Polygon mịn bằng 4 góc hộp biên để thuật toán không bị lỗi
            # Khớp nối kín theo chiều kim đồng hồ
            simulated_polygon = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax]
            ]
            
            # Cập nhật lại vào bản ghi
            data["bbox"] = real_bbox
            data["polygon"] = simulated_polygon
            
            # Đồng bộ hóa trường thời gian cho file train_rl_fast.py dễ đọc
            data["t_start"] = data.get("segment", [0.0, 10.0])[0]
            data["t_end"] = data.get("segment", [0.0, 10.0])[1]
            data["query"] = data.get("query_vi", "Phương tiện di chuyển.")
            
        ready_records.append(data)

# Ghi đè hoặc xuất ra file mới chuẩn hóa
with open(NEW_ANN, 'w', encoding='utf-8') as f:
    for rec in ready_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Đã chuẩn hóa xong {len(ready_records)} câu nhãn!")
print(f"👉 File sẵn sàng huấn luyện nằm tại: {NEW_ANN}")
