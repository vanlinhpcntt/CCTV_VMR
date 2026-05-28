import os
import json
import torch
import cv2
import time
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import torch.nn.functional as F
from tqdm import tqdm

from rl_agent import BoundaryRefinementAgent, apply_action

print("🚀 Khởi động Hệ thống Đánh giá Baseline (Evaluation)...")

# ==========================================
# 1. CẤU HÌNH ĐÁNH GIÁ (CHỌN CHẾ ĐỘ Ở ĐÂY)
# ==========================================
# CHỌN MODE TỪ 1 ĐẾN 3:
# MODE 1: Baseline 1 (CLIP only) - Không ROI, không RL
# MODE 2: Baseline 2 (CLIP + ROI) - Có ROI, không RL
# MODE 3: Baseline 3 (Full Model) - Có ROI, Có RL
EVAL_MODE = 3  # <--- THAY ĐỔI SỐ NÀY ĐỂ ĐỔI BASELINE

TEST_FILE = r'D:\ThucTap\CCTV_VMR\AI\data\dataset_split\test.jsonl'
VIDEO_DIR = r'D:\ThucTap\CCTV_VMR\AI\data\video'
RL_WEIGHT_PATH = './checkpoints/rl_clip_agent_epoch_100.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Thiết lập cờ (Flags) dựa trên Mode
USE_ROI = EVAL_MODE >= 2
USE_RL = EVAL_MODE == 3

print(f"🎯 ĐANG CHẠY ĐÁNH GIÁ: BASELINE {EVAL_MODE}")
print(f"   [+] Bật YOLO ROI Masking: {'CÓ' if USE_ROI else 'KHÔNG'}")
print(f"   [+] Bật Actor-Critic RL : {'CÓ' if USE_RL else 'KHÔNG'}")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH VÀ QUÉT VIDEO
# ==========================================
print("⏳ Đang tải các mô hình...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

if USE_ROI:
    yolo_model = YOLO('yolov8n.pt') 

if USE_RL:
    rl_agent = BoundaryRefinementAgent(visual_dim=512, text_dim=512).to(device)
    if os.path.exists(RL_WEIGHT_PATH):
        rl_agent.load_state_dict(torch.load(RL_WEIGHT_PATH, map_location=device))
    rl_agent.eval()

print("🔍 Đang lập chỉ mục Video...")
video_index = {}
for root, dirs, files in os.walk(VIDEO_DIR):
    for f in files:
        if f.endswith(('.mp4', '.avi', '.mkv')):
            video_index[f] = os.path.join(root, f)

# ==========================================
# 3. HÀM BỔ TRỢ
# ==========================================
def calculate_iou(pred_span, gt_span):
    inter_start = max(pred_span[0], gt_span[0])
    inter_end = min(pred_span[1], gt_span[1])
    inter = max(0, inter_end - inter_start)
    union = (pred_span[1] - pred_span[0]) + (gt_span[1] - gt_span[0]) - inter
    return inter / union if union > 0 else 0.0

def apply_roi_masking(frame, results):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    has_object = False
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in [0, 2, 3, 5, 7]: 
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                has_object = True
    if not has_object: return frame
    return cv2.bitwise_and(frame, frame, mask=mask)

def process_video(video_path, query_text):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    
    # Text Encoding
    with torch.no_grad():
        t_inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        t_outputs = clip_model.text_model(**t_inputs)
        t_feat = clip_model.text_projection(t_outputs.pooler_output)
        t_feat = t_feat / t_feat.norm(p=2, dim=-1, keepdim=True)
    
    WINDOW_SIZE = 5.0
    STRIDE = 3.0
    curr_start = 0.0
    best_score = -1
    best_candidate = None
    
    # Quét toàn bộ video để tìm cửa sổ trùng khớp nhất
    while curr_start + WINDOW_SIZE <= duration:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((curr_start + WINDOW_SIZE/2) * fps))
        ret, frame = cap.read()
        if not ret: break
        
        # Tiền xử lý (Bật/Tắt theo Mode)
        if USE_ROI:
            res = yolo_model(frame, verbose=False)
            frame = apply_roi_masking(frame, res)
            
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Image Encoding
        with torch.no_grad():
            v_inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
            v_outputs = clip_model.vision_model(**v_inputs)
            v_feat = clip_model.visual_projection(v_outputs.pooler_output)
            v_feat = v_feat / v_feat.norm(p=2, dim=-1, keepdim=True)
            
            score = F.cosine_similarity(v_feat, t_feat).item()
            
            if score > best_score:
                best_score = score
                best_candidate = {
                    'start': curr_start, 'end': curr_start + WINDOW_SIZE,
                    'v_feat': v_feat, 't_feat': t_feat
                }
        curr_start += STRIDE
        
    cap.release()
    if not best_candidate: return None, 0
    
    # Tinh chỉnh bằng Actor-Critic (Bật/Tắt theo Mode)
    final_span = [best_candidate['start'], best_candidate['end']]
    if USE_RL:
        current_span_tensor = torch.tensor([0.2, 0.8], device=device) # Tương đương -1.5s và +1.5s từ tâm
        with torch.no_grad():
            for _ in range(5):
                action_probs, _ = rl_agent(best_candidate['v_feat'], best_candidate['t_feat'], current_span_tensor.unsqueeze(0))
                action = torch.argmax(action_probs).item()
                if action == 4: break
                current_span_tensor = apply_action(current_span_tensor, action, delta=0.05).to(device)
        
        # Đổi ngược từ tọa độ 0-1 ra giây thực tế
        center_time = (best_candidate['start'] + best_candidate['end']) / 2
        f_start = center_time + (current_span_tensor[0].item() - 0.5) * WINDOW_SIZE * 2
        f_end = center_time + (current_span_tensor[1].item() - 0.5) * WINDOW_SIZE * 2
        final_span = [max(0, f_start), min(duration, f_end)]
        
    latency = time.time() - start_time
    return final_span, latency

# ==========================================
# 4. CHẠY ĐÁNH GIÁ (INFERENCE)
# ==========================================
if not os.path.exists(TEST_FILE):
    print(f"❌ LỖI: Không tìm thấy tập test {TEST_FILE}")
    exit()

with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_lines = [line.strip() for line in f if line.strip()]

total_iou = 0
r1_05_hits = 0
r1_07_hits = 0
total_latency = 0
valid_samples = 0

progress_bar = tqdm(test_lines, desc="Đang đánh giá")

for line in progress_bar:
    try:
        data = json.loads(line)
    except: continue
        
    base_name = os.path.basename(data.get('clip_id', ''))
    if not base_name.endswith(('.mp4', '.avi')): base_name += '.mp4'
    query = data.get('query_vi')
    gt_start = float(data.get('t_start', 0.0))
    gt_end = float(data.get('t_end', 15.0))
    
    video_path = video_index.get(base_name)
    if not video_path or not query: continue
    
    pred_span, latency = process_video(video_path, query)
    if not pred_span: continue
    
    iou = calculate_iou(pred_span, [gt_start, gt_end])
    
    total_iou += iou
    if iou >= 0.5: r1_05_hits += 1
    if iou >= 0.7: r1_07_hits += 1
    total_latency += latency
    valid_samples += 1
    
    progress_bar.set_postfix({"mIoU": f"{total_iou/valid_samples:.3f}"})

# ==========================================
# 5. TỔNG KẾT SỐ LIỆU ĐỂ ĐIỀN BÁO CÁO
# ==========================================
if valid_samples > 0:
    mIoU = total_iou / valid_samples
    r1_05 = r1_05_hits / valid_samples
    r1_07 = r1_07_hits / valid_samples
    avg_latency = total_latency / valid_samples
    
    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ BASELINE {EVAL_MODE}:")
    print(f"   - mIoU           : {mIoU:.4f}")
    print(f"   - R@1 (IoU=0.5)  : {r1_05:.4f}")
    print(f"   - R@1 (IoU=0.7)  : {r1_07:.4f}")
    print(f"   - Avg Latency    : {avg_latency:.2f}s")
    print("="*50)
else:
    print("⚠️ Không có mẫu hợp lệ nào được đánh giá!")