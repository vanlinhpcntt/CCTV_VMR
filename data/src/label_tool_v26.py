import cv2
import json
import os
import glob

# =================================================================================
# 1. CẤU HÌNH HỆ THỐNG
# =================================================================================
DATA_DIR = "data"
# Đường dẫn chứa video clip đầu vào
CLIPS_DIR = r"D:\ThucTap\CCTV_VMR\data\event_clips\NgaTuCayMe\12\12.33"
# Đường dẫn đích để lưu file annotations.jsonl
ANN_DIR = r"D:\ThucTap\CCTV_VMR\data\ann\NgaTuCayMe\12\12.33"

CAM_IDS = [""] 
OUTPUT_FILE = "annotations.jsonl"
REVIEW_MODE = False  

# =================================================================================
# 2. TỪ ĐIỂN GIAO THÔNG TOÀN THƯ (V26 - TỐI ƯU HÓA)
# =================================================================================
VN_TO_EN = {
    # --- A. TỪ ĐỆM & CỤM TỪ (XỬ LÝ NGỮ PHÁP) ---
    "màu": "", "mau": "", "có": "", "co": "", "bị": "", "chiếc": "", "cái": "",
    "phủ": "", "phu": "", "đang": "",
    
    # Cụm từ thùng xe & cẩu
    "khung": "cage", "thùng khung": "cage box", "khung sắt": "cage box", "thùng rào": "cage box",
    "thùng phủ bạt đen": "black tarpaulin box", "thùng bạt đen": "black tarpaulin box",
    "thùng phủ bạt xanh": "blue tarpaulin box", "thùng bạt xanh": "blue tarpaulin box",
    "thùng phủ bạt": "tarpaulin box", "phủ bạt": "tarpaulin box", "phu bat": "tarpaulin box", "có bạt": "tarpaulin box",
    "thùng bạt": "tarpaulin box", "bạt": "tarpaulin box",
    "thùng kín": "enclosed box", "thùng lửng": "open bed", 
    "thùng rỗng": "empty box", "thùng đông lạnh": "refrigerated box",
    "thùng xốp": "styrofoam box", "thùng": "box",
      
    "cẩu": "crane", "cau": "crane", "cần cẩu": "crane",
    "cẩu vàng": "yellow crane", "cẩu xanh": "blue crane", "cẩu đỏ": "red crane", 
    "cẩu trắng": "white crane", "cẩu đen": "black crane", "cẩu cam": "orange crane",
    
    # Màu + Bạt (Ghép sẵn)
    "phủ bạt màu đen": "black tarpaulin box", "phủ bạt đen": "black tarpaulin box", "bạt đen": "black tarpaulin box",
    "phủ bạt màu xanh": "blue tarpaulin box", "phủ bạt xanh": "blue tarpaulin box", "bạt xanh": "blue tarpaulin box",
    "bạt cam": "orange tarpaulin box", "bạt vàng": "yellow tarpaulin box", "bạt xám": "grey tarpaulin box",

    # --- B. BẢNG MÀU CHI TIẾT (COLORS) ---
    "đen": "black", "den": "black", "trắng": "white", "trang": "white",
    "đỏ": "red", "do": "red", "đỏ tươi": "bright red", "đỏ đô": "dark red", "đỏ mận": "burgundy", "đỏ gạch": "brick red",
    "xanh": "blue", "xanh dương": "blue", "xanh nước biển": "blue", "xanh da trời": "sky blue", "xanh mực": "navy blue",
    "xanh lá": "green", "xanh lục": "green", "xanh bộ đội": "army green", "xanh rêu": "moss green", "xanh chuối": "lime green",
    "xanh ngọc": "turquoise", "xanh lơ": "cyan",
    "vàng": "yellow", "vang": "yellow", "vàng chanh": "lime yellow", "vàng đồng": "gold", "vàng cát": "beige",
    "bạc": "silver", "bac": "silver", "ghi": "grey", "màu ghi": "grey", "xám": "grey", "lông chuột": "dark grey",
    "nâu": "brown", "nâu đất": "earth brown", "cà phê": "coffee",
    "cam": "orange", 
    "hồng": "pink", "hồng nhạt": "light pink", "hồng đậm": "hot pink",
    "tím": "purple", "tím than": "dark violet",
    "kem": "cream", "be": "beige", "sữa": "milky white",
    "nhiều màu": "multicolor", "kẻ ca rô": "plaid", "sọc": "striped", "tem đấu": "patterned", "kẻ ca ro": "plaid", "kẻ caro": "plaid",

    # --- C. PHƯƠNG TIỆN (VEHICLES) ---
    "xe tải": "truck", "xe tai": "truck", "tải": "truck",
    "xe container": "container truck", "container": "container truck", "cont": "container truck", "xe công": "container truck",
    "đầu kéo": "tractor head", "xe đầu kéo": "tractor head",
    "xe cẩu": "crane truck", "xe cau": "crane truck",
    "xe bồn": "tanker truck", "xe chở xăng": "fuel tanker", "xe téc": "tanker truck",
    "xe trộn": "concrete mixer", "xe bê tông": "concrete mixer",
    "xe rác": "garbage truck", "xe môi trường": "garbage truck",
    "xe cứu hỏa": "fire truck", "xe cứu hộ": "tow truck", "xe ba gác": "cargo tricycle",
    "xe con": "car", "xe hơi": "car", "ô tô": "car", "oto": "car", "taxi": "taxi",
    "xe khách": "coach", "xe buýt": "bus", "xe 16 chỗ": "van", "xe 29 chỗ": "minibus", "bán tải": "pickup truck", "xe bán tải": "pickup truck",
    "xe máy": "motorcycle", "tay ga": "scooter", "xe số": "motorcycle", "xe điện": "electric bike", "xe đạp": "bicycle",
    "xe ba gác": "freight tricycle", "xe ba bánh": "freight tricycle", "ba gác": "freight tricycle",

    # --- D. ĐỐI TƯỢNG & PHỤ KIỆN ---
    "người": "person", "nguoi": "person", "tài xế": "driver",
    "mũ": "helmet", "nón": "helmet", "không nón": "no helmet", "không đội nón": "no helmet", "không đội mũ": "no helmet", "không mũ": "no helmet",
    "khẩu trang": "mask", "đeo khẩu trang": "wearing a mask",
    "balo": "backpack", "cặp": "backpack", "túi": "bag",
    "áo mưa": "raincoat", "áo khoác": "jacket", "áo sọc": "striped shirt", "áo kẻ": "plaid shirt"
}

# =================================================================================
# 3. HÀM XỬ LÝ
# =================================================================================
def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def append_jsonl(filepath, data):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def get_done_clips(filepath):
    if not os.path.exists(filepath): return set()
    done = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try: d = json.loads(line); done.add(d['clip_id'])
            except: pass
    return done

def translate_word(word):
    if not word: return ""
    word = word.lower().strip().replace(',', ' ').replace(' và ', ' ')
    if word in VN_TO_EN: return VN_TO_EN[word]
    parts = word.split()
    translated_parts = []
    for p in parts:
        trans = VN_TO_EN.get(p, p) 
        translated_parts.append(trans)
    return " ".join(translated_parts)

def process_action_smart(action_vi):
    if not action_vi: return "", ""
    act_lower = action_vi.lower()
    en_parts = []

    # 1. Xử lý các cụm phức tạp trước (Complex Actions)
    if "đi thẳng ra xa camera" in act_lower:
        if "rẽ phải" in act_lower:
            en_parts.append("going straight away from the camera and then turning right")
        elif "đèn đỏ" in act_lower or "chờ đèn đỏ" in act_lower:
            en_parts.append("going straight away from the camera and stopping at a red light")
        elif "ngã tư" in act_lower:
            en_parts.append("going straight away from the camera through the intersection")
        else:
            en_parts.append("going straight away from the camera")
    # 2. Xử lý hướng di chuyển qua ngã tư
    elif "ngã tư" in act_lower:
        if "phải sang trái" in act_lower: en_parts.append("moving from right to left through the intersection")
        elif "trái sang phải" in act_lower: en_parts.append("moving from left to right through the intersection")
        else: en_parts.append("crossing the intersection")

    # 3. Các hành động đơn lẻ (Nếu chưa có trong en_parts thì mới check tiếp hoặc dùng if độc lập)
    # Dùng list các từ khóa đơn giản để tránh trùng lặp
    simple_map = {
        "ngược chiều": "driving against traffic",
        "vỉa hè": "driving on the sidewalk",
        "vượt đèn": "running a red light",
        "rẽ trái": "turning left",
        "rẽ phải": "turning right",
        "quay đầu": "making a u-turn",
        "đeo balo": "wearing a backpack",
        "đeo túi": "carrying a bag",
    }
    
    for vi, en in simple_map.items():
        if vi in act_lower and en not in " ".join(en_parts):
            en_parts.append(en)

    # 4. Phụ kiện & Chở hàng (Cái này có thể đi kèm hành động chính nên dùng if riêng)
    if "chở người" in act_lower or "chở 1 người" in act_lower: en_parts.append("carrying a passenger")
    if "chở 2 người" in act_lower or "kẹp ba" in act_lower: en_parts.append("carrying two passengers")
    if "chở hàng" in act_lower: en_parts.append("carrying goods")

    # Fallback
    if not en_parts:
        trans = translate_word(act_lower)
        if trans: en_parts.append(trans)

    vi_out = action_vi
    if "từ xa về gần" in act_lower and "camera" not in act_lower:
        vi_out = vi_out.replace("từ xa về gần", "từ xa về gần camera")

    # Clean up: dùng set để loại bỏ trùng lặp nếu có và nối lại
    return vi_out, ", ".join(dict.fromkeys(en_parts))

def generate_queries(group, attrs):
    q_vi = ""; q_en = ""
    action_raw = attrs.get("action", "").strip()
    act_vi, act_en = process_action_smart(action_raw)

    # === NHÓM 1: XE MÁY / XE ĐẠP (Đã fix lỗi lặp từ "motorcyclist - motorcycle") ===
    if group == "1":
        shirt = attrs.get("shirt", ""); shirt_en = translate_word(shirt)
        helmet = attrs.get("helmet", ""); helmet_en = translate_word(helmet)
        bike = attrs.get("bike_type", "xe máy"); bike_en = translate_word(bike)
        color = attrs.get("bike_color", ""); color_en = translate_word(color)
        
        # --- TIẾNG VIỆT ---
        parts_vi = []
        if shirt: parts_vi.append(f"Người mặc áo {shirt}")
        else: parts_vi.append("Người")
        
        if helmet:
            val = helmet.lower().replace("mũ", "nón")
            if any(x in val for x in ["không", "ko"]): parts_vi.append("không đội nón")
            elif "nón" in val: parts_vi.append(f"đội {val}")
            else: parts_vi.append(f"đội nón {val}")
        
        veh_str = f"chạy {bike}" if bike.lower().startswith("xe") else f"chạy xe {bike}"
        if color: veh_str += f" màu {color}"
        parts_vi.append(veh_str)
        if act_vi: parts_vi.append(act_vi)
        q_vi = " ".join(parts_vi) + "."

        # --- TIẾNG ANH ---
        base_subj = "A person"
        article = "an" if shirt_en.lower().startswith(('a', 'e', 'i', 'o', 'u')) else "a"
        subj = f"{base_subj} in {article} {shirt_en} shirt" if shirt_en else base_subj

        helm = "without a helmet" if helmet_en == "no helmet" else (f"wearing a {helmet_en} helmet" if helmet_en else "")
        veh = f"{color_en} {bike_en if bike_en else 'motorcycle'}"
        
        q_en = f"{subj} {helm} riding a {veh} {act_en}."

    # === NHÓM 2: XE CON / KHÁCH ===
    elif group == "2":
        ctype = attrs.get("type", "xe con"); ctype_en = translate_word(ctype)
        color = attrs.get("color", ""); color_en = translate_word(color)
        
        final_type_vi = ctype[0].upper() + ctype[1:] if ctype.lower().startswith("xe") else f"Xe {ctype}"
        q_vi = f"{final_type_vi} màu {color}"
        if act_vi: q_vi += f" {act_vi}"
        q_vi += "."

        q_en = f"A {color_en} {ctype_en} {act_en}."

    # === NHÓM 3: XE TẢI / CÔNG / CẨU (Chuẩn ngữ pháp "WITH") ===
    elif group == "3":
        ctype = attrs.get("type", "xe tải"); ctype_en = translate_word(ctype)
        head_col = attrs.get("head_color", "").strip()
        box_col = attrs.get("box_val", "").strip()
        
        head_en = translate_word(head_col)
        box_en = translate_word(box_col)
        
      # --- VIETNAMESE ---
        q_vi = f"{ctype.capitalize()}"
        if head_col:
            prefix = "đầu màu"
            if any(w in head_col.lower() for w in ["đầu", "cabin"]): prefix = ""
            elif "màu" in head_col.lower(): prefix = "đầu"
            q_vi += f" {prefix} {head_col}"
            
        if box_col:
            prefix = "thùng xe"
            box_lower = box_col.lower()
            if "thùng" in box_lower: 
                prefix = ""
            elif "cẩu" in box_lower or "cẩu" in ctype.lower(): 
                prefix = "có"
            # Nếu gặp các loại thùng đặc thù thì chỉ ghép chữ "thùng" (VD: thùng khung, thùng lửng, thùng kín)
            elif any(w in box_lower for w in ["khung", "rào", "kín", "lửng", "rỗng", "xốp"]):
                prefix = "thùng"
            # Nếu chỉ gõ mỗi màu sắc (bạc, đỏ, xanh...) thì ghép "thùng xe màu"
            elif not any(w in box_lower for w in ["bạt", "màu", "đông lạnh"]):
                prefix = "thùng xe màu"
                
            q_vi += f" {prefix} {box_col}"

        if act_vi: q_vi += f" {act_vi}"
        q_vi = " ".join(q_vi.split()) + "."

        # --- ENGLISH ---
        veh_core = ctype_en
        if head_en: veh_core = f"{head_en} {ctype_en}"
        
        box_part = ""
        if box_en:
            if "box" not in box_en and "tarpaulin" not in box_en and "crane" not in box_en and "bed" not in box_en:
                box_en += " box"
            box_part = f"with {box_en}"

        q_en = f"A {veh_core} {box_part} {act_en}."

    # Clean up extra spaces
    q_vi = " ".join(q_vi.split())
    q_en = " ".join(q_en.split())
    return q_vi, q_en

# =================================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# =================================================================================
def label_clip(cam_id, clip_path, output_path):
    filename = os.path.basename(clip_path)
    clip_id = filename.split(".")[0]
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS); frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = round(frame_count / fps, 2) if fps > 0 else 0.0

    print(f"\n🎥 ĐANG XỬ LÝ: {filename} (Time: {total_duration}s)")
    print("="*65)
    print(" CÁC PHÍM TẮT ĐIỀU KHIỂN:")
    print(" [SPACE]: Dừng/Phát    [A]: Gán nhãn đối tượng")
    print(" [N]: Bỏ qua clip      [R]: Lùi lại clip trước    [ESC]: Thoát")
    print("="*65)

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            display_frame = cv2.resize(frame, (960, 540))
        
        if paused:
            cv2.putText(display_frame, "PAUSED - PRESS 'A' TO LABEL", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Label Tool V26 (Ban Hoan Chinh Tuyet Doi)", display_frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27: return "EXIT"
        elif key == ord('n'): break
        elif key == ord('r'): return "PREV"
        elif key == ord(' '): paused = not paused
        elif key == ord('a') and paused:
            print("\n" + "*"*40)
            print("1. XE MÁY, XE ĐẠP (2-Wheelers)")
            print("2. XE CON / KHÁCH")
            print("3. XE TẢI / CÔNG / CẨU (Dynamic Prompt)")
            print("*"*40)
            
            while True:
                choice = input(">> Chọn (1/2/3): ").strip()
                if choice in ['1', '2', '3']: break
            
            print(f">> Vẽ Box. ENTER khi xong.")
            bbox = cv2.selectROI("Label Tool V26 (Ban Hoan Chinh Tuyet Doi)", display_frame, fromCenter=False, showCrosshair=True)
            h_orig, w_orig = frame.shape[:2]
            real_bbox = [int(bbox[0]*(w_orig/960)), int(bbox[1]*(h_orig/540)), int(bbox[2]*(w_orig/960)), int(bbox[3]*(h_orig/540))]
            
            attrs = {}
            cls_name = "vehicle" 
            if choice == "1":
                cls_name = "motorcyclist"
                attrs['shirt'] = input("1. Màu áo (vd: ke ca ro): ").strip()
                attrs['helmet'] = input("2. Nón/Mũ (vd: trang / khong non): ").strip()
                attrs['bike_color'] = input("3. Màu xe (vd: den): ").strip()
                attrs['bike_type'] = input("4. Loại xe (vd: xe so, tay ga, xe dap): ").strip()

                # ===== MENU HƯỚNG ĐI =====
                print("\n5. Hướng di chuyển:")
                print("1. đi thẳng ra xa camera qua ngã tư")
                print("2. đi thẳng ra xa camera rồi rẽ phải")
                print("3. đi từ phải sang trái qua ngã tư")
                print("4. đi từ trái sang phải qua ngã tư")
                print("5. đi thẳng ra xa camera rồi dừng chờ đèn đỏ")
                print("6. đi từ trên xuống và rẽ phải qua ngã tư")
                print("7. đi từ trên xuống và rẽ trái qua ngã tư")
                print("8. đi từ trên xuống và đi thẳng qua ngã tư")
                print("9. hành động khác (nhập tự do)")

                action_map = {
                    "1": "đi thẳng ra xa camera qua ngã tư",
                    "2": "đi thẳng ra xa camera rồi rẽ phải",
                    "3": "đi từ phải sang trái qua ngã tư",
                    "4": "đi từ trái sang phải qua ngã tư",
                    "5": "đi thẳng ra xa camera rồi dừng chờ đèn đỏ",
                    "6": "đi từ trên xuống và rẽ phải qua ngã tư",
                    "7": "đi từ trên xuống và rẽ trái qua ngã tư",
                    "8": "đi từ trên xuống và đi thẳng qua ngã tư"
                }

                choice_action = input("Chọn hướng (1-9): ").strip()
                attrs['action'] = action_map.get(choice_action, "")
                if choice_action == "9":
                    attrs['action'] = input(">> Nhập hành động tự do: ").strip()
                else:
                    attrs['action'] = action_map.get(choice_action, choice_action)


            elif choice == "2":
                cls_name = "car"
                attrs['type'] = input("1. Loại xe (vd: xe con): ").strip()
                attrs['color'] = input("2. Màu xe (vd: trang): ").strip()

                # ===== MENU HƯỚNG ĐI =====
                print("\n3. Hướng di chuyển:")
                print("1. đi thẳng ra xa camera qua ngã tư")
                print("2. đi thẳng ra xa camera rồi rẽ phải")
                print("3. đi từ phải sang trái qua ngã tư")
                print("4. đi từ trái sang phải qua ngã tư")
                print("5. đi thẳng ra xa camera rồi dừng chờ đèn đỏ")
                print("6. đi từ trên xuống và rẽ phải qua ngã tư")
                print("7. đi từ trên xuống và rẽ trái qua ngã tư")
                print("8. đi từ trên xuống và đi thẳng qua ngã tư")
                
                action_map = {
                    "1": "đi thẳng ra xa camera qua ngã tư",
                    "2": "đi thẳng ra xa camera rồi rẽ phải",
                    "3": "đi từ phải sang trái qua ngã tư",
                    "4": "đi từ trái sang phải qua ngã tư",
                    "5": "đi thẳng ra xa camera rồi dừng chờ đèn đỏ",
                    "6": "đi từ trên xuống và rẽ phải qua ngã tư",
                    "7": "đi từ trên xuống và rẽ trái qua ngã tư",
                    "8": "đi từ trên xuống và đi thẳng qua ngã tư"
                }

                choice_action = input("Chọn hướng (1-8): ").strip()
                attrs['action'] = action_map.get(choice_action, "")


            elif choice == "3":
                cls_name = "truck"
                v_type = input("1. Loại xe (vd: xe tải, container, xe cẩu): ").strip()
                attrs['type'] = v_type
                attrs['head_color'] = input("2. Màu ĐẦU xe (vd: xanh): ").strip()

                # Dynamic Prompt thông minh
                if "cẩu" in v_type.lower():
                    box_input = input("3. Màu CẨU (vd: vàng, đỏ): ").strip()
                    if box_input and "cẩu" not in box_input.lower():
                        box_input = "cẩu " + box_input
                    attrs['box_val'] = box_input
                else:
                    attrs['box_val'] = input("3. Màu/Kiểu THÙNG (vd: phủ bạt đen, xanh): ").strip()

                # ===== MENU HƯỚNG ĐI =====
                print("\n4. Hướng di chuyển:")
                print("1. Đi thẳng ra xa camera qua ngã tư")
                print("2. Đi thẳng ra xa camera rồi rẽ phải")
                print("3. Đi từ phải sang trái qua ngã tư")
                print("4. Đi từ trái sang phải qua ngã tư")
                print("5. Đi thẳng ra xa camera rồi dừng chờ đèn đỏ")
                print("6. đi từ trên xuống và rẽ phải qua ngã tư")
                print("7. đi từ trên xuống và rẽ trái qua ngã tư")
                print("8. đi từ trên xuống và đi thẳng qua ngã tư")

                action_map = {
                    "1": "Đi thẳng ra xa camera qua ngã tư",
                    "2": "Đi thẳng ra xa camera rồi rẽ phải",
                    "3": "Đi từ phải sang trái qua ngã tư",
                    "4": "Đi từ trái sang phải qua ngã tư",
                    "5": "Đi thẳng ra xa camera rồi dừng chờ đèn đỏ",
                    "6": "đi từ trên xuống và rẽ phải qua ngã tư",
                    "7": "đi từ trên xuống và rẽ trái qua ngã tư",
                    "8": "đi từ trên xuống và đi thẳng qua ngã tư"
                }

                choice_action = input("Chọn hướng (1-5): ").strip()
                attrs['action'] = action_map.get(choice_action, "")

            q_vi, q_en = generate_queries(choice, attrs)
            print("-" * 50)
            print(f"🇻🇳 VI: {q_vi}")
            print(f"🇬🇧 EN: {q_en}")
            print("-" * 50)
            
            if input(">> ENTER để lưu (no để hủy): ").lower() != 'no':
                record = {
                    "clip_id": clip_id, "image_path": filename, "timestamp": round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2),
                    "duration": total_duration, "segment": [0.0, total_duration],
                    "bbox": real_bbox, "class_name": cls_name, "attributes": attrs,
                    "query_vi": q_vi, "query_en": q_en
                }
                append_jsonl(output_path, record)
                print("✅ ĐÃ LƯU!")
            
            cont = input("\n>> Tiếp tục clip này? (Enter=Có / n=Qua clip khác): ").strip().lower()
            if cont in ['n', 'no', 'k']: break
            
    cap.release()
    cv2.destroyAllWindows()
    return "NEXT"

def main():
    print("=== TOOL GÁN NHÃN V26 (BẢN HOÀN CHỈNH TUYỆT ĐỐI) ===")
    
    START_CLIP_ID = "video1_e000016"  # 👈 đặt clip bạn muốn resume

    for cam_id in CAM_IDS:
        cam_clip_dir = os.path.join(CLIPS_DIR, cam_id)
        cam_ann_dir = os.path.join(ANN_DIR, cam_id)
        ensure_dir(cam_ann_dir)
        output_path = os.path.join(cam_ann_dir, OUTPUT_FILE)

        done_clips = get_done_clips(output_path)
        clips = sorted(glob.glob(os.path.join(cam_clip_dir, "*.mp4")))

        # ===== TÌM INDEX THEO CLIP ID =====
        idx = 0
        for i, clip_path in enumerate(clips):
            clip_id = os.path.basename(clip_path).split(".")[0]
            if clip_id == START_CLIP_ID:
                idx = i
                break

        force_open_review = False

        while idx < len(clips):
            clip_path = clips[idx]
            clip_id = os.path.basename(clip_path).split(".")[0]

            if not REVIEW_MODE and not force_open_review and clip_id in done_clips:
                idx += 1
                continue

            force_open_review = False

            status = label_clip(cam_id, clip_path, output_path)

            if status == "EXIT":
                return
            elif status == "NEXT":
                idx += 1
            elif status == "PREV":
                idx = max(0, idx - 1)
                force_open_review = True


if __name__ == "__main__":
    main()