import cv2
import numpy as np
import os
import time

# ==========================================
# ⚙️ 超參數設定區 (核心控制台)
# ==========================================
NUM_SLICES = 30           # 想要切分的總份數 (例如 20 或 30)
ENABLE_CLAHE = True       # 是否開啟色彩/光影預處理 (對抗反光)
OUTPUT_BASE_DIR = "training_data" # 存檔的主資料夾名稱

# ==========================================
# 🛠️ 影像處理工具組
# ==========================================

def apply_clahe(img):
    """
    CLAHE (對比受限自適應直方圖均衡化)
    能有效平衡電阻表面的強烈反光與陰影，讓顏色特徵更穩定。
    """
    # 轉到 LAB 色彩空間，針對 L (亮度) 通道進行處理
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # clipLimit 越高對比越強，tileGridSize 是處理區域大小
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 合併回去並轉回 BGR
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def slice_resistor_final(img):
    """
    終極切片函數：執行預處理、精華區裁切、等分切割與自動存檔。
    """
    if img is None:
        print("❌ 錯誤：傳入的圖片為空！")
        return

    # 1. 🌟 預處理：對抗光影
    if ENABLE_CLAHE:
        img = apply_clahe(img)

    h, w = img.shape[:2]
    
    # 2. 定義精華區：削掉上下各 25% (避開最嚴重的弧面反光)
    y_top, y_bottom = int(h * 0.25), int(h * 0.75)
    
    # 3. 計算步進寬度
    slice_step = w / NUM_SLICES
    
    # 4. 建立存檔路徑 (例如: training_data_30)
    save_dir = f"{OUTPUT_BASE_DIR}_{NUM_SLICES}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用時間戳記當作這顆電阻的唯一 ID
    resistor_id = int(time.time() * 1000)
    
    result_img = img.copy()
    display_slices = []

    print(f"\n🚀 啟動預處理與切割管線...")
    print(f"   - 切分數量: {NUM_SLICES}")
    print(f"   - 預處理開啟: {ENABLE_CLAHE}")

    for i in range(NUM_SLICES):
        x_start = int(i * slice_step)
        x_end = int((i + 1) * slice_step)
        
        # A. 執行裁切 (只取精華高度區段)
        color_slice = img[y_top:y_bottom, x_start:x_end]
        if color_slice.size == 0: continue

        # B. 繪製視覺化線條 (自動根據份數調整粗細)
        thickness = 1 if NUM_SLICES > 15 else 2
        cv2.line(result_img, (x_start, 0), (x_start, h), (0, 255, 0), thickness)
        
        # 每隔幾份標註一次數字
        label_interval = max(1, int(NUM_SLICES / 10))
        if i % label_interval == 0:
            cv2.putText(result_img, str(i), (x_start + 2, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # C. 存檔 (使用 zfill 確保排序正確)
        file_name = f"res_{resistor_id}_s{str(i).zfill(2)}.jpg"
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, color_slice)

        # D. 準備顯示畫面 (動態調整單張寬度，總和不超過螢幕)
        disp_w = max(30, int(1200 / NUM_SLICES))
        enlarged = cv2.resize(color_slice, (disp_w, 300), interpolation=cv2.INTER_NEAREST)
        cv2.putText(enlarged, f"{i}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        display_slices.append(enlarged)

    # 5. 顯示最後成果
    cv2.imshow(f"Result - {NUM_SLICES} Slices", result_img)
    if display_slices:
        combined_slices = cv2.hconcat(display_slices)
        cv2.imshow("Extracted Slices (After CLAHE)", combined_slices)

    print(f"✅ 處理完成！切片已存入 '{save_dir}/'")
    print("💡 小提醒：你可以直接去資料夾分類這些圖片來訓練 CNN。")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================
# ⚡ 測試執行入口
# ==========================================
if __name__ == "__main__":
    # ⚠️ 這裡請換成你手邊的電阻照片路徑
    test_image_path = "captured_images/resistor_20260502_190927.jpg" 
    
    if os.path.exists(test_image_path):
        raw_img = cv2.imread(test_image_path)
        
        # --- 模擬情境：假設隊友已經傳來一張水平且去背的電阻圖 ---
        # 這裡我們手動裁切原圖的中間部分來當作範例圖片
        h, w, _ = raw_img.shape
        # 抓取畫面中心區域作為模擬輸入
        sample_resistor = raw_img[int(h*0.4):int(h*0.6), int(w*0.2):int(w*0.8)]
        
        # 執行最終預處理與切割
        slice_resistor_final(sample_resistor)
    else:
        print(f"❌ 找不到測試檔案：{test_image_path}，請檢查檔案路徑。")