import cv2
import numpy as np
import os

# 這是經過物理漆色校正、且已轉換為 OpenCV 專用 BGR 格式的數值
COLOR_DICT = {
    "0_black":  (40, 40, 40),      # 黑 (幾乎沒變)
    "1_brown":  (19, 69, 139),     # 棕
    "2_red":    (40, 40, 200),     # 紅
    "3_orange": (30, 110, 235),    # 橘
    "4_yellow": (30, 200, 230),    # 黃 (芥末黃)
    "5_green":  (60, 130, 50),     # 綠 (橄欖深綠)
    "6_blue":   (170, 80, 40),     # 藍 (深海藍)
    "7_purple": (140, 50, 110),    # 紫
    "8_gray":   (140, 140, 140),   # 灰
    "9_white":  (230, 230, 230),   # 白
    "10_gold":  (50, 140, 180),    # 金 (暗土黃)
    "11_silver":(190, 185, 180)    # 銀 (冷淺灰)
}

NUM_IMAGES_PER_COLOR = 400  # 每種顏色要生成幾張？ (你可以隨時調高)
IMG_SIZE = 64               # CNN 預設吃的尺寸
BASE_DIR = "training_data"  # 🌟 這裡對準你截圖裡的母資料夾名稱

print("🚀 開始生成訓練照片並分類放入資料夾...")

for folder_name, bgr_value in COLOR_DICT.items():
    # 組合出完整的資料夾路徑 (例如: training_data/3_orange)
    save_dir = os.path.join(BASE_DIR, folder_name)
    
    # 確保資料夾存在 (雖然你已經建好了，這行算是個保險)
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(NUM_IMAGES_PER_COLOR):
        # A. 建立基礎純色畫布
        img = np.full((IMG_SIZE, IMG_SIZE, 3), bgr_value, dtype=np.uint8)
        
        # B. 加入隨機亮度擾動 (更安全的做法，防止顏色數值爆掉)
        brightness_shift = np.random.randint(-40, 40)
        img = np.clip(img.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)
        
        # C. 加入高斯雜訊 (模擬相機噪點與預處理顆粒感)
        noise = np.random.normal(0, 15, (IMG_SIZE, IMG_SIZE, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # D. 🌟 關鍵：將照片存進對應的資料夾中
        file_path = os.path.join(save_dir, f"synth_{i:04d}.jpg")
        cv2.imwrite(file_path, img)

    print(f"✅ {folder_name} 已生成 {NUM_IMAGES_PER_COLOR} 張照片！")

print("\n🎉 0 到 11 號顏色生成完畢！")
print("⚠️ 提醒：請手動將真實電阻的底漆/無顏色切片放入 '12_background' 資料夾！")