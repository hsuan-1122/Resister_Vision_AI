import cv2
import numpy as np
from scipy.signal import find_peaks, peak_widths

import csv  # 🌟 新增：處理 CSV 檔案
import os   # 🌟 新增：用來檢查檔案是否已存在

import joblib  # 🌟 新增：用來讀取訓練好的 .pkl 模型

# ==========================================
# 🌟 全域設定開關
# ==========================================
# 設為 True：每次辨識都會把 RGBHSV 記錄到 CSV 中 (適合蒐集資料階段)
# 設為 False：純粹進行辨識，不寫入任何檔案 (適合正式上線階段)
ENABLE_DATA_COLLECTION = True

# ==========================================
# 🌟 載入訓練好的 SVM 模型 (放在全域避免重複載入)
# ==========================================
print("🚀 正在載入 SVM 顏色分類模型...")
# 確保檔名與路徑和你儲存的 pkl 檔一致
svm_model = joblib.load('resistor_color_svm.pkl') 
print("✅ SVM 模型載入完成！")

def package_band_data(robust_hsv, band_positions):
    w = len(robust_hsv)
    bands_data = []
    radius = 2

    for pos in band_positions:
        left = max(0, int(pos) - radius)
        right = min(w - 1, int(pos) + radius)
        
        local_window = robust_hsv[left : right + 1]
        local_hsv_mean = np.mean(local_window, axis=0)
        
        h_val, s_val, v_val = local_hsv_mean
        
        hsv_pixel = np.uint8([[[h_val, s_val, v_val]]])
        rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0][0]
        
        band_info = {
            "absolute_x": int(pos),
            "relative_x": round(pos / w, 4),
            "hsv": {"h": round(h_val, 2), "s": round(s_val, 2), "v": round(v_val, 2)},
            "rgb": {"r": int(rgb_pixel[0]), "g": int(rgb_pixel[1]), "b": int(rgb_pixel[2])}
        }
        bands_data.append(band_info)
        
    return bands_data


def detect_resistor_bands_centroid(image_path, num_bands=4):
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤：無法讀取圖片 {image_path}")
        return []

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    
    slice_hsv = img_hsv[int(h * 0.20):int(h * 0.80), :]
    robust_hsv = np.median(slice_hsv, axis=0)

    s_line = robust_hsv[:, 1]
    v_line = robust_hsv[:, 2]

    interest_signal = s_line + (255 - v_line)
    kernel_size = 5
    interest_signal = np.convolve(interest_signal, np.ones(kernel_size)/kernel_size, mode='same')

    min_dist = max(1, int(w * 0.05))
    peaks, properties = find_peaks(interest_signal, distance=min_dist, prominence=10)

    if len(peaks) < num_bands:
        top_peaks = peaks
    else:
        top_indices = np.argsort(properties['prominences'])[-num_bands:]
        top_peaks = peaks[top_indices]

    widths, width_heights, left_ips, right_ips = peak_widths(interest_signal, top_peaks, rel_height=0.5)
    
    refined_positions = []
    for i in range(len(top_peaks)):
        left, right = int(left_ips[i]), int(right_ips[i])
        
        if right > left:
            x_region = np.arange(left, right + 1)
            weights = interest_signal[left:right + 1]
            centroid = np.sum(x_region * weights) / np.sum(weights)
            refined_positions.append(centroid) # 這裡保留 float，交給下一層轉 int
        else:
            refined_positions.append(top_peaks[i])

    band_positions = np.sort(refined_positions)
    return package_band_data(robust_hsv, band_positions)


def get_resistor_color(image_path, num_bands=4):
    band_data = detect_resistor_bands_centroid(image_path, num_bands=num_bands) 
    color_array = []
    data_to_save = [] # 用來暫存準備寫入 CSV 的數據
    
    # 防呆：如果完全沒偵測到任何色環，直接回傳 None
    if not band_data or len(band_data) == 0:
        print("⚠️ 警告：完全沒有偵測到色環特徵！")
        return None

    # ==========================================
    # 🌟 新增：利用間距判斷電阻方向並自動排序
    # ==========================================
    # 至少要有 3 個環才能比較「頭尾間距」
    if len(band_data) >= 3:
        # 計算最左邊的間距 (第1環與第2環的距離)
        first_gap = band_data[1]["absolute_x"] - band_data[0]["absolute_x"]
        # 計算最右邊的間距 (倒數第2環與最後1環的距離)
        last_gap = band_data[-1]["absolute_x"] - band_data[-2]["absolute_x"]

        # 如果左邊間距明顯大於右邊間距，代表電阻被反放了 (誤差環在最左邊)
        # 我們將陣列反轉，確保後續讀取永遠是從「數值環」讀到「誤差環」
        if first_gap > last_gap:
            print("🔄 偵測到電阻反向 (最大間距在左側)，自動反轉讀取順序...")
            band_data.reverse()
        else:
            print("➡️ 電阻方向正確 (最大間距在右側)。")
    else:
        print("⚠️ 警告：偵測到的色環少於 3 個，無法透過間距判斷方向。")

    # 安全地迴圈抓取資料
    for i in range(num_bands):
        # 檢查目前的 index 是否還在偵測到的陣列長度內
        if i < len(band_data):
            # 1. 提取特徵數值 (保留 RGB 供模型預測，但之後不存入 CSV)
            r = band_data[i]["rgb"]["r"]
            g = band_data[i]["rgb"]["g"]
            b = band_data[i]["rgb"]["b"]
            h = band_data[i]["hsv"]["h"]
            s = band_data[i]["hsv"]["s"]
            v = band_data[i]["hsv"]["v"]

            # ==========================================
            # 🌟 使用新模型預測
            # ==========================================
            features = [[r, g, b, h, s, v]]
            predicted_color = svm_model.predict(features)[0]

            # 💡 商業邏輯防呆
            is_metal_pos = (i >= num_bands - 2)
            if not is_metal_pos and predicted_color in ["gold", "silver"]:
                predicted_color = "yellow" if predicted_color == "gold" else "gray"

            color_array.append(predicted_color)
            print(f"🎯 第 {i+1} 環預測結果: {predicted_color} (X座標: {band_data[i]['absolute_x']})")

            # 2. 如果開關打開，只將 HSV 數據暫存起來
            if ENABLE_DATA_COLLECTION:
                # 若發生反轉，存入 CSV 的數據也會是校正後的正確順序 (數值 -> 誤差)
                data_to_save.append([h, s]) 
                
        else:
            print(f"⚠️ 警告：第 {i+1} 環特徵遺失，補上 unknown。")
            color_array.append("unknown")
            
    # ==========================================
    # 🌟 批次寫入 CSV 功能
    # ==========================================
    if ENABLE_DATA_COLLECTION and len(data_to_save) > 0:
        csv_filename = "color_data_log2.csv"
        file_exists = os.path.exists(csv_filename)

        with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            if not file_exists:
                writer.writerow(['H', 'S'])
                
            writer.writerows(data_to_save)
        
    return color_array

