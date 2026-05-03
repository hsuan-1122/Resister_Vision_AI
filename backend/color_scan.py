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
ENABLE_DATA_COLLECTION = False

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


# def get_resistor_color(image_path, num_bands=4):
#     band_data = detect_resistor_bands_centroid(image_path, num_bands=4)
#     color_array = []
    
#     # 🌟 防呆：如果完全沒偵測到任何色環，直接回傳 None 讓 app.py 報錯
#     if not band_data or len(band_data) == 0:
#         print("⚠️ 警告：完全沒有偵測到色環特徵！")
#         return None

#     # 🌟 修改：安全地迴圈抓取資料
#     for i in range(num_bands):
#         # 檢查目前的 index 是否還在偵測到的陣列長度內
#         if i < len(band_data):
#             color = classifier.classify_band_color(band_data[i], i, num_bands)
#             color_array.append(color)
#             print(color)
#         else:
#             # 如果偵測到的環不夠（例如預期4環只找到3環），剩下的直接補 "unknown"
#             print(f"⚠️ 警告：第 {i+1} 環特徵遺失，補上 unknown。")
#             color_array.append("unknown")
            
#     return color_array

def get_resistor_color(image_path, num_bands=4):
    band_data = detect_resistor_bands_centroid(image_path, num_bands=num_bands) 
    color_array = []
    data_to_save = [] # 用來暫存準備寫入 CSV 的數據
    
    # 防呆：如果完全沒偵測到任何色環，直接回傳 None
    if not band_data or len(band_data) == 0:
        print("⚠️ 警告：完全沒有偵測到色環特徵！")
        return None

    # 安全地迴圈抓取資料
    for i in range(num_bands):
        # 檢查目前的 index 是否還在偵測到的陣列長度內
        if i < len(band_data):
            # 1. 提取 6 個特徵數值
            r = band_data[i]["rgb"]["r"]
            g = band_data[i]["rgb"]["g"]
            b = band_data[i]["rgb"]["b"]
            h = band_data[i]["hsv"]["h"]
            s = band_data[i]["hsv"]["s"]
            v = band_data[i]["hsv"]["v"]

            # ==========================================
            # 🌟 使用新模型取代舊的距離計算
            # ==========================================
            # 把特徵打包成 2D 陣列餵給模型預測
            features = [[r, g, b, h, s, v]]
            predicted_color = svm_model.predict(features)[0]

            # 💡 商業邏輯防呆：金銀不該出現在前段環數
            is_metal_pos = (i >= num_bands - 2)
            if not is_metal_pos and predicted_color in ["gold", "silver"]:
                predicted_color = "yellow" if predicted_color == "gold" else "gray"

            color_array.append(predicted_color)
            print(f"🎯 第 {i+1} 環預測結果: {predicted_color}")

            # 2. 如果開關打開，將這組數據暫存起來
            if ENABLE_DATA_COLLECTION:
                data_to_save.append([r, g, b, h, s, v])
                
        else:
            # 如果偵測到的環不夠，剩下的直接補 "unknown"
            print(f"⚠️ 警告：第 {i+1} 環特徵遺失，補上 unknown。")
            color_array.append("unknown")
            
    # ==========================================
    # 🌟 批次寫入 CSV 功能 (受全域開關控制)
    # ==========================================
    if ENABLE_DATA_COLLECTION and len(data_to_save) > 0:
        csv_filename = "color_data_log.csv"
        file_exists = os.path.exists(csv_filename)

        # 使用 'a' (append) 模式開啟檔案
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 如果是新建立的檔案，先寫入第一行的欄位名稱
            if not file_exists:
                writer.writerow(['R', 'G', 'B', 'H', 'S', 'V'])
                
            # 一次把剛才暫存的所有顏色數據寫入，效能更好
            writer.writerows(data_to_save)
        
    return color_array


# class ResistorColorClassifier:
#     def __init__(self):
#         # 1. 建立「標準色環字典」(RGB)
#         # 這些是完美白光下的基準點，你可以根據實際環境微調
#         self.standard_colors = {
#             "black":  (0,  0,  0), #
#             "brown":  (25, 0,  0), #
#             "red":    (73, 0,  18), #
#             "orange": (90, 30, 0), #
#             "yellow": (100, 110, 20),
#             "green":  (0,  40, 0), #
#             "blue":   (35,  65,  100),
#             "purple": (120, 60,  140),
#             "gray":   (115, 115, 115),
#             "white":  (200, 200, 200),
#             "gold":   (90, 75, 40),
#             "silver": (160, 160, 170)
#         }
        
#         # 預先將所有標準色轉換為 LAB 空間，節省未來運算時間
#         self.lab_references = {}
#         for name, rgb in self.standard_colors.items():
#             self.lab_references[name] = self._rgb_to_lab(rgb)
#     def _rgb_to_lab(self, rgb_tuple):
#         """將單一 RGB tuple 轉換為 LAB 空間"""
#         pixel_img = np.uint8([[rgb_tuple]])
#         lab_pixel = cv2.cvtColor(pixel_img, cv2.COLOR_RGB2LAB)
#         return lab_pixel[0][0].astype(np.float32)

#     def classify_band_color(self, band_info, band_index, total_bands):
#         """
#         根據 CIELAB 色彩距離與位置權重，分類電阻顏色。
#         """
#         # 直接拿你上一步打包好的 RGB 數值
#         r = band_info["rgb"]["r"]
#         g = band_info["rgb"]["g"]
#         b = band_info["rgb"]["b"]
        
#         # 將目標顏色轉為 LAB
#         target_lab = self._rgb_to_lab((r, g, b))
        
#         # 位置邏輯：判斷是否為乘數環 (倒數第二) 或誤差環 (倒數第一)
#         is_last_band = (band_index == total_bands - 1)
#         is_multiplier_band = (band_index == total_bands - 2)
#         can_be_metal = is_last_band or is_multiplier_band

#         min_distance = float('inf')
#         best_match = "unknown"
        
#         # 遍歷標準色，尋找最短距離 (最相似的顏色)
#         for color_name, ref_lab in self.lab_references.items():
            
#             # 【核心過濾機制】：如果不是末端色環，直接剝奪金、銀的參賽資格！
#             # 這樣系統就絕對不會把第一環的黃色誤判為金色。
#             if not can_be_metal and color_name in ["gold", "silver"]:
#                 continue
                
#             # 計算歐幾里得距離 (Delta E)
#             distance = np.linalg.norm(target_lab - ref_lab)
            
#             if distance < min_distance:
#                 min_distance = distance
#                 best_match = color_name
                
#         return best_match

# classifier = ResistorColorClassifier()