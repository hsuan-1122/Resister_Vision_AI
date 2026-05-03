import cv2
import numpy as np
from scipy.signal import find_peaks, peak_widths

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



def classify_band_color(band_info, band_index, total_bands):
    """
    根據 HSV 數值將特徵點分類為 12 種電阻顏色之一。
    band_index: 目前是第幾個色環 (0 起始)
    total_bands: 總共有幾個色環 (用來輔助判斷金/銀)
    """
    hsv = band_info["hsv"]
    h, s, v = hsv["h"], hsv["s"], hsv["v"]
    
    # 判斷是否為最後一個色環 (誤差環)，金銀通常只出現在這裡
    is_last_band = (band_index == total_bands - 1)
    is_multiplier_band = (band_index == total_bands - 2)

    # 1. 先抓出「無色彩」的顏色：黑、白、灰、銀
    # V (亮度) 極低 -> 黑色
    if v < 45:
        return "black"
    
    # S (飽和度) 很低代表沒有顏色，只剩灰階的明暗
    if s < 50:
        if v > 200:
            return "white"
        elif 100 < v <= 200:
            # 灰跟銀在視覺上極度相似，通常用位置來猜測
            return "silver" if (is_last_band or is_multiplier_band) else "gray"
        else:
            return "gray"

    # 2. 判斷「有色彩」的顏色 (根據 H 色相)
    # 注意：假設你的 OpenCV H 範圍是 0~180 (如果是 0~360，請把下列的 H 判斷乘以 2)
    
    if (0 <= h < 10) or (170 <= h <= 180):
        # 紅色與棕色的 Hue 是一樣的，差別在於棕色比較暗 (V 較低)
        if v < 130:
            return "brown"
        else:
            return "red"
            
    elif 10 <= h < 22:
        # 橘色與棕色的邊界，暗的橘色也是棕色
        if v < 150:
            return "brown"
        else:
            return "orange"
            
    elif 22 <= h < 35:
        # 黃色與金色的 Hue 一樣。
        # 金色通常飽和度 (S) 較低一點，且經常出現在末端。
        if s < 180 and (is_last_band or is_multiplier_band):
            return "gold"
        else:
            return "yellow"
            
    elif 35 <= h < 85:
        return "green"
        
    elif 85 <= h < 130:
        return "blue"
        
    elif 130 <= h < 170:
        return "purple"

    # 如果掉出範圍 (非常罕見)，預設回傳未知或取近似
    return "unknown"

def get_resistor_color(image_path, num_bands=4):
    band_data = detect_resistor_bands_centroid(image_path, num_bands=4)
    color_array = []
    for i in range(num_bands):
        color_array.append(classify_band_color(band_data[i], i, num_bands))
        print(color_array[i])
    return color_array