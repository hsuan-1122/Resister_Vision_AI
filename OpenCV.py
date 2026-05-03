import cv2
import numpy as np

# ==========================================
# 全域設定
# ==========================================
USE_CLAHE = True  # 是否啟用 CLAHE 影像增強

def extract_resistor_body(image, bbox):
    """
    優化版：激進去背與金屬腳排除邏輯。
    """
    xmin, ymin, xmax, ymax = bbox
    h_img, w_img = image.shape[:2]
    xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(w_img, xmax), min(h_img, ymax)
    
    roi = image[ymin:ymax, xmin:xmax]
    if roi.size == 0: return None, None
        
    # 1. 多維度特徵提取 (結合灰階邊緣與飽和度)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1] # 飽和度通道：色環與本體通常比金屬腳更鮮豔
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 結合邊緣與飽和度資訊：讓有顏色、有線條的地方更明顯
    combined_map = cv2.addWeighted(edges, 0.7, s_channel, 0.3, 0)
    
    # 形態學強化：讓電阻本體變成一個紮實的長方塊
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(combined_map, cv2.MORPH_CLOSE, kernel)
    
    # 2. 投影與邊界計算
    # 提高門檻值 (從 0.15 提高到 0.25)，過濾掉微弱的金屬腳特徵
    col_sums = np.sum(mask, axis=0)
    col_thresh = np.max(col_sums) * 0.25 
    body_cols = np.where(col_sums > col_thresh)[0]
    
    row_sums = np.sum(mask, axis=1)
    row_thresh = np.max(row_sums) * 0.25
    body_rows = np.where(row_sums > row_thresh)[0]
    
    if len(body_cols) > 0 and len(body_rows) > 0:
        b_xmin, b_xmax = body_cols[0], body_cols[-1]
        b_ymin, b_ymax = body_rows[0], body_rows[-1]
        
        # --- 激進修正：左右內縮 5%，排除末端過渡區域與殘留電線 ---
        width = b_xmax - b_xmin
        offset_x = int(width * 0.05) 
        b_xmin += offset_x
        b_xmax -= offset_x
        
        # 上下稍微內縮 12%，避開弧面反光背景
        height = b_ymax - b_ymin
        offset_y = int(height * 0.12)
        b_ymin += offset_y
        b_ymax -= offset_y
    else:
        return None, None

    # 3. 執行裁切
    body_roi = roi[b_ymin:b_ymax, b_xmin:b_xmax]
    if body_roi.size == 0: return None, None
    
    original_body = body_roi.copy()

    # 4. CLAHE 增強
    if USE_CLAHE:
        lab = cv2.cvtColor(body_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        body_roi = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    return body_roi, original_body

# ==========================================
# 測試與展示區塊
# ==========================================
if __name__ == "__main__":
    # ⚠️ 請換成你實際的圖片路徑
    image_path = "captured_images/resistor_20260502_190830.jpg"
    test_img = cv2.imread(image_path)
    
    if test_img is None:
        print(f"❌ 錯誤：找不到圖片，請檢查路徑是否正確：{image_path}")
    else:
        h, w = test_img.shape[:2]
        
        # 【修正】放寬模擬 YOLO BBox 的範圍！
        # 讓 X 軸取更廣的 15%~85%，Y軸取 35%~65%，確保一開始的綠框能包住整個電阻
        xmin = int(w * 0.15)
        ymin = int(h * 0.35)
        xmax = int(w * 0.85)
        ymax = int(h * 0.65)
        
        mock_bbox = [xmin, ymin, xmax, ymax]
        
        processed_img, original_img = extract_resistor_body(test_img, mock_bbox)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        img_with_box = test_img.copy()
        cv2.rectangle(img_with_box, (mock_bbox[0], mock_bbox[1]), (mock_bbox[2], mock_bbox[3]), (0, 255, 0), 2)
        plt.title("Original with YOLO BBox")
        plt.imshow(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        if original_img is not None:
            # 抓取實際裁切出來的長寬顯示在標題
            oh, ow = original_img.shape[:2]
            plt.title(f"Body Extracted (Before CLAHE)\n{ow} x {oh}")
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        if processed_img is not None:
            ph, pw = processed_img.shape[:2]
            title_suffix = " (CLAHE Enabled)" if USE_CLAHE else " (CLAHE Disabled)"
            plt.title(f"Final Output{title_suffix}\n{pw} x {ph}")
            plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()