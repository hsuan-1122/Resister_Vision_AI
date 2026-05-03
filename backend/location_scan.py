import os
import cv2
from ultralytics import YOLO

# ==========================================
# 🌟 新增：建立儲存 YOLO 畫框結果的資料夾
# ==========================================
YOLO_OUTPUT_DIR = 'yolo_results'
if not os.path.exists(YOLO_OUTPUT_DIR):
    os.makedirs(YOLO_OUTPUT_DIR)

# 1. 將模型載入放在全域 (Global)
# 這樣 Flask 伺服器啟動時只會載入一次，避免每次請求都重新讀取 best.pt 導致嚴重延遲
print("🚀 正在載入 YOLO 模型...")
model = YOLO('backend/best.pt')  # 請確認你的路徑是否正確
print("✅ 模型載入完成！")

def get_resistor_bbox(image_path):
    """
    接收圖片路徑，使用 YOLO 進行預測，並回傳信心最高的 Bounding Box。
    同時將包含偵測框的圖片存檔以供除錯/檢視。
    回傳格式: [xmin, ymin, xmax, ymax] 或 None (如果沒偵測到)
    """
    # 進行預測 (設定 verbose=False 避免終端機被大量 log 洗版)
    results = model.predict(source=image_path, conf=0.5, verbose=False)

    for r in results:
        # 第一道防線：檢查有沒有偵測到東西
        if len(r.boxes) == 0:
            print("⚠️ YOLO: 畫面中沒有偵測到任何超過門檻的目標。")
            return None
            
        # ==========================================
        # 🌟 新增：將 YOLO 畫好框的影像存檔
        # ==========================================
        # r.plot() 會畫上所有預測框，並回傳 BGR 格式的 numpy array
        im_bgr = r.plot()
        
        # 抓取傳入的原圖檔名 (例如: resistor_20260503.jpg)
        original_filename = os.path.basename(image_path)
        
        # 組合成新的存檔路徑 (例如: yolo_results/yolo_resistor_20260503.jpg)
        save_path = os.path.join(YOLO_OUTPUT_DIR, f"yolo_{original_filename}")
        
        # 使用 OpenCV 存檔
        cv2.imwrite(save_path, im_bgr)
        print(f"🖼️ YOLO 畫框結果已存至: {save_path}")
        # ==========================================

        # 找出信心分數最高的框
        best_idx = int(r.boxes.conf.argmax())
        best_box = r.boxes.xyxy[best_idx].cpu().numpy()
        best_conf = float(r.boxes.conf[best_idx])
        
        # 拆解出 xmin, ymin, xmax, ymax (轉成整數，方便後續給 OpenCV 裁切使用)
        xmin, ymin, xmax, ymax = map(int, best_box)
        
        print(f"🎯 YOLO 鎖定目標！ (信心分數: {best_conf:.2f}) -> BBox: [{xmin}, {ymin}, {xmax}, {ymax}]")
        
        # 回傳座標
        return [xmin, ymin, xmax, ymax]
        
    return None