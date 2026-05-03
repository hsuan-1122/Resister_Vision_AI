from ultralytics import YOLO

# 1. 將模型載入放在全域 (Global)
# 這樣 Flask 伺服器啟動時只會載入一次，避免每次請求都重新讀取 best.pt 導致嚴重延遲
print("🚀 正在載入 YOLO 模型...")
model = YOLO('backend/best.pt')
print("✅ 模型載入完成！")

def get_resistor_bbox(image_path):
    """
    接收圖片路徑，使用 YOLO 進行預測，並回傳信心最高的 Bounding Box。
    回傳格式: [xmin, ymin, xmax, ymax] 或 None (如果沒偵測到)
    """
    # 進行預測 (設定 verbose=False 避免終端機被大量 log 洗版)
    results = model.predict(source=image_path, conf=0.5, verbose=False)

    for r in results:
        # 第一道防線：檢查有沒有偵測到東西
        if len(r.boxes) == 0:
            print("⚠️ YOLO: 畫面中沒有偵測到任何超過門檻的目標。")
            return None
            
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