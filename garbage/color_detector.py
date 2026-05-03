import cv2
import numpy as np
import tensorflow as tf

# 1. 在伺服器啟動時，先載入模型 (不要每次預測都重新載入，會很慢)
print("載入模型中...")
model = tf.keras.models.load_model('resistor_color_model.keras')

# 2. 定義你的 13 個類別字典 (順序必須跟訓練時的資料夾 0~12 完全一樣)
CLASS_NAMES = [
    "black", "brown", "red", "orange", "yellow", 
    "green", "blue", "purple", "gray", "white", 
    "gold", "silver", "background"
]

def predict_color(image_path):
    # 3. 讀取並處理圖片
    # 假設前端傳來了一張裁切好的色環照片
    img = cv2.imread(image_path)
    
    # 將圖片強制縮放到模型要求的 64x64 大小
    img_resized = cv2.resize(img, (64, 64))
    
    # OpenCV 讀取的是 BGR，但 Keras 訓練時如果用預設讀取通常是 RGB
    # 保險起見，將 BGR 轉為 RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 模型吃的是「批次(Batch)」陣列，所以要多包一層維度：從 (64, 64, 3) 變成 (1, 64, 64, 3)
    img_array = np.expand_dims(img_rgb, axis=0)
    
    # 4. 讓模型進行預測
    predictions = model.predict(img_array)
    
    # 5. 抓出機率最高的那個類別索引 (0~12)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    result_color = CLASS_NAMES[predicted_index]
    
    return result_color, confidence

# ====== 測試看看 ======
result, conf = predict_color("training_data_30/res_1777723609615_s16.jpg")
print(f"預測結果: {result} (信心水準: {conf:.2f}%)")