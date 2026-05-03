import os
import base64
import cv2  # 🌟 新增：需要用 cv2 來讀取與存檔圖片
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# 🌟 引入 YOLO 與剛剛寫好的 OpenCV 預處理模組
from location_scan import get_resistor_bbox
from preprocess import extract_resistor_body

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 建立儲存照片的資料夾
UPLOAD_FOLDER = 'captured_images'
CROPPED_FOLDER = 'cropped_images'  # 🌟 新增：專門放裁切後乾淨電阻的資料夾

for folder in [UPLOAD_FOLDER, CROPPED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json
    image_data = data.get('image') 
    bands_count = data.get('bands') # 🌟 這裡會收到 4 或 5
    
    if image_data:
        header, encoded = image_data.split(",", 1)
        
        # 1. 儲存原始前端傳來的照片
        filename = f"resistor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(encoded))
        
        print(f"✅ 原始照片已存儲至: {filepath}")
        
        # 2. 呼叫 YOLO 進行影像辨識抓座標
        bbox = get_resistor_bbox(filepath)
        
        if bbox:
            # ==========================================
            # 🌟 3. 執行 OpenCV 精準裁切與預處理
            # ==========================================
            # 讀取剛存好的原圖
            original_img = cv2.imread(filepath)
            
            # 丟進我們的神仙演算法
            processed_img, _ = extract_resistor_body(original_img, bbox)
            
            if processed_img is not None:
                # 裁切成功，將乾淨的電阻圖存到專屬資料夾
                cropped_filename = f"clean_{filename}"
                cropped_filepath = os.path.join(CROPPED_FOLDER, cropped_filename)
                cv2.imwrite(cropped_filepath, processed_img)
                print(f"✂️ 裁切成功！乾淨電阻已存儲至: {cropped_filepath}")
                
                return jsonify({
                    "status": "success", 
                    "message": "Image processed successfully.", 
                    "original_image": filename,
                    "cropped_image": cropped_filename, # 回傳裁切後的檔名給前端
                    "bbox": bbox
                })
            else:
                return jsonify({
                    "status": "warning",
                    "message": "YOLO detected object, but OpenCV extraction failed.",
                    "filename": filename
                })
        else:
            return jsonify({
                "status": "warning", 
                "message": "Image saved, but no resistor detected by YOLO.", 
                "filename": filename
            })
    
    return jsonify({"status": "error", "message": "No image data"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)