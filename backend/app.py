import os
import base64
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# 引入你的各種處理模組
from location_scan import get_resistor_bbox
from preprocess import extract_resistor_body
# 🌟 新增：引入你剛寫好的顏色辨識函式 (假設存在 color_detector.py 中)
from color_scan import get_resistor_color

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'captured_images'
CROPPED_FOLDER = 'cropped_images'

for folder in [UPLOAD_FOLDER, CROPPED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json
    image_data = data.get('image')
    # 🌟 新增：接收前端傳來的環數 (4 或 5)
    bands_count = data.get('bands', 4) # 如果前端沒傳，預設為 4
    
    if image_data:
        header, encoded = image_data.split(",", 1)
        
        filename = f"resistor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(encoded))
        
        print(f"✅ 原始照片已存儲至: {filepath}")
        print(f"   設定辨識環數: {bands_count}")
        
        bbox = get_resistor_bbox(filepath)
        
        if bbox:
            original_img = cv2.imread(filepath)
            processed_img, _ = extract_resistor_body(original_img, bbox)
            
            if processed_img is not None:
                cropped_filename = f"clean_{filename}"
                cropped_filepath = os.path.join(CROPPED_FOLDER, cropped_filename)
                cv2.imwrite(cropped_filepath, processed_img)
                print(f"✂️ 裁切成功！乾淨電阻已存儲至: {cropped_filepath}")
                
                # ==========================================
                # 🌟 新增：呼叫顏色辨識函式
                # ==========================================
                # 將乾淨的電阻圖路徑與前端指定的環數傳入
                detected_colors = get_resistor_color(cropped_filepath, num_bands=bands_count)
                
                # 如果辨識函式執行成功，會回傳顏色陣列 (例如 ['brown', 'black', 'orange', 'gold'])
                if detected_colors:
                    return jsonify({
                        "status": "success", 
                        "message": "Resistor identified successfully.", 
                        "original_image": filename,
                        "cropped_image": cropped_filename,
                        "colors": detected_colors # 🌟 將陣列打包回傳
                    })
                else:
                     return jsonify({
                        "status": "error",
                        "message": "Color extraction failed on the cropped image."
                    })
            else:
                return jsonify({
                    "status": "warning",
                    "message": "YOLO detected object, but OpenCV extraction failed."
                })
        else:
            return jsonify({
                "status": "warning", 
                "message": "No resistor detected by YOLO."
            })
    
    return jsonify({"status": "error", "message": "No image data"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)