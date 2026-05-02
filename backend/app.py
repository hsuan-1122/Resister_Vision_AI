import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 允許跨網域請求（前端網頁連到後端）

# 建立儲存照片的資料夾
UPLOAD_FOLDER = 'captured_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json
    image_data = data.get('image') # 取得 Base64 字串
    
    if image_data:
        # 去除 Base64 前綴 (data:image/jpeg;base64,...)
        header, encoded = image_data.split(",", 1)
        
        # 使用時間戳記命名檔案，方便在 VS Code 觀察
        filename = f"resistor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 解碼並存檔
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(encoded))
        
        print(f"✅ 照片已存儲至: {filepath}")
        return jsonify({"status": "success", "message": "Image saved", "filename": filename})
    
    return jsonify({"status": "error", "message": "No image data"}), 400

if __name__ == '__main__':
    # 注意：如果要在手機上測試，這裡要用 0.0.0.0
    app.run(host='0.0.0.0', port=5000, debug=True)