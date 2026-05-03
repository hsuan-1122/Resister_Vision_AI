from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

# 1. 載入模型
model = YOLO('best.pt')

# 2. 指定圖片路徑
test_images = '/content/drive/MyDrive/Gemini Gems/R1.jpg'

# 3. 進行預測
results = model.predict(source=test_images, conf=0.5)

# 4. 提取並計算最高信心目標的座標
for r in results:
    # 第一道防線：檢查到底有沒有偵測到任何東西
    if len(r.boxes) == 0:
        print("⚠️ 畫面中沒有偵測到任何超過門檻的電阻。")
        continue # 跳過這張圖片的後續處理
        
    # 找出信心分數最高的那個框的索引值 (Index)
    # r.boxes.conf 是一組包含所有分數的張量，argmax() 會回傳最大值的位置
    best_idx = int(r.boxes.conf.argmax())
    
    # 透過索引值，精準提取「最高分」的座標與分數
    best_box = r.boxes.xyxy[best_idx].cpu().numpy()
    best_conf = float(r.boxes.conf[best_idx])
    
    # 拆解出 x1, y1, x2, y2
    x1, y1, x2, y2 = best_box
    
    # 組合四個點的座標
    tl = (x1, y1) # 左上
    tr = (x2, y1) # 右上
    br = (x2, y2) # 右下
    bl = (x1, y2) # 左下
    
    im_array = r.plot()  # 繪製含有框框的 BGR numpy array
    im = Image.fromarray(im_array[..., ::-1])  # 轉成 RGB 展示
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.axis('off')
    plt.show()
    
    # 印出結果
    print(f"🎯 鎖定最高信心目標！ (信心分數: {best_conf:.2f})")
    print(f"  左上: ({tl[0]:.2f}, {tl[1]:.2f})")
    print(f"  右上: ({tr[0]:.2f}, {tr[1]:.2f})")
    print(f"  右下: ({br[0]:.2f}, {br[1]:.2f})")
    print(f"  左下: ({bl[0]:.2f}, {bl[1]:.2f})")
    print("-" * 30)