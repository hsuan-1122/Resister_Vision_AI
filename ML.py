import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. 讀取 CSV 資料 (Pandas 是處理表格資料的神器)
print("讀取資料中...")
df = pd.read_csv('ML_data2.csv')
df = df.dropna()  # 確保沒有空值干擾模型訓練

# 2. 切分特徵 (X) 與標籤 (y)
X = df[['R', 'G', 'B', 'H', 'S', 'V']]
y = df['label']

# 3. 切分訓練集與測試集 (80% 給模型學習，20% 當作模擬考來驗證)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 建立與訓練模型 (使用 SVM，處理這種非線性分類效果極佳)
print("模型訓練中...")
# kernel='rbf' 讓它可以處理複雜的邊界，C=10 是針對容錯率的微調參數
model = SVC(kernel='rbf', C=10, gamma='scale') 
model.fit(X_train, y_train)

# 5. 驗證模型準確度
predictions = model.predict(X_test)
print(f"\n模型準確率: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
print("各顏色分類報告:")
print(classification_report(y_test, predictions))

# 6. 將訓練好的大腦「存檔」 (極度重要！)
joblib.dump(model, 'resistor_color_svm.pkl')
print("模型已儲存為 resistor_color_svm.pkl")