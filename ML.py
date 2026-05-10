import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. 讀取 CSV 資料
print("讀取資料中...")
df = pd.read_csv('ML_data_merge.csv')
df = df.dropna()  

# 建議加入這行：檢查各顏色資料量是否平均
print("\n各顏色資料量分佈:")
print(df['label'].value_counts())

# 2. 切分特徵 (X) 與標籤 (y)
X = df[['H', 'S']]
y = df['label']

# 3. 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 建立 Pipeline (這步最關鍵！)
# 將 StandardScaler (特徵縮放) 和 SVC (分類器) 綁定在一起
# 這樣存檔時，縮放比例和模型會一起被存下來，未來預測時才不會出錯
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf'))
])

# 5. 使用 GridSearchCV 自動尋找最佳參數
print("\n尋找最佳參數與模型訓練中 (這可能需要幾秒鐘)...")
# 設定你要讓電腦嘗試的參數組合 (注意：Pipeline 中的參數要加上 'svc__' 前綴)
param_grid = {
    'svc__C': [0.1, 1, 10, 50, 100],
    'svc__gamma': ['scale', 0.01, 0.1, 1]
}

# cv=5 代表使用 5 折交叉驗證，確保模型不是靠運氣好才準確
grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1) 
grid_search.fit(X_train, y_train)

# 6. 驗證模型準確度 (使用找到的最強模型)
best_model = grid_search.best_estimator_

print(f"\n✅ 找到最佳參數: {grid_search.best_params_}")
predictions = best_model.predict(X_test)
print(f"✅ 模型準確率: {accuracy_score(y_test, predictions) * 100:.2f}%\n")

print("各顏色分類報告:")
print(classification_report(y_test, predictions))

# 7. 將訓練好的 Pipeline 存檔
joblib.dump(best_model, 'resistor_color_svm_v5.pkl')
print("✅ 包含縮放器的完整模型已儲存為 resistor_color_svm_v5.pkl")