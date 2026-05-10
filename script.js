const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const btnStart = document.getElementById('btn-start');
const btnCapture = document.getElementById('btn-capture');
const resultArea = document.getElementById('result-area');
const valueText = document.getElementById('value-text');
const colorTags = document.getElementById('color-tags');

// 1. 啟動相機
btnStart.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" }, // 強制使用後置鏡頭
            audio: false
        });
        video.srcObject = stream;
        btnStart.disabled = true;
        btnStart.innerText = "相機已啟動";
        btnCapture.disabled = false;
    } catch (err) {
        console.error("無法開啟相機: ", err);
        alert("請確認是否給予相機權限");
    }
});

// 2. 拍攝並辨識
btnCapture.addEventListener('click', async () => {
    // 拍照存入 canvas
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 轉換為 Base64
    const imageData = canvas.toDataURL('image/jpeg');

    // ==========================================
    // 🌟 新增：讀取目前選擇的是四環還是五環
    // ==========================================
    const selectedBands = document.querySelector('input[name="resistorBands"]:checked').value;

    // 顯示載入狀態
    btnCapture.innerText = "辨識中...";
    btnCapture.disabled = true;

    try {
        // 🌟 修改：將 imageData 與 selectedBands 一起傳給 identifyResistor 函式
        const result = await identifyResistor(imageData, parseInt(selectedBands));
        displayResult(result);
    } catch (error) {
        alert("辨識失敗，請重試");
    } finally {
        btnCapture.innerText = "拍攝辨識";
        btnCapture.disabled = false;
    }
});

// ==========================================
// 🌟 新增：電阻色碼查表字典 (包含數值、倍率、誤差與 UI 顏色)
// ==========================================
const resistorColorDict = {
    "black":  { nameZh: "黑", value: 0, multiplier: 1, tolerance: null, hex: "#222222", textColor: "#fff" },
    "brown":  { nameZh: "棕", value: 1, multiplier: 10, tolerance: "±1%", hex: "#8B4513", textColor: "#fff" },
    "red":    { nameZh: "紅", value: 2, multiplier: 100, tolerance: "±2%", hex: "#D32F2F", textColor: "#fff" },
    "orange": { nameZh: "橘", value: 3, multiplier: 1000, tolerance: null, hex: "#F57C00", textColor: "#fff" },
    "yellow": { nameZh: "黃", value: 4, multiplier: 10000, tolerance: null, hex: "#FBC02D", textColor: "#000" },
    "green":  { nameZh: "綠", value: 5, multiplier: 100000, tolerance: "±0.5%", hex: "#388E3C", textColor: "#fff" },
    "blue":   { nameZh: "藍", value: 6, multiplier: 1000000, tolerance: "±0.25%", hex: "#1976D2", textColor: "#fff" },
    "purple": { nameZh: "紫", value: 7, multiplier: 10000000, tolerance: "±0.1%", hex: "#7B1FA2", textColor: "#fff" },
    "gray":   { nameZh: "灰", value: 8, multiplier: 100000000, tolerance: "±0.05%", hex: "#757575", textColor: "#fff" },
    "white":  { nameZh: "白", value: 9, multiplier: 1000000000, tolerance: null, hex: "#FAFAFA", textColor: "#000" },
    "gold":   { nameZh: "金", value: null, multiplier: 0.1, tolerance: "±5%", hex: "#FFD700", textColor: "#000" },
    "silver": { nameZh: "銀", value: null, multiplier: 0.01, tolerance: "±10%", hex: "#C0C0C0", textColor: "#000" }
};

// ==========================================
// 🌟 新增：將數值美化 (例如 1000 -> 1k, 1000000 -> 1M)
// ==========================================
function formatResistance(ohms) {
    if (ohms >= 1000000) return (ohms / 1000000) + "M Ω";
    if (ohms >= 1000) return (ohms / 1000) + "k Ω";
    // 處理浮點數誤差 (例如 0.1)
    return parseFloat(ohms.toFixed(2)) + " Ω";
}

// ==========================================
// 🌟 新增：查表計算核心邏輯
// ==========================================
function calculateResistance(colorArray) {
    let resistance = 0;
    let tolerance = "";

    // try {
        if (colorArray.length === 4) {
            // 四環：(環1*10 + 環2) * 倍率
            const val1 = resistorColorDict[colorArray[0]].value;
            const val2 = resistorColorDict[colorArray[1]].value;
            const mult = resistorColorDict[colorArray[2]].multiplier;
            tolerance = resistorColorDict[colorArray[3]].tolerance || "±20%";
            
            resistance = (val1 * 10 + val2) * mult;

        } else if (colorArray.length === 5) {
            // 五環：(環1*100 + 環2*10 + 環3) * 倍率
            const val1 = resistorColorDict[colorArray[0]].value;
            const val2 = resistorColorDict[colorArray[1]].value;
            const val3 = resistorColorDict[colorArray[2]].value;
            const mult = resistorColorDict[colorArray[3]].multiplier;
            tolerance = resistorColorDict[colorArray[4]].tolerance || "±20%";
            
            resistance = (val1 * 100 + val2 * 10 + val3) * mult;
        } else {
            return { displayValue: "無法辨識色環數量", uiColors: [] };
        }

        // 轉換陣列為 UI 需要的顏色格式
        const uiColors = colorArray.map(colorName => {
            const data = resistorColorDict[colorName];
            return {
                // 🌟 關鍵修改：如果有找到對應的資料，就用 nameZh (中文)，否則 fallback 原本的英文
                name: data ? data.nameZh : colorName, 
                hex: data ? data.hex : "#CCC",
                textColor: data ? data.textColor : "#000"
            };
        });

        return {
            displayValue: `${formatResistance(resistance)} ${tolerance}`,
            uiColors: uiColors
        };

    // } catch (e) {
    //     console.error("查表計算發生錯誤:", e);
    //     return { displayValue: "顏色辨識錯誤", uiColors: [] };
    // }
}

// 3. 呼叫真實 API
async function identifyResistor(imageData, bands) {
    const BACKEND_URL = "https://cascade-antiques-catcher.ngrok-free.dev/upload";

    try {
        const response = await fetch(BACKEND_URL, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json", 
                "ngrok-skip-browser-warning": "true" 
            },
            // 將影像資料與選擇的環數一起傳送
            body: JSON.stringify({ image: imageData, bands: bands })
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const result = await response.json();
        console.log("後端回傳資料：", result);

        if (result.status === "success" && result.colors) {
            // 🌟 核心：將後端真實傳來的顏色陣列，丟進前端的查表計算機
            return calculateResistance(result.colors);
        } else {
            console.error("後端處理失敗：", result.message);
            return { displayValue: "辨識失敗", uiColors: [] };
        }

    } catch (error) {
        console.error("API 呼叫失敗：", error);
        return { displayValue: "網路連線錯誤", uiColors: [] };
    }
}

// ==========================================
// 4. 渲染結果 (使用算出來的資料)
// ==========================================
function displayResult(resultData) {
    resultArea.classList.remove('hidden');
    
    // 顯示算好的 10k Ω ± 5%
    valueText.innerText = resultData.displayValue;
    
    colorTags.innerHTML = ''; // 清空舊標籤
    
    // 動態產生色塊
    resultData.uiColors.forEach(colorObj => {
        const span = document.createElement('span');
        span.className = 'color-tag';
        span.innerText = colorObj.name;
        span.style.backgroundColor = colorObj.hex;
        span.style.color = colorObj.textColor; // 動態設定白字或黑字，避免看不清楚
        colorTags.appendChild(span);
    });
}