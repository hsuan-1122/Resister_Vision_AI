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
    "black": { value: 0, multiplier: 1, tolerance: null, hex: "#222222", textColor: "#fff" },
    "brown": { value: 1, multiplier: 10, tolerance: "±1%", hex: "#8B4513", textColor: "#fff" },
    "red": { value: 2, multiplier: 100, tolerance: "±2%", hex: "#D32F2F", textColor: "#fff" },
    "orange": { value: 3, multiplier: 1000, tolerance: null, hex: "#F57C00", textColor: "#fff" },
    "yellow": { value: 4, multiplier: 10000, tolerance: null, hex: "#FBC02D", textColor: "#000" },
    "green": { value: 5, multiplier: 100000, tolerance: "±0.5%", hex: "#388E3C", textColor: "#fff" },
    "blue": { value: 6, multiplier: 1000000, tolerance: "±0.25%", hex: "#1976D2", textColor: "#fff" },
    "purple": { value: 7, multiplier: 10000000, tolerance: "±0.1%", hex: "#7B1FA2", textColor: "#fff" },
    "gray": { value: 8, multiplier: 100000000, tolerance: "±0.05%", hex: "#757575", textColor: "#fff" },
    "white": { value: 9, multiplier: 1000000000, tolerance: null, hex: "#FAFAFA", textColor: "#000" },
    "gold": { value: null, multiplier: 0.1, tolerance: "±5%", hex: "#FFD700", textColor: "#000" },
    "silver": { value: null, multiplier: 0.01, tolerance: "±10%", hex: "#C0C0C0", textColor: "#000" }
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

    try {
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
                name: colorName,
                hex: data ? data.hex : "#CCC",
                textColor: data ? data.textColor : "#000"
            };
        });

        return {
            displayValue: `${formatResistance(resistance)} ${tolerance}`,
            uiColors: uiColors
        };

    } catch (e) {
        console.error("查表計算發生錯誤:", e);
        return { displayValue: "顏色辨識錯誤", uiColors: [] };
    }
}

// ==========================================
// 3. 修改後的模擬 API 呼叫 (模擬後端只回傳陣列)
// ==========================================
async function identifyResistor(imageData, bands) {
    const BACKEND_URL = "https://cascade-antiques-catcher.ngrok-free.dev/upload";

    /*
    // TODO: 正式串接時把這段註解拿掉
    const response = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", "ngrok-skip-browser-warning": "true" },
        body: JSON.stringify({ image: imageData, bands: bands })
    });
    const data = await response.json();
    const detectedColors = data.colors; // 假設後端回傳格式為 { colors: ["棕", "黑", "橘", "金"] }
    */

    // 這裡我們模擬後端 AI 成功回傳了一組顏色字串陣列
    let detectedColors = [];
    if (bands === 4) {
        detectedColors = ["棕", "黑", "橘", "金"]; // 模擬: 10k Ω ± 5%
    } else {
        detectedColors = ["棕", "黑", "黑", "紅", "棕"]; // 模擬: 10k Ω ± 1% (五環)
    }

    // 將後端傳來的陣列丟進我們寫好的計算機
    return calculateResistance(detectedColors);
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