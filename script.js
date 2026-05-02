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

    // 轉換為 Base64 (後端 API 通常接收此格式或 Blob)
    const imageData = canvas.toDataURL('image/jpeg');

    // 顯示載入狀態
    btnCapture.innerText = "辨識中...";
    btnCapture.disabled = true;

    try {
        const result = await identifyResistor(imageData);
        displayResult(result);
    } catch (error) {
        alert("辨識失敗，請重試");
    } finally {
        btnCapture.innerText = "拍攝辨識";
        btnCapture.disabled = false;
    }
});

// 3. 模擬 API 呼叫 (Mock API)
async function identifyResistor(imageData) {
    // 將 localhost 改成你電腦的區域網路 IP (例如 192.168.x.x)
    // 這樣你的手機才能連到電腦上的 VS Code 後端
    const BACKEND_URL = "https://cascade-antiques-catcher.ngrok-free.dev/upload";

    const response = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData })
    });

    if (!response.ok) throw new Error("網路請求失敗");

    const result = await response.json();
    console.log("後端回傳：", result);

    // 這裡暫時維持 Mock Data 讓 UI 顯示
    return {
        value: "10k Ω ± 5%",
        colors: [
            { name: "棕", hex: "#8B4513" },
            { name: "黑", hex: "#000000" },
            { name: "橘", hex: "#FFA500" },
            { name: "金", hex: "#FFD700" }
        ]
    };
}

// 4. 渲染結果
function displayResult(data) {
    resultArea.classList.remove('hidden');
    valueText.innerText = data.value;
    
    colorTags.innerHTML = ''; // 清空舊標籤
    data.colors.forEach(color => {
        const span = document.createElement('span');
        span.className = 'color-tag';
        span.innerText = color.name;
        span.style.backgroundColor = color.hex;
        // 根據背景深淺調整文字顏色
        span.style.color = (color.name === "黑" || color.name === "棕") ? "#fff" : "#000";
        colorTags.appendChild(span);
    });
}