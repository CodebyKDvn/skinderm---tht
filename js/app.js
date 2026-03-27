/* =========================================
   APP.JS - DermAI Workspace Logic
   ========================================= */

document.addEventListener('DOMContentLoaded', () => {

    // --- DOM Elements ---
    
    // Views
    const uploadView = document.getElementById('upload-view');
    const cameraView = document.getElementById('camera-view');
    const previewSection = document.getElementById('previewSection');
    const toggleBtns = document.querySelectorAll('.toggle-btn');
    
    // Inputs
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    // Camera
    const webcam = document.getElementById('webcam');
    const cameraCanvas = document.getElementById('cameraCanvas');
    const btnCapture = document.getElementById('btnCapture');
    const btnSwitchCamera = document.getElementById('btnSwitchCamera');
    
    // Preview & Action
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.querySelector('.image-preview-container');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnRetake = document.getElementById('btnRetake');
    
    // Results
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const placeholderText = document.getElementById('placeholderText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultContent = document.getElementById('resultContent');
    const confidenceCircle = document.getElementById('confidenceCircle');
    const confidenceText = document.getElementById('confidenceText');
    const resultAdvice = document.getElementById('resultAdvice');
    const medicalWarning = document.getElementById('medicalWarning');
    
    // AI Chat Elements (Overlay)
    const btnOpenAI = document.getElementById('btnOpenAI');
    const aiChatOverlay = document.getElementById('aiChatOverlay');
    const closeAiChat = document.getElementById('closeAiChat');
    const chatWelcome = document.getElementById('chatWelcome');
    const chatFlow = document.getElementById('chatFlow');
    const aiMasterInput = document.getElementById('aiMasterInput');
    const btnSendMaster = document.getElementById('btnSendMaster');
    const suggCards = document.querySelectorAll('.sugg-card');
    
    // Custom Model Dropdown Elements
    const activeModelLabel = document.getElementById('activeModelLabel');
    const modelOptions = document.getElementById('modelOptions');
    const aiModelSelect = document.getElementById('aiModelSelect');
    const modelOptionElements = document.querySelectorAll('.model-option');
    
    // Canvas & Explanability
    const resultCanvas = document.getElementById('resultCanvas');
    const toggleHeatmap = document.getElementById('toggleHeatmap');
    const toggleBBox = document.getElementById('toggleBBox');
    
    // History & Compare
    const historyList = document.getElementById('historyList');
    const btnCompare = document.getElementById('btnCompare');
    const compareModal = document.getElementById('compareModal');
    const closeCompareModal = document.getElementById('closeCompareModal');
    
    // State
    let currentStream = null;
    let usingFrontCamera = false;
    let currentImageData = null; // Base64 or Blob URL
    let currentAIResult = null;
    let historyData = JSON.parse(localStorage.getItem('dermai_history')) || [];
    let selectedHistoryIndexes = []; // For comparison

    // --- Initialize ---
    initHistory();
    
    // --- View Toggling ---
    toggleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update buttons
            toggleBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Switch views
            const target = btn.dataset.target;
            if (target === 'upload-view') {
                cameraView.style.display = 'none';
                previewSection.style.display = 'none';
                uploadView.style.display = 'block';
                stopCamera();
            } else if (target === 'camera-view') {
                uploadView.style.display = 'none';
                previewSection.style.display = 'none';
                cameraView.style.display = 'block';
                startCamera();
            }
        });
    });

    // --- File Upload Logic ---
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) return alert('Chỉ hỗ trợ file hình ảnh!');
        
        const reader = new FileReader();
        reader.onload = (e) => {
            showPreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    // --- Camera Logic ---
    async function startCamera() {
        if (currentStream) stopCamera();

        const constraints = {
            video: {
                facingMode: usingFrontCamera ? 'user' : 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };

        try {
            currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            webcam.srcObject = currentStream;
            webcam.style.transform = usingFrontCamera ? 'scaleX(-1)' : 'scaleX(1)';
        } catch (err) {
            console.error(err);
            alert('Không thể truy cập Camera. Xin vui lòng kiểm tra quyền trên trình duyệt.');
        }
    }

    function stopCamera() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
    }

    btnSwitchCamera.addEventListener('click', () => {
        usingFrontCamera = !usingFrontCamera;
        startCamera();
    });

    btnCapture.addEventListener('click', () => {
        if (!currentStream) return;
        cameraCanvas.width = webcam.videoWidth;
        cameraCanvas.height = webcam.videoHeight;
        const ctx = cameraCanvas.getContext('2d');
        
        if (usingFrontCamera) {
            ctx.translate(cameraCanvas.width, 0);
            ctx.scale(-1, 1);
        }
        
        ctx.drawImage(webcam, 0, 0, cameraCanvas.width, cameraCanvas.height);
        const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.9);
        stopCamera();
        
        showPreview(dataUrl);
    });

    // --- Preview Flow ---
    function showPreview(dataUrl) {
        uploadView.style.display = 'none';
        cameraView.style.display = 'none';
        previewSection.style.display = 'flex';
        
        currentImageData = dataUrl;
        imagePreview.src = dataUrl;
        
        // Reset Result Panel
        resultContent.style.display = 'none';
        resultPlaceholder.style.display = 'flex';
        placeholderText.textContent = "Sẵn sàng phân tích. Bấm 'Bắt đầu' để tiếp tục.";
        loadingSpinner.style.display = 'none';
        previewContainer.classList.remove('scanning');
    }

    btnRetake.addEventListener('click', () => {
        previewSection.style.display = 'none';
        
        const activeTarget = document.querySelector('.toggle-btn.active').dataset.target;
        if (activeTarget === 'camera-view') {
            cameraView.style.display = 'block';
            startCamera();
        } else {
            uploadView.style.display = 'block';
        }
    });

    // --- Connect to Backend API ---
    btnAnalyze.addEventListener('click', () => {
        // Start scanning UI
        previewContainer.classList.add('scanning');
        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang phân tích dữ liệu AI...';
        
        resultPlaceholder.style.display = 'flex';
        placeholderText.textContent = "Khởi tạo kết nối đến Cụm máy chủ GPU...";
        loadingSpinner.style.display = 'inline-block';
        
        // Convert base64 data to blob for upload
        fetch(currentImageData)
            .then(res => res.blob())
            .then(blob => {
                const formData = new FormData();
                formData.append("file", blob, "scan.jpg");
                
                placeholderText.textContent = "Đang áp dụng mô hình U-Net & ConvNeXtV2...";
                
                // Call actual backend API
                return fetch("http://127.0.0.1:8000/api/analyze", {
                    method: "POST",
                    body: formData
                });
            })
            .then(response => {
                if (!response.ok) throw new Error("Network response was not ok");
                return response.json();
            })
            .then(data => {
                completeAnalysis(data);
            })
            .catch(error => {
                console.error("Lỗi API Backend:", error);
                placeholderText.textContent = "Kết nối Backend thất bại (" + error.message + "). Thử lại sau.";
                loadingSpinner.style.display = 'none';
                previewContainer.classList.remove('scanning');
                btnAnalyze.disabled = false;
                btnAnalyze.innerHTML = '<i class="fa-solid fa-microscope"></i> Bắt đầu phân tích AI';
            });
    });

    function completeAnalysis(apiData) {
        previewContainer.classList.remove('scanning');
        btnAnalyze.disabled = false;
        btnAnalyze.innerHTML = '<i class="fa-solid fa-microscope"></i> Bắt đầu phân tích AI';
        
        resultPlaceholder.style.display = 'none';
        resultContent.style.display = 'block';
        
        // Map backend classification to UI logic
        let prediction = "benign";
        let hasWarning = false;
        
        if (apiData.classification.includes("MEL") || apiData.classification.includes("BCC") || apiData.classification.includes("SCC") || apiData.classification.includes("Malignant") || apiData.classification.includes("Suspected")) {
            prediction = "danger";
            hasWarning = true;
        } else if (apiData.classification.includes("AK") || apiData.classification.includes("Atypical")) {
            prediction = "warning";
        }

        let conf = apiData.confidence || 0;
        let diagnosisTitle = apiData.classification;
        let adviceStr = "";

        // Reset Stroke colors
        confidenceCircle.classList.remove('stroke-benign', 'stroke-warning', 'stroke-danger');
        resultDiagnosis.classList.remove('text-benign', 'text-warning', 'text-danger');

        if (prediction === "benign") {
            adviceStr = "Không phát hiện khối u ác tính. Tính đồng nhất màu sắc và bờ nốt ruồi bình thường. Tiếp tục theo dõi mỗi 6 tháng.";
            confidenceCircle.classList.add('stroke-benign');
            resultDiagnosis.classList.add('text-benign');
        } else if (prediction === "warning") {
            adviceStr = "Bờ không đều hoặc màu sắc có sự khác biệt nhẹ. Nên theo dõi sát sao sự thay đổi trong 1-2 tháng tới.";
            confidenceCircle.classList.add('stroke-warning');
            resultDiagnosis.classList.add('text-warning');
        } else {
            adviceStr = "Đặc trưng biểu bì dị sản, màu sắc bất đồng nhất nhiều. Nguy cơ đặc trưng ung thư da hắc tố cao. Đề nghị sinh thiết lập tức.";
            confidenceCircle.classList.add('stroke-danger');
            resultDiagnosis.classList.add('text-danger');
        }

        // Apply UI
        resultDiagnosis.textContent = diagnosisTitle;
        resultAdvice.textContent = adviceStr;
        medicalWarning.style.display = hasWarning ? 'flex' : 'none';
        
        // Show AI Button & Set Context
        if (btnOpenAI) btnOpenAI.style.display = 'block';
        window.currentDiagnosisContext = `Dự đoán: ${diagnosisTitle}. Phân loại: ${prediction} (Độ tin cậy: ${conf}%). Lời khuyên hệ thống: ${adviceStr}`;
        if (apiData.abcde) {
            window.currentDiagnosisContext += ` Kết quả ABCDE: Asymmetry: ${apiData.abcde.A_asymmetry.status}, Border: ${apiData.abcde.B_border.status}, Color: ${apiData.abcde.C_color.status}, Diameter: ${apiData.abcde.D_diameter.value}mm, Evolution: ${apiData.abcde.E_evolution.status}.`;
        }
        
        // Render Top 3 List
        const top3ListEl = document.getElementById('top3List');
        if (top3ListEl && apiData.top3) {
            top3ListEl.innerHTML = apiData.top3.map((item, index) => {
                let badgeClass = '';
                if (item.label.includes('MEL') || item.label.includes('BCC') || item.label.includes('SCC')) badgeClass = 'text-danger';
                else if (item.label.includes('AK') || item.label.includes('Atypical')) badgeClass = 'text-warning';
                
                return `
                <div class="top3-item">
                    <span class="top3-label ${badgeClass}">${index + 1}. ${item.label}</span>
                    <span class="top3-score">${item.score}%</span>
                </div>
                `;
            }).join('');
        }
        
        // Animate Circle
        confidenceCircle.setAttribute('stroke-dasharray', `${conf}, 100`);
        confidenceText.textContent = `${conf}%`;

        // Save result object globally (Map bounding box scaled coordinates)
        currentAIResult = {
            id: Date.now().toString(),
            date: new Date().toLocaleDateString('vi-VN', { hour: '2-digit', minute: '2-digit' }),
            image: currentImageData,
            prediction: prediction,
            confidence: conf,
            title: diagnosisTitle,
            rectX: apiData.bbox ? apiData.bbox.x : 0.2, // Coordinates from U-Net
            rectY: apiData.bbox ? apiData.bbox.y : 0.2,
            rectW: apiData.bbox ? apiData.bbox.w : 0.4,
            rectH: apiData.bbox ? apiData.bbox.h : 0.4
        };

        // Render Canvas overlay
        renderExplainabilityCanvas();

        // Save History
        saveHistory(currentAIResult);
    }

    // --- EXPLAINABILITY CANVAS (HEATMAP & BBOX) ---
    function renderExplainabilityCanvas() {
        const ctx = resultCanvas.getContext('2d');
        const imgObj = new Image();
        imgObj.onload = () => {
            // Setup canvas sizes
            const size = Math.min(imgObj.width, imgObj.height);
            resultCanvas.width = 500; // max reasonable internal res
            resultCanvas.height = 500;
            
            // Draw image scaled
            const scale = Math.min(resultCanvas.width / imgObj.width, resultCanvas.height / imgObj.height);
            const cw = imgObj.width * scale;
            const ch = imgObj.height * scale;
            const cx = (resultCanvas.width - cw) / 2;
            const cy = (resultCanvas.height - ch) / 2;

            function drawAll() {
                ctx.clearRect(0,0, resultCanvas.width, resultCanvas.height);
                ctx.drawImage(imgObj, cx, cy, cw, ch);

                const res = currentAIResult;
                const bx = cx + (res.rectX * cw);
                const by = cy + (res.rectY * ch);
                const bw = res.rectW * cw;
                const bh = res.rectH * ch;

                if (toggleHeatmap.checked) {
                    // Create mock radial heatmap gradient
                    const grd = ctx.createRadialGradient(
                        bx + bw/2, by + bh/2, 10,
                        bx + bw/2, by + bh/2, bw * 1.5
                    );
                    
                    // Heatmap color map
                    if (res.prediction === 'danger') {
                        grd.addColorStop(0, "rgba(255, 0, 0, 0.6)");
                        grd.addColorStop(0.5, "rgba(255, 255, 0, 0.4)");
                        grd.addColorStop(1, "rgba(0, 0, 255, 0)");
                    } else if (res.prediction === 'warning') {
                        grd.addColorStop(0, "rgba(255, 165, 0, 0.6)");
                        grd.addColorStop(0.5, "rgba(0, 255, 0, 0.3)");
                        grd.addColorStop(1, "rgba(0, 0, 255, 0)");
                    } else {
                        grd.addColorStop(0, "rgba(0, 255, 0, 0.5)");
                        grd.addColorStop(0.7, "rgba(0, 100, 255, 0.2)");
                        grd.addColorStop(1, "rgba(0, 0, 255, 0)");
                    }
                    
                    ctx.save();
                    ctx.globalCompositeOperation = "screen";
                    ctx.fillStyle = grd;
                    ctx.fillRect(cx, cy, cw, ch);
                    ctx.restore();
                }

                if (toggleBBox.checked) {
                    ctx.strokeStyle = res.prediction === 'danger' ? '#ff5f56' : (res.prediction === 'warning' ? '#ffbd2e' : '#27c93f');
                    ctx.lineWidth = 3;
                    ctx.strokeRect(bx, by, bw, bh);
                    // Label
                    ctx.fillStyle = ctx.strokeStyle;
                    ctx.font = "14px Inter";
                    const txt = `${res.prediction.toUpperCase()} ${res.confidence}%`;
                    const m = ctx.measureText(txt);
                    ctx.fillRect(bx, by - 20, m.width + 10, 20);
                    ctx.fillStyle = "#fff";
                    ctx.fillText(txt, bx + 5, by - 5);
                }
            }

            drawAll();
            
            toggleHeatmap.onchange = drawAll;
            toggleBBox.onchange = drawAll;
        };
        imgObj.src = currentAIResult.image;
    }

    // --- HISTORY MANAGER ---
    function saveHistory(resultObj) {
        historyData.unshift(resultObj);
        if (historyData.length > 20) historyData.pop(); // Keep max 20
        localStorage.setItem('dermai_history', JSON.stringify(historyData));
        initHistory();
    }

    function initHistory() {
        historyList.innerHTML = '';
        if (historyData.length === 0) {
            historyList.innerHTML = '<p style="color:var(--text-secondary); text-align:center; padding-top:2rem;">Chưa có lịch sử, hãy bắt đầu phân tích đầu tiên.</p>';
            return;
        }

        historyData.forEach((item, index) => {
            const div = document.createElement('div');
            div.className = `history-item ${selectedHistoryIndexes.includes(index) ? 'selected' : ''}`;
            
            let badgeClass = 'badge-benign';
            let badgeText = 'Benign';
            if (item.prediction === 'danger') { badgeClass = 'badge-danger'; badgeText = 'Malignant'; }
            if (item.prediction === 'warning') { badgeClass = 'badge-warning'; badgeText = 'Atypical'; }

            div.innerHTML = `
                <img src="${item.image}" class="history-thumb">
                <div class="history-info">
                    <h4>${item.date}</h4>
                    <span class="history-badge ${badgeClass}">${badgeText} - ${item.confidence}%</span>
                </div>
            `;
            
            div.addEventListener('click', () => {
                const idxInArr = selectedHistoryIndexes.indexOf(index);
                if (idxInArr > -1) {
                    // Deselect
                    selectedHistoryIndexes.splice(idxInArr, 1);
                    div.classList.remove('selected');
                } else {
                    // Select
                    if(selectedHistoryIndexes.length >= 2) {
                        // Max 2 selected, pop first
                        const firstIdx = selectedHistoryIndexes.shift();
                        historyList.children[firstIdx].classList.remove('selected');
                    }
                    selectedHistoryIndexes.push(index);
                    div.classList.add('selected');
                }
                btnCompare.disabled = selectedHistoryIndexes.length !== 2;
            });
            
            historyList.appendChild(div);
        });
    }

    // --- COMPARE MODAL ---
    btnCompare.addEventListener('click', () => {
        if (selectedHistoryIndexes.length !== 2) return;
        const i1 = historyData[selectedHistoryIndexes[0]];
        const i2 = historyData[selectedHistoryIndexes[1]];

        document.getElementById('compareDate1').textContent = i1.date;
        document.getElementById('compareImg1').src = i1.image;
        document.getElementById('compareRes1').textContent = `${i1.title} (${i1.confidence}%)`;
        document.getElementById('compareRes1').className = i1.prediction === 'danger' ? 'text-danger' : (i1.prediction === 'warning' ? 'text-warning' : 'text-benign');

        document.getElementById('compareDate2').textContent = i2.date;
        document.getElementById('compareImg2').src = i2.image;
        document.getElementById('compareRes2').textContent = `${i2.title} (${i2.confidence}%)`;
        document.getElementById('compareRes2').className = i2.prediction === 'danger' ? 'text-danger' : (i2.prediction === 'warning' ? 'text-warning' : 'text-benign');

        compareModal.classList.add('show');
    });

    closeCompareModal.addEventListener('click', () => {
        compareModal.classList.remove('show');
    });

    window.addEventListener('click', (e) => {
        if (e.target === compareModal) {
            compareModal.classList.remove('show');
        }
    });

    // --- AI DOCTOR CHAT LOGIC ---
    if (btnOpenAI) {
        btnOpenAI.addEventListener('click', () => {
            aiChatOverlay.style.display = 'flex';
            setTimeout(() => {
                aiMasterInput.focus();
            }, 300);
        });
    }

    if (closeAiChat) {
        closeAiChat.addEventListener('click', () => {
            aiChatOverlay.style.display = 'none';
        });
    }

    suggCards.forEach(card => {
        card.addEventListener('click', () => {
            const cmd = card.dataset.cmd;
            aiMasterInput.value = cmd;
            sendMasterMessage();
        });
    });

    // --- Custom Model Dropdown Logic ---
    if (activeModelLabel && modelOptions) {
        activeModelLabel.addEventListener('click', (e) => {
            e.stopPropagation();
            modelOptions.style.display = modelOptions.style.display === 'none' ? 'flex' : 'none';
        });

        window.addEventListener('click', (e) => {
            if (!e.target.closest('.custom-model-dropdown') && modelOptions.style.display === 'flex') {
                modelOptions.style.display = 'none';
            }
        });

        modelOptionElements.forEach(option => {
            option.addEventListener('click', (e) => {
                e.stopPropagation();
                // Update active states
                modelOptionElements.forEach(o => o.classList.remove('active'));
                option.classList.add('active');
                
                // Update hidden input
                aiModelSelect.value = option.dataset.value;
                
                // Update Label HTML combining icon and title
                const iconHtml = option.querySelector('i').outerHTML;
                const name = option.querySelector('strong').textContent;
                activeModelLabel.innerHTML = `${iconHtml} <span>${name}</span> <i class="fa-solid fa-chevron-down" style="font-size: 0.7rem; color: var(--titanium-dark); margin-left: 5px;"></i>`;
                
                // Close dropdown
                modelOptions.style.display = 'none';
            });
        });
    }

    async function sendMasterMessage() {
        const text = aiMasterInput.value.trim();
        if (!text) return;

        // Transition from Welcome to Flow
        if (chatWelcome.style.display !== 'none') {
            chatWelcome.style.display = 'none';
            chatFlow.style.display = 'flex';
        }

        // Add User Message
        const userDiv = document.createElement('div');
        userDiv.className = 'flow-msg flow-user';
        userDiv.textContent = text;
        chatFlow.appendChild(userDiv);
        
        aiMasterInput.value = '';
        btnSendMaster.disabled = true;
        chatFlow.scrollTop = chatFlow.scrollHeight;

        // Add AI Placeholder
        const aiDiv = document.createElement('div');
        aiDiv.className = 'flow-msg flow-ai';
        aiDiv.innerHTML = '<i class="fa-solid fa-spinner fa-spin" style="color:var(--medical-blue-base)"></i> AI bác sĩ đang suy nghĩ...';
        chatFlow.appendChild(aiDiv);
        chatFlow.scrollTop = chatFlow.scrollHeight;

        try {
            const context = window.currentDiagnosisContext || "Chưa có kết quả chẩn đoán.";
            const model = aiModelSelect.value;
            
            const response = await fetch("http://127.0.0.1:8000/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: text,
                    language: "vi",
                    model: model,
                    context: context
                })
            });

            if (!response.ok) throw new Error("API Error");

            aiDiv.innerHTML = ''; // clear loading
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let fullAiResponse = ""; // Lưu trữ toàn bộ text
            
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split("\n");
                
                for (let line of lines) {
                    if (line.startsWith("data: ") && !line.includes("[DONE]")) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.error) {
                                aiDiv.textContent = data.error;
                            } else if (data.choices && data.choices[0].delta && data.choices[0].delta.content) {
                                fullAiResponse += data.choices[0].delta.content;
                                
                                // 1. Render Markdown trước
                                if (window.marked) {
                                    aiDiv.innerHTML = marked.parse(fullAiResponse);
                                } else {
                                    aiDiv.innerHTML = fullAiResponse.replace(/\n/g, '<br>');
                                }
                                
                                // 2. Render LaTeX math nếu có KaTeX
                                if (window.renderMathInElement) {
                                    renderMathInElement(aiDiv, {
                                        delimiters: [
                                            {left: "$$", right: "$$", display: true},
                                            {left: "$", right: "$", display: false},
                                            {left: "\\(", right: "\\)", display: false},
                                            {left: "\\[", right: "\\]", display: true}
                                        ],
                                        throwOnError: false
                                    });
                                }
                                
                                chatFlow.scrollTop = chatFlow.scrollHeight;
                            }
                        } catch (e) {
                            // ignore incomplete json chunk parsing errors
                        }
                    }
                }
            }
        } catch (error) {
            console.error("Chat Error:", error);
            aiDiv.textContent = "Xin lỗi, đã xảy ra lỗi khi kết nối tới Hệ thống AI. Vui lòng thử lại.";
        } finally {
            btnSendMaster.disabled = false;
            aiMasterInput.focus();
        }
    }

    if (btnSendMaster) btnSendMaster.addEventListener('click', sendMasterMessage);
    if (aiMasterInput) aiMasterInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMasterMessage();
    });

});
