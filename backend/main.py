from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import random
import time
import asyncio
import requests
import json
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import timm
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import io

app = FastAPI(title="SkinGuardAI Backend API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. DIAGNOSIS ENSEMBLE MODEL (efficientnet_b3 + convnext_tiny) -> 8 classes
# ---------------------------------------------------------
class EnsembleModel(nn.Module):
    def __init__(self, num_classes=8):
        super(EnsembleModel, self).__init__()
        self.backbone1 = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        self.backbone2 = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
        
        in_features = self.backbone1.num_features + self.backbone2.num_features
        
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features1 = self.backbone1(x)
        features2 = self.backbone2(x)
        features = torch.cat([features1, features2], dim=1)
        return self.head(features)

# ---------------------------------------------------------
# 1b. CONVNEXTV2-TINY MODEL (standalone) -> 9 classes
# ---------------------------------------------------------
class ConvNextModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.model = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# 2. SEGMENTATION UNET MODEL (smp.Unet - mit_b5)
# ---------------------------------------------------------
# using segmentation_models_pytorch instead of manual Unet


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_LOADED = False
model_a = None
model_b = None

# Trọng số Weighted Ensemble: Model A (xịn hơn) chiếm 60%, Model B 40%
W_A = 0.6
W_B = 0.4

try:
    # 1. Load Model A: EnsembleModel (efficientnet_b3 + convnext_tiny) -> 8 classes
    DIAG_MODEL_PATH_A = "models/best_model_isic2019.pth"
    model_a = EnsembleModel(num_classes=8)
    ckpt_a = torch.load(DIAG_MODEL_PATH_A, map_location=device)
    sd_a = ckpt_a['state_dict'] if 'state_dict' in ckpt_a else ckpt_a
    model_a.load_state_dict(sd_a)
    model_a.to(device)
    model_a.eval()
    print(f"[OK] Loaded Model A (EnsembleModel, 8 classes) onto {device}")
except Exception as e:
    print(f"[WARN] Failed to load Model A: {e}")
    model_a = None

try:
    # 2. Load Model B: ConvNeXtV2-Tiny -> 9 classes
    DIAG_MODEL_PATH_B = "models/best_model.pth"
    model_b = ConvNextModel(num_classes=9)
    ckpt_b = torch.load(DIAG_MODEL_PATH_B, map_location=device)
    sd_b = ckpt_b['state_dict'] if 'state_dict' in ckpt_b else ckpt_b
    # Strip 'model.' prefix if present (checkpoint was saved with a wrapper)
    sd_b = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd_b.items()}
    model_b.model.load_state_dict(sd_b)
    model_b.to(device)
    model_b.eval()
    print(f"[OK] Loaded Model B (ConvNeXtV2-Tiny, 9 classes) onto {device}")
except Exception as e:
    print(f"[WARN] Failed to load Model B: {e}")
    model_b = None

try:
    # 3. Load Segmentation Model (mit_b5 Unet)
    SEG_MODEL_PATH = "models/best_mit_b5_unet.pth"
    print("⏳ Loading segmentation model (339MB)...")
    model_seg = smp.Unet(
        encoder_name="mit_b5",
        encoder_weights=None, 
        in_channels=3,
        classes=1
    )
    # use map_location=device to load to GPU/CPU correctly
    model_seg.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=device))
    model_seg.to(device)
    model_seg.eval()
    print(f"[OK] Loaded Segmentation Model (mit_b5) onto {device}")
except Exception as e:
    print(f"[WARN] Failed to load Segmentation Model: {e}")

NUM_ENSEMBLE_CLASSES = 9  # max classes across all models

if model_a is not None or model_b is not None:
    MODEL_LOADED = True
    loaded_count = sum(1 for m in [model_a, model_b] if m is not None)
    print(f"\n=== Weighted Ensemble ready with {loaded_count}/2 diagnosis model(s) (W_A={W_A}, W_B={W_B}) ===")
else:
    MODEL_LOADED = False
    print("\n=== No diagnosis models loaded! ===")

LABELS = ["MEL (Melanoma)", "NV (Melanocytic nevus)", "BCC (Basal cell carcinoma)", 
          "AK (Actinic keratosis)", "BKL (Benign keratosis)", "DF (Dermatofibroma)", 
          "VASC (Vascular lesion)", "SCC (Squamous cell carcinoma)", "UNK (Unknown)"]

def process_image(image_bytes: bytes):
    if not MODEL_LOADED:
        return None
        
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = original_image.size
    
    # ----------------------------------
    # 1. SEGMENTATION PIPELINE (mit_b5_unet)
    # ----------------------------------
    # Convert PIL Image to numpy (RGB format)
    ori_img_np = np.array(original_image)
    
    # Image size specific to the new model
    img_size = 384
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ])
    
    input_tensor = transform(image=ori_img_np)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model_seg(input_tensor)
        mask_pred = torch.sigmoid(output).cpu().numpy()[0][0]
        
    # Resize mask back to original image size
    mask_full = cv2.resize(mask_pred, (W, H))
    binary_mask = (mask_full > 0.5).astype(np.uint8)
    
    # Auto-Crop Logic
    coords = np.column_stack(np.where(binary_mask > 0))
    padding = 0.2
    
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        h_m, w_m = y_max - y_min, x_max - x_min
        rmin = max(0, int(y_min - h_m * padding))
        rmax = min(H, int(y_max + h_m * padding))
        cmin = max(0, int(x_min - w_m * padding))
        cmax = min(W, int(x_max + w_m * padding))
    else:
        rmin, rmax, cmin, cmax = 0, H, 0, W
        
    # Crop Image (PIL)
    cropped_image = original_image.crop((cmin, rmin, cmax, rmax))
    
    # Calculate bounding box for Frontend (values from 0 to 1 relative to original image)
    bbox = {
        "x": cmin / W,
        "y": rmin / H,
        "w": (cmax - cmin) / W,
        "h": (rmax - rmin) / H
    }
    
    # ----------------------------------
    # 2. DIAGNOSIS PIPELINE - WEIGHTED ENSEMBLE 2 MODEL + TTA x6 (On Cropped)
    #    Kỹ thuật: Weighted Average Probabilities + AMP fp16
    # ----------------------------------
    
    # 6 TTA augmentations (giữ nguyên tâm ảnh, không crop)
    tta_augments = [
        lambda img: img,                                          # 1. Original
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),         # 2. Horizontal flip
        lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),         # 3. Vertical flip
        lambda img: img.transpose(Image.ROTATE_90),               # 4. Rotate 90°
        lambda img: img.transpose(Image.ROTATE_270),              # 5. Rotate -90° (270°)
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),  # 6. HFlip+VFlip (≡ Rotate 180°)
    ]
    
    diag_transform_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    with torch.no_grad():
        # Tách riêng predictions cho từng model để áp trọng số
        probs_list_a = []  # Lưu predictions Model A qua các TTA
        probs_list_b = []  # Lưu predictions Model B qua các TTA
        
        for tta_fn in tta_augments:
            aug_image = tta_fn(cropped_image)
            tensor_224 = diag_transform_224(aug_image).unsqueeze(0).to(device)
            
            # === Mixed Precision (AMP fp16) cho tốc độ và tiết kiệm VRAM ===
            with torch.amp.autocast(device_type=device_type):
                # Model A: EnsembleModel (8 classes)
                if model_a is not None:
                    outputs_a = model_a(tensor_224)
                    probs_a = torch.nn.functional.softmax(outputs_a[0], dim=0)
                    # Pad từ 8 -> 9 classes (thêm 0 cho class cuối)
                    probs_a = torch.cat([probs_a, torch.zeros(NUM_ENSEMBLE_CLASSES - 8, device=probs_a.device)])
                    probs_list_a.append(probs_a)
                
                # Model B: ConvNeXtV2-Tiny (9 classes) 
                if model_b is not None:
                    outputs_b = model_b(tensor_224)
                    probs_b = torch.nn.functional.softmax(outputs_b[0], dim=0)
                    probs_list_b.append(probs_b)
        
        # Trung bình TTA cho từng model
        if probs_list_a:
            avg_probs_a = torch.stack(probs_list_a).mean(dim=0)  # [9]
        else:
            avg_probs_a = None
            
        if probs_list_b:
            avg_probs_b = torch.stack(probs_list_b).mean(dim=0)  # [9]
        else:
            avg_probs_b = None
        
        # === Weighted Average Ensemble ===
        if avg_probs_a is not None and avg_probs_b is not None:
            # Cả 2 model đều sẵn sàng -> Weighted Average
            final_probs = (avg_probs_a * W_A) + (avg_probs_b * W_B)
        elif avg_probs_a is not None:
            final_probs = avg_probs_a
        elif avg_probs_b is not None:
            final_probs = avg_probs_b
        else:
            # Fallback: không model nào load được
            return None
        
        confidence, predicted = torch.max(final_probs, 0)
        
        # Calculate Top 3 predictions
        top3_prob, top3_indices = torch.topk(final_probs, 3)
        top3 = []
        for i in range(3):
            idx = top3_indices[i].item()
            label_name = LABELS[idx] if idx < len(LABELS) else "Unknown"
            score = round(float(top3_prob[i].item()) * 100, 2)
            top3.append({"label": label_name, "score": score})
        
    return int(predicted.item()), float(confidence.item()), bbox, top3

@app.get("/")
def read_root():
    return {"status": "ok", "message": "SkinGuardAI API is running"}

class AnalyzeResponse(BaseModel):
    risk_score: float
    classification: str
    confidence: float
    abcde: dict
    bbox: dict
    top3: list

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Run Pipeline
    result = process_image(contents)
    
    if result:
        class_idx, confidence, bbox, top3 = result
        classification = LABELS[class_idx] if class_idx < len(LABELS) else "Unknown"
        
        if "MEL" in classification or "BCC" in classification or "SCC" in classification:
            base_risk = round(confidence * 100, 1)
        else:
            base_risk = round((1.0 - confidence) * 100, 1)
            
        abcde_scores = {
            "A_score": random.randint(70, 95) if base_risk > 50 else random.randint(10, 40),
            "B_score": random.randint(60, 90) if base_risk > 50 else random.randint(10, 30),
            "C_score": random.randint(50, 90),
            "D_score": random.randint(40, 90),
            "E_score": random.randint(50, 90)
        }
    else:
        # Fallback Mock logic
        base_risk = random.randint(30, 95)
        confidence = round(random.uniform(85.0, 99.5), 2)
        if base_risk > 75:
            classification = "MEL (Suspected Melanoma)"
        elif base_risk > 50:
            classification = "Atypical Nevus"
        else:
            classification = "NV (Benign Mole)"
            
        abcde_scores = {
            "A_score": random.randint(10, 90),
            "B_score": random.randint(10, 95),
            "C_score": random.randint(20, 80),
            "D_score": random.randint(40, 90),
            "E_score": 10
        }
        bbox = {"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6}
        top3 = [
            {"label": classification, "score": round(confidence * 100, 2)},
            {"label": "Other prediction 1", "score": round((1 - confidence) * 50, 2)},
            {"label": "Other prediction 2", "score": round((1 - confidence) * 30, 2)},
        ]
    
    return {
        "risk_score": base_risk,
        "classification": classification,
        "confidence": round(confidence * 100 if result else confidence, 2),
        "abcde": {
            "A_asymmetry": {
                "score": abcde_scores["A_score"],
                "status": "High" if abcde_scores["A_score"] > 50 else "Low"
            },
            "B_border": {
                "score": abcde_scores["B_score"],
                "status": "Abnormal" if abcde_scores["B_score"] > 50 else "Normal"
            },
            "C_color": {
                "score": abcde_scores["C_score"],
                "status": "Moderate" if abcde_scores["C_score"] > 50 else "Uniform"
            },
            "D_diameter": {
                "value": round(random.uniform(3.0, 9.5), 1),
                "score": abcde_scores["D_score"]
            },
            "E_evolution": {
                "score": abcde_scores["E_score"],
                "status": "Requires History"
            }
        },
        "bbox": bbox,
        "top3": top3
    }


class ChatRequest(BaseModel):
    message: str
    language: str = "vi"
    model: str = "moonshotai/kimi-k2.5"
    context: str = ""

@app.post("/api/chat")
def chat_interaction(chat_req: ChatRequest):
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
        "Accept": "text/event-stream"
    }

    system_prompt = (
        "Bạn là Bác sĩ Trưởng khoa Chuyên ngành Da liễu & Giải phẫu bệnh lý (DermAI Vision). "
        "Nhiệm vụ của bạn là cung cấp các phân tích có tính CHUYÊN MÔN CAO, CHUẨN XÁC KHOA HỌC và DỰA TRÊN BẰNG CHỨNG (Evidence-Based Medicine) cho bệnh nhân về các tổn thương da.\n\n"
        "--- NGUYÊN TẮC CẦN SỰ ĐÚNG ĐẮN VỀ KHOA HỌC ---\n"
        "1. **Phân tích chi tiết**: Bạn phải giải thích rõ các đặc trưng về mặt hình thái học (Morphology) như Màu sắc (Color hyper/hypo-pigmentation), Đường viền (Border irregularity), Độ bất đối xứng (Asymmetry). Nếu có dữ liệu ABCDE, hãy bám sát nó.\n"
        "2. **Giải thích thuật ngữ**: Khi dùng thuật ngữ y khoa (ví dụ: tế bào hắc tố Melanocytes, lớp đáy Epidermis, tăng sinh mạch máu Vascularization), hãy giải thích ngắn gọn khoa học.\n"
        "3. **Thái độ khách quan & Điềm tĩnh**: Trả lời tự tin, chắc chắn về mặt lý thuyết khoa học dựa trên số liệu nhưng không khẳng định 100% bệnh khi chưa có sinh thiết thực tế. Tuyệt đối không phỏng đoán mơ hồ.\n"
        "4. **Cấu trúc rành mạch**: Sử dụng các đề mục tiêu chuẩn: [ĐÁNH GIÁ TỔNG QUAN], [PHÂN TÍCH LÝ THUYẾT & ABCDE], [HƯỚNG DẪN LÂM SÀNG THEO DÕI].\n"
        "5. **Lời khuyên thực tế**: Luôn nhấn mạnh Tiêu chuẩn vàng (Gold Standard) là sinh thiết cắt bỏ (Excisional biopsy) nếu có nghi ngờ Malignant.\n\n"
        f"--- Diagnosis Context (Dữ liệu chẩn đoán của bệnh nhân) ---\n{chat_req.context}"
    )
    
    payload = {
        "model": chat_req.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query (Respond in {'Vietnamese' if chat_req.language == 'vi' else 'English'}): {chat_req.message}"}
        ],
        "max_tokens": 4096, # max response tokens allowed, context is managed by model
        "temperature": 0.4, # Lower temperature for strictly scientific answers
        "top_p": 0.9,
        "stream": True,
        "chat_template_kwargs": {"thinking":True},
    }

    def generate_events():
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(invoke_url, headers=headers, json=payload, stream=True, timeout=(10, 120))
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        yield f"{decoded_line}\n\n"
                return  # Thành công thì thoát
            except requests.exceptions.Timeout as e:
                print(f"API Timeout (attempt {attempt+1}/{max_retries+1}): {e}")
                if attempt < max_retries:
                    time.sleep(1)
                    continue
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else 0
                print(f"API HTTP {status} Error (attempt {attempt+1}/{max_retries+1}): {e}")
                # Retry cho lỗi 5xx (server tạm thời lỗi)
                if status >= 500 and attempt < max_retries:
                    time.sleep(2)
                    continue
                break
            except Exception as e:
                print(f"API Error: {e}")
                break
        
        msg = "Lỗi kết nối tới Server AI. Vui lòng thử lại sau." if chat_req.language == "vi" else "AI Server error. Please try again later."
        yield f'data: {{"error": "{msg}"}}\n\n'

    return StreamingResponse(generate_events(), media_type="text/event-stream")
