import torch, cv2, os
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 1. CẤU HÌNH (EM CHỈ CẦN ĐỔI ĐƯỜNG DẪN ẢNH)
# ==========================================
CFG = {
    'model_path': 'best_mit_b5_unet.pth',
    'encoder': 'mit_b5',
    'img_size': 384,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'test_image': 'test_image.jpg' # Can change to whatever image path
}

# ==========================================
# 2. HÀM TỰ ĐỘNG CẮT (AUTO-CROP)
# ==========================================
def auto_crop(image, mask, padding=0.2):
    """Cắt vùng nốt ruồi dựa trên mask với lề 20%"""
    coords = np.column_stack(np.where(mask > 0.5))
    if coords.size == 0: return image
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    h_m, w_m = y_max - y_min, x_max - x_min
    y_min = max(0, int(y_min - h_m * padding))
    y_max = min(image.shape[0], int(y_max + h_m * padding))
    x_min = max(0, int(x_min - w_m * padding))
    x_max = min(image.shape[1], int(x_max + w_m * padding))
    
    return image[y_min:y_max, x_min:x_max]

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==========================================
def run_inference(img_path):
    # --- Bước A: Load Model "Siêu nặng" ---
    print("⏳ Đang nạp bộ não 339MB...")
    model = smp.Unet(
        encoder_name=CFG['encoder'],
        encoder_weights=None, # Không cần nạp imagenet vì mình đã có weights xịn
        in_channels=3,
        classes=1
    ).to(CFG['device'])
    
    model.load_state_dict(torch.load(CFG['model_path'], map_location=CFG['device']))
    model.eval()

    # --- Bước B: Xử lý ảnh đầu vào ---
    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = ori_img.shape[:2]

    transform = A.Compose([
        A.Resize(CFG['img_size'], CFG['img_size']),
        A.Normalize(),
        ToTensorV2()
    ])
    
    input_tensor = transform(image=ori_img)['image'].unsqueeze(0).to(CFG['device'])

    # --- Bước C: Predict ---
    with torch.no_grad():
        output = model(input_tensor)
        mask_pred = torch.sigmoid(output).cpu().numpy()[0][0]
    
    # Resize mask về kích thước gốc của ảnh
    mask_full = cv2.resize(mask_pred, (w_orig, h_orig))
    binary_mask = (mask_full > 0.5).astype(np.uint8)

    # --- Bước D: Auto-Crop ---
    cropped_result = auto_crop(ori_img, binary_mask)

    # --- Bước E: Khoe thành quả ---
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1); plt.imshow(ori_img); plt.title("1. Ảnh gốc")
    plt.subplot(1, 3, 2); plt.imshow(binary_mask, cmap='jet', alpha=0.5); plt.imshow(ori_img, alpha=0.5); plt.title("2. Mask AI (Dice 0.91)")
    plt.subplot(1, 3, 3); plt.imshow(cropped_result); plt.title("3. Vùng nốt ruồi đã cắt")
    plt.show()
    
    return cropped_result

if __name__ == '__main__':
    if os.path.exists(CFG['test_image']):
        result = run_inference(CFG['test_image'])
        print("✅ Đã hoàn thành 'phi vụ' cắt nốt ruồi!")
    else:
        print("⚠️ Bro ơi, kiểm tra lại đường dẫn ảnh test nhé!")
