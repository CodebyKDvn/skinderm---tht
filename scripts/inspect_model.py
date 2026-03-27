import torch

ckpt = torch.load('segmentation_model.pth', map_location='cpu')
with open('model_keys.txt', 'w') as f:
    if isinstance(ckpt, dict) and 'state_dict' not in ckpt:
        keys = list(ckpt.keys())
        for k in keys:
            f.write(f"{k}: {ckpt[k].shape}\n")
    elif 'state_dict' in ckpt:
        keys = list(ckpt['state_dict'].keys())
        for k in keys:
            f.write(f"{k}: {ckpt['state_dict'][k].shape}\n")
