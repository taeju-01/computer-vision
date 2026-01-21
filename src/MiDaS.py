import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

device = "cuda" if torch.cuda.is_available() else "cpu"

# MiDaS 계열 모델 (가볍고 잘 되는 편)
model_id = "Intel/dpt-hybrid-midas"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
model.eval()

# 이미지 로드
img = Image.open("test_image.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth  # (1, H, W)

# 원본 크기로 업샘플
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=img.size[::-1],  # (H, W)
    mode="bicubic",
    align_corners=False,
).squeeze()

depth = prediction.cpu().numpy()

# 0~1 정규화
depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

# ===== 시각화 =====
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(img); plt.axis("off"); plt.title("Input")

plt.subplot(1,2,2)
plt.imshow(depth_norm, cmap="inferno")
plt.axis("off")
plt.title("MiDaS depth (normalized)")
plt.show()

# ===== 🔥 컬러 depth 저장 (수정된 부분) =====
depth_uint8 = (depth_norm * 255).astype(np.uint8)

# 컬러맵 적용 (inferno와 유사)
depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

# BGR → RGB 변환 후 저장
depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
Image.fromarray(depth_color_rgb).save("midas_depth_color.png")

print("Saved: midas_depth_color.png")
