from PIL import Image
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# Load Attention U-Net Prediction
pred = Image.open(
    "outputs/attention_prediction.png"
).convert("RGB")

# Load Ground Truth Image
gt = Image.open(
    "dataset/clean/0.png"
).convert("RGB")

# Resize Images
pred = pred.resize((256, 256))
gt = gt.resize((256, 256))

pred = np.array(pred)
gt = np.array(gt)

# Calculate PSNR
psnr = peak_signal_noise_ratio(
    gt,
    pred,
    data_range=255
)

# Calculate SSIM
ssim = structural_similarity(
    gt,
    pred,
    channel_axis=2,
    data_range=255
)

print("=" * 50)
print("Attention U-Net Evaluation")
print("=" * 50)
print(f"PSNR : {psnr:.2f}")
print(f"SSIM : {ssim:.4f}")
print("=" * 50)