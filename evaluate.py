from PIL import Image
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

pred = Image.open(
    "outputs/prediction.png"
).convert("RGB")

gt = Image.open(
    "dataset/clean/0.png"
).convert("RGB")

pred = pred.resize((256, 256))
gt = gt.resize((256, 256))

pred = np.array(pred)
gt = np.array(gt)

psnr = peak_signal_noise_ratio(
    gt,
    pred,
    data_range=255
)

ssim = structural_similarity(
    gt,
    pred,
    channel_axis=2,
    data_range=255
)

print(f"PSNR: {psnr:.2f}")
print(f"SSIM: {ssim:.4f}")