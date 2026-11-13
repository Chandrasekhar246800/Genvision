print("Script Started")


import torch
from attention_unet import AttentionUNet

# Device
device = torch.device("cpu")

# Create model
model = AttentionUNet().to(device)

# Create dummy input
x = torch.randn(1, 3, 256, 256).to(device)

# Forward pass
with torch.no_grad():
    y = model(x)

print("=" * 50)
print("Attention U-Net Test")
print("=" * 50)
print("Input Shape :", x.shape)
print("Output Shape:", y.shape)
print("=" * 50)