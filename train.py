import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import CloudDataset
from unet import UNet

import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

torch.use_deterministic_algorithms(True)

# Device
device = torch.device("cpu")

# Dataset
dataset = CloudDataset(
    "dataset/cloudy",
    "dataset/clean"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# Model
model = UNet().to(device)

# Loss Function
criterion = nn.L1Loss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# Number of Epochs
epochs = 20

# Store losses
losses = []

print("=" * 50)
print("Training Started...")
print("=" * 50)

for epoch in range(epochs):

    total_loss = 0.0

    for cloudy, clean in loader:

        cloudy = cloudy.to(device)
        clean = clean.to(device)

        # Forward Pass
        output = model(cloudy)

        # Compute Loss
        loss = criterion(output, clean)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    losses.append(avg_loss)

    print(
        f"Epoch [{epoch + 1}/{epochs}] | "
        f"Loss: {avg_loss:.6f}"
    )

# Save Model
torch.save(
    model.state_dict(),
    "models/unet.pth"
)

print("\nModel saved to models/unet.pth")

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)

plt.savefig("outputs/loss_curve.png")
plt.close()

print("Loss curve saved to outputs/loss_curve.png")

print("=" * 50)
print("Training Complete!")
print("=" * 50)