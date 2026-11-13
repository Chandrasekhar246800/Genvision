import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from dataset import CloudDataset
from attention_unet import AttentionUNet

import random
import numpy as np

# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

torch.use_deterministic_algorithms(True)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Dataset
# -----------------------------
dataset = CloudDataset(
    "dataset/cloudy",
    "dataset/clean"
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")

# -----------------------------
# Model
# -----------------------------
model = AttentionUNet().to(device)

# -----------------------------
# Loss Function
# -----------------------------
criterion = nn.L1Loss()

# -----------------------------
# Optimizer
# -----------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# -----------------------------
# Learning Rate Scheduler
# -----------------------------
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# -----------------------------
# Epochs
# -----------------------------
epochs = 20

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_epoch = 0

print("=" * 60)
print("Attention U-Net Training Started...")
print("=" * 60)

for epoch in range(epochs):

    # =========================
    # Training
    # =========================
    model.train()

    total_train_loss = 0.0

    for cloudy, clean in train_loader:

        cloudy = cloudy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        output = model(cloudy)

        loss = criterion(output, clean)

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    train_losses.append(avg_train_loss)

    # =========================
    # Validation
    # =========================
    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for cloudy, clean in val_loader:

            cloudy = cloudy.to(device)
            clean = clean.to(device)

            output = model(cloudy)

            loss = criterion(output, clean)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    val_losses.append(avg_val_loss)

    # -------------------------
    # Save Best Model
    # -------------------------
    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            "models/best_attention_unet.pth"
        )

    # -------------------------
    # Scheduler Step
    # -------------------------
    scheduler.step(avg_val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1:02d}/{epochs}] | "
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Val Loss: {avg_val_loss:.6f} | "
        f"LR: {current_lr:.6f}"
    )

# -----------------------------
# Save Final Model
# -----------------------------
torch.save(
    model.state_dict(),
    "models/attention_unet.pth"
)

print("\nFinal model saved to models/attention_unet.pth")
print("Best model saved to models/best_attention_unet.pth")

print(f"\nBest Epoch : {best_epoch}")
print(f"Best Validation Loss : {best_val_loss:.6f}")

# -----------------------------
# Plot Loss Curves
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(
    train_losses,
    marker="o",
    label="Train Loss"
)

plt.plot(
    val_losses,
    marker="s",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Attention U-Net Training")

plt.legend()

plt.grid(True)

plt.savefig(
    "outputs/attention_train_val_loss.png"
)

plt.close()

print("Loss curve saved to outputs/attention_train_val_loss.png")

print("=" * 60)
print("Training Complete!")
print("=" * 60)