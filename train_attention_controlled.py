import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt

from dataset import CloudDataset
from attention_unet import AttentionUNet


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)


# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cpu")
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 0.0001

SPLIT_PATH = Path("results/data_split.json")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # -----------------------------
    # Load saved split
    # -----------------------------
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Run create_data_split.py first."
        )

    with SPLIT_PATH.open("r", encoding="utf-8") as f:
        split = json.load(f)

    dataset = CloudDataset(
        "dataset/cloudy",
        "dataset/clean"
    )

    if len(dataset) != split["total_images"]:
        raise ValueError(
            "Dataset size differs from the saved split."
        )

    current_files = sorted(
        p.name
        for p in Path("dataset/cloudy").iterdir()
        if p.is_file()
        and p.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
        }
    )

    if current_files != split["filenames"]:
        raise ValueError(
            "Dataset filenames changed since the split was created."
        )

    train_dataset = Subset(dataset, split["train"])
    val_dataset = Subset(dataset, split["validation"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("=" * 60)
    print("CONTROLLED ATTENTION U-NET TRAINING")
    print("=" * 60)
    print(f"Training images   : {len(train_dataset)}")
    print(f"Validation images : {len(val_dataset)}")
    print(f"Test images       : {len(split['test'])}")
    print(f"Device            : {DEVICE}")
    print(f"Epochs            : {EPOCHS}")
    print("=" * 60)

    # -----------------------------
    # Model
    # -----------------------------
    model = AttentionUNet().to(DEVICE)

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_epoch = 0

    train_losses = []
    val_losses = []

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(EPOCHS):

        model.train()
        total_train_loss = 0.0

        for cloudy, clean in train_loader:

            cloudy = cloudy.to(DEVICE)
            clean = clean.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(cloudy)
            loss = criterion(predictions, clean)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = (
            total_train_loss / len(train_loader)
        )

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():

            for cloudy, clean in val_loader:

                cloudy = cloudy.to(DEVICE)
                clean = clean.to(DEVICE)

                predictions = model(cloudy)
                loss = criterion(predictions, clean)

                total_val_loss += loss.item()

        avg_val_loss = (
            total_val_loss / len(val_loader)
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                MODEL_DIR / "attention_controlled_best.pth"
            )

        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.7f}"
        )

    # -----------------------------
    # Save final checkpoint
    # -----------------------------
    torch.save(
        model.state_dict(),
        MODEL_DIR / "attention_controlled_final.pth"
    )

    # -----------------------------
    # Save loss history
    # -----------------------------
    history = {
        "train_loss": train_losses,
        "validation_loss": val_losses,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "seed": SEED,
        "training_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "test_images": len(split["test"])
    }

    with open(
        OUTPUT_DIR / "attention_controlled_history.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(history, f, indent=2)

    # -----------------------------
    # Plot training curves
    # -----------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        train_losses,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        range(1, EPOCHS + 1),
        val_losses,
        marker="s",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Controlled Attention U-Net Training")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "attention_controlled_loss.png",
        dpi=150
    )

    plt.close()

    print("\n" + "=" * 60)
    print("CONTROLLED ATTENTION U-NET TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best epoch          : {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(
        "Best checkpoint    : "
        "models/attention_controlled_best.pth"
    )
    print(
        "Final checkpoint   : "
        "models/attention_controlled_final.pth"
    )
    print(
        "Loss curve         : "
        "outputs/attention_controlled_loss.png"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()