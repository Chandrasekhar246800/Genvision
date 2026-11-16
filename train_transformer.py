"""
GenVision Phase 4.2: Controlled CS-TransUNet Training Pipeline
=============================================================

Reproducible, controlled, and deterministic training pipeline for the
Cloud Severity-Aware Hybrid Transformer (CS-TransUNet) model on satellite imagery.

Key Specifications:
- Model: CS-TransUNet (7,128,906 parameters)
- Loss: Strict L1 Reconstruction Loss (||X_hat - Y||_1)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Learning Rate Schedule: ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)
- Gradient Accumulation: 2 steps (effective batch size: 4)
- Gradient Norm Clipping: max_norm = 1.0
- Device: CPU (deterministic mode)
- Fixed Seed: 42 (strict 350 train / 75 val / 75 held-out test split)
- Checkpoints:
    models/transformer_controlled_best.pth
    models/transformer_controlled_latest.pth
- Telemetry:
    results/phase4/training/training_history.csv
    results/phase4/training/training_log.txt
    results/phase4/training/loss_curve.png
    results/phase4/training/learning_rate_curve.png
    results/phase4/training/training_summary.json
"""

import argparse
import csv
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import CloudDataset
from models.transformer.cs_transunet import CSTransUNet


# -----------------------------------------------------------------------------
# Default Hyperparameters & Settings
# -----------------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM_STEPS = 2
DEFAULT_EPOCHS = 25
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_NORM = 1.0
DEFAULT_LR_FACTOR = 0.5
DEFAULT_LR_PATIENCE = 3
DEFAULT_MIN_LR = 1e-6

DEFAULT_SPLIT_PATH = PROJECT_ROOT / "results" / "data_split.json"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4" / "training"


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds across all libraries and configure deterministic operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)


class TrainingLogger:
    """Handles formatted console and file logging."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize or clear log file
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"GENVISION TRAINING SESSION - {datetime.datetime.now().isoformat()}\n")
            f.write(f"{'=' * 80}\n")

    def log(self, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")


def verify_dataset_split(split_path: Path) -> Dict:
    """
    Load and verify data split integrity against dataset directories.
    Strictly verifies disjoint 350/75/75 train/val/test partitions.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Data split file not found at {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    cloudy_dir = PROJECT_ROOT / "dataset" / "cloudy"
    clean_dir = PROJECT_ROOT / "dataset" / "clean"

    if not cloudy_dir.is_dir() or not clean_dir.is_dir():
        raise FileNotFoundError(f"Dataset directories missing: {cloudy_dir} or {clean_dir}")

    cloudy_files = sorted(
        p.name for p in cloudy_dir.iterdir() if p.is_file() and p.suffix.lower() in CloudDataset.SUPPORTED_EXTENSIONS
    )
    clean_files = sorted(
        p.name for p in clean_dir.iterdir() if p.is_file() and p.suffix.lower() in CloudDataset.SUPPORTED_EXTENSIONS
    )

    if cloudy_files != clean_files:
        raise ValueError("Filename mismatch between cloudy and clean directories.")

    if cloudy_files != split["filenames"]:
        raise ValueError("Current dataset filenames differ from stored data_split.json.")

    train_indices = set(split["train"])
    val_indices = set(split["validation"])
    test_indices = set(split["test"])

    if len(train_indices & val_indices) > 0 or len(train_indices & test_indices) > 0 or len(val_indices & test_indices) > 0:
        raise ValueError("Data split leakage detected: train, validation, or test sets overlap!")

    if len(train_indices) != 350 or len(val_indices) != 75 or len(test_indices) != 75:
        raise ValueError(
            f"Expected 350 train, 75 val, 75 test. Got: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
        )

    return split


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val_loss: float,
    best_epoch: int,
    train_history: List[Dict],
    seed: int = DEFAULT_SEED,
) -> None:
    """Save full training state dictionary for recovery and reproduction (weights_only=True safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np_state = np.random.get_state()
    safe_np_state = (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4])

    state = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "train_history": train_history,
        "seed": seed,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": safe_np_state,
        "python_rng_state": random.getstate(),
        "model_architecture": "CS-TransUNet",
        "parameters_count": sum(p.numel() for p in model.parameters()),
        "saved_at": datetime.datetime.now().isoformat(),
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
    logger: Optional[TrainingLogger] = None,
) -> Tuple[int, float, int, List[Dict]]:
    """Load checkpoint and restore training states (compatible with weights_only=True)."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    best_epoch = checkpoint.get("best_epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    train_history = checkpoint.get("train_history", [])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if "numpy_rng_state" in checkpoint:
        s = checkpoint["numpy_rng_state"]
        np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

    msg = f"Restored checkpoint from {path} (Epoch {epoch}, Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f})"
    if logger:
        logger.log(msg)
    else:
        print(msg)

    return epoch, best_val_loss, best_epoch, train_history


def plot_curves(history: List[Dict], output_dir: Path) -> None:
    """Generate and save loss and learning rate curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    lrs = [h["learning_rate"] for h in history]

    # 1. Loss curve
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, "b-o", label="Training Loss (L1)", linewidth=1.8, markersize=5)
    plt.plot(epochs, val_losses, "r-s", label="Validation Loss (L1)", linewidth=1.8, markersize=5)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("L1 Loss", fontsize=11)
    plt.title("CS-TransUNet Controlled Training & Validation Loss", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close()

    # 2. Learning rate curve
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, lrs, "g-^", label="Learning Rate (AdamW)", linewidth=1.8, markersize=5)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Learning Rate", fontsize=11)
    plt.yscale("log")
    plt.title("CS-TransUNet Learning Rate Schedule (ReduceLROnPlateau)", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate_curve.png", dpi=180)
    plt.close()


def run_training(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_norm: float = DEFAULT_MAX_NORM,
    seed: int = DEFAULT_SEED,
    resume_path: Optional[Path] = None,
    smoke_test: bool = False,
    smoke_train_batches: int = 4,
    smoke_val_batches: int = 2,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> Dict:
    """
    Main training execution function.
    Supports standard controlled training and smoke-test verification.
    """
    set_seed(seed)
    device = torch.device("cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ("smoke_test_log.txt" if smoke_test else "training_log.txt")
    logger = TrainingLogger(log_file)

    logger.log("=" * 80)
    logger.log("GENVISION CS-TRANSUNET CONTROLLED TRAINING PIPELINE")
    logger.log("=" * 80)
    logger.log(f"Mode: {'SMOKE TEST' if smoke_test else 'FULL CONTROLLED TRAINING'}")
    logger.log(f"Device: {device} | PyTorch: {torch.__version__} | Deterministic: True")
    logger.log(f"Seed: {seed} | Epochs: {epochs} | Batch Size: {batch_size} | Grad Accum Steps: {grad_accum_steps}")
    logger.log(f"Effective Batch Size: {batch_size * grad_accum_steps}")
    logger.log(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    logger.log(f"Gradient Clipping: max_norm={max_norm}")
    logger.log(f"Loss Formulation: Strict L1 Reconstruction (||X_hat - Y||_1)")
    logger.log("Note: CSEB severity prediction branch active via AdaLN; no severity cross-entropy loss applied.")

    # 1. Dataset & Split Verification
    split = verify_dataset_split(DEFAULT_SPLIT_PATH)
    logger.log(f"Dataset split verified: 350 train, 75 val, 75 held-out test images.")
    logger.log("Isolation check: Held-out test set will NOT be loaded or evaluated.")

    full_dataset = CloudDataset(PROJECT_ROOT / "dataset" / "cloudy", PROJECT_ROOT / "dataset" / "clean")

    train_subset = Subset(full_dataset, split["train"])
    val_subset = Subset(full_dataset, split["validation"])

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,  # Single-process avoids Windows multiprocessing overhead
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 2. Model Initialization
    t0_model = time.perf_counter()
    model = CSTransUNet(
        in_channels=3,
        out_channels=3,
        stage_channels=(48, 96, 192, 384),
        transformer_depths=(2, 2),
        num_heads=(6, 12),
        severity_embedding_dim=192,
        num_severity_classes=3,
        ffn_expansion_factor=2.0,
    ).to(device)
    model_init_time = (time.perf_counter() - t0_model) * 1000

    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model initialized in {model_init_time:.2f}ms. Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # 3. Loss & Optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=DEFAULT_LR_FACTOR,
        patience=DEFAULT_LR_PATIENCE,
        min_lr=DEFAULT_MIN_LR,
    )

    best_checkpoint_path = model_dir / ("transformer_smoke_best.pth" if smoke_test else "transformer_controlled_best.pth")
    latest_checkpoint_path = model_dir / ("transformer_smoke_latest.pth" if smoke_test else "transformer_controlled_latest.pth")

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    history: List[Dict] = []

    # 4. Resume if specified
    if resume_path and resume_path.exists():
        loaded_epoch, loaded_best_val, loaded_best_epoch, loaded_history = load_checkpoint(
            resume_path, model, optimizer, scheduler, device, logger
        )
        start_epoch = loaded_epoch + 1
        best_val_loss = loaded_best_val
        best_epoch = loaded_best_epoch
        history = loaded_history
        logger.log(
            f"Resuming training from Epoch {start_epoch} of {epochs} "
            f"(Restored Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f})."
        )

    csv_path = output_dir / ("smoke_test_history.csv" if smoke_test else "training_history.csv")
    csv_headers = [
        "epoch",
        "train_loss",
        "val_loss",
        "learning_rate",
        "epoch_duration_sec",
        "elapsed_sec",
        "is_best",
        "prob_thin",
        "prob_moderate",
        "prob_thick",
    ]

    # Initialize CSV if starting fresh
    if start_epoch == 1 or not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    training_start_time = time.perf_counter()
    fwd_bwd_times: List[float] = []
    opt_step_times: List[float] = []
    val_step_times: List[float] = []

    # 5. Epoch Loop
    logger.log("-" * 80)
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.perf_counter()

        # -----------------------------
        # Training Phase
        # -----------------------------
        model.train()
        total_train_loss = 0.0
        train_batches_run = 0
        train_prob_accumulator = torch.zeros(3)

        optimizer.zero_grad()

        total_train_batches = smoke_train_batches if smoke_test else len(train_loader)

        for batch_idx, (cloudy, clean) in enumerate(train_loader):
            if smoke_test and batch_idx >= smoke_train_batches:
                break

            # Exact dynamic group size for accumulation window (handles odd remainder batches)
            start_of_group = (batch_idx // grad_accum_steps) * grad_accum_steps
            end_of_group = min(start_of_group + grad_accum_steps, total_train_batches)
            current_group_size = end_of_group - start_of_group

            t_fb_start = time.perf_counter()

            cloudy = cloudy.to(device)
            clean = clean.to(device)

            outputs = model(cloudy, return_dict=True)
            reconstructed = outputs["reconstructed"]
            sev_probs = outputs["severity_probabilities"].detach().cpu()

            # L1 reconstruction loss strictly, normalized by the current group's batch count
            loss = criterion(reconstructed, clean)
            scaled_loss = loss / current_group_size
            scaled_loss.backward()

            t_fb_end = time.perf_counter()
            fwd_bwd_times.append(t_fb_end - t_fb_start)

            total_train_loss += loss.item()
            train_prob_accumulator += sev_probs.mean(dim=0)
            train_batches_run += 1

            # Flush gradients: step when window ends or at the final batch of the epoch
            is_step_batch = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == total_train_batches)
            if is_step_batch:
                t_opt_start = time.perf_counter()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
                optimizer.zero_grad()
                t_opt_end = time.perf_counter()
                opt_step_times.append(t_opt_end - t_opt_start)

        avg_train_loss = total_train_loss / max(1, train_batches_run)
        mean_train_probs = (train_prob_accumulator / max(1, train_batches_run)).tolist()

        # -----------------------------
        # Validation Phase
        # -----------------------------
        model.eval()
        total_val_loss = 0.0
        val_batches_run = 0
        val_prob_accumulator = torch.zeros(3)

        with torch.no_grad():
            for val_idx, (cloudy, clean) in enumerate(val_loader):
                if smoke_test and val_idx >= smoke_val_batches:
                    break

                t_val_start = time.perf_counter()

                cloudy = cloudy.to(device)
                clean = clean.to(device)

                outputs = model(cloudy, return_dict=True)
                reconstructed = outputs["reconstructed"]
                sev_probs = outputs["severity_probabilities"].detach().cpu()

                val_loss = criterion(reconstructed, clean)

                t_val_end = time.perf_counter()
                val_step_times.append(t_val_end - t_val_start)

                total_val_loss += val_loss.item()
                val_prob_accumulator += sev_probs.mean(dim=0)
                val_batches_run += 1

        avg_val_loss = total_val_loss / max(1, val_batches_run)
        mean_val_probs = (val_prob_accumulator / max(1, val_batches_run)).tolist()

        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.perf_counter() - epoch_start_time
        total_elapsed = time.perf_counter() - training_start_time

        # Checkpoint selection evaluation
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": current_lr,
            "epoch_duration_sec": epoch_duration,
            "elapsed_sec": total_elapsed,
            "is_best": is_best,
            "prob_thin": mean_val_probs[0],
            "prob_moderate": mean_val_probs[1],
            "prob_thick": mean_val_probs[2],
        }
        history.append(record)

        # Save best checkpoint
        if is_best:
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                best_epoch,
                history,
                seed,
            )

        # Always save latest for recovery
        save_checkpoint(
            latest_checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            best_epoch,
            history,
            seed,
        )

        # Write CSV record
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train_loss:.6f}",
                f"{avg_val_loss:.6f}",
                f"{current_lr:.8f}",
                f"{epoch_duration:.2f}",
                f"{total_elapsed:.2f}",
                "YES" if is_best else "NO",
                f"{mean_val_probs[0]:.4f}",
                f"{mean_val_probs[1]:.4f}",
                f"{mean_val_probs[2]:.4f}",
            ])

        best_tag = " [NEW BEST]" if is_best else ""
        lr_change_tag = f" (LR adjusted: {old_lr:.2e} -> {current_lr:.2e})" if current_lr != old_lr else ""
        logger.log(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_duration:.1f}s{best_tag}{lr_change_tag}"
        )
        logger.log(
            f"       CSEB Predicted Distribution (Val): "
            f"Thin={mean_val_probs[0]*100:.1f}%, Mod={mean_val_probs[1]*100:.1f}%, Thick={mean_val_probs[2]*100:.1f}%"
        )

    # 6. Finalization & Plotting
    total_training_time = time.perf_counter() - training_start_time
    plot_curves(history, output_dir)

    avg_fwd_bwd_ms = (sum(fwd_bwd_times) / len(fwd_bwd_times) * 1000) if fwd_bwd_times else 0.0
    avg_opt_step_ms = (sum(opt_step_times) / len(opt_step_times) * 1000) if opt_step_times else 0.0
    avg_val_step_ms = (sum(val_step_times) / len(val_step_times) * 1000) if val_step_times else 0.0

    # Projected runtime calculations for full training set (350 train / 75 val):
    # 175 batches @ bs=2 -> 87 full groups of 2 + 1 odd group of 1 = 88 optimizer steps.
    # 38 validation batches @ bs=2.
    projected_epoch_train_s = (175 * (avg_fwd_bwd_ms / 1000.0)) + (88 * (avg_opt_step_ms / 1000.0))
    projected_epoch_val_s = 38 * (avg_val_step_ms / 1000.0)
    projected_epoch_duration_s = projected_epoch_train_s + projected_epoch_val_s
    projected_25_epochs_min = (projected_epoch_duration_s * 25) / 60.0

    # Composite average step duration per batch
    avg_step_s = (avg_fwd_bwd_ms + (avg_opt_step_ms / max(1, grad_accum_steps))) / 1000.0

    summary = {
        "status": "COMPLETED",
        "mode": "SMOKE_TEST" if smoke_test else "FULL_TRAINING",
        "model_architecture": "CS-TransUNet",
        "parameters_count": total_params,
        "seed": seed,
        "total_epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "total_training_duration_seconds": total_training_time,
        "average_step_duration_seconds": avg_step_s,
        "avg_fwd_bwd_duration_ms": avg_fwd_bwd_ms,
        "avg_optimizer_step_duration_ms": avg_opt_step_ms,
        "avg_validation_step_duration_ms": avg_val_step_ms,
        "projected_epoch_duration_seconds": projected_epoch_duration_s,
        "projected_25_epochs_duration_minutes": projected_25_epochs_min,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "learning_rate_initial": learning_rate,
        "learning_rate_final": history[-1]["learning_rate"] if history else learning_rate,
        "best_checkpoint": str(best_checkpoint_path.resolve()),
        "latest_checkpoint": str(latest_checkpoint_path.resolve()),
        "csv_history": str(csv_path.resolve()),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    summary_file = output_dir / ("smoke_test_summary.json" if smoke_test else "training_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.log("=" * 80)
    logger.log(f"TRAINING COMPLETE ({summary['mode']})")
    logger.log(f"Total Duration: {total_training_time:.2f}s")
    logger.log(f"Measured Forward/Backward: {avg_fwd_bwd_ms:.1f}ms | Optimizer Step: {avg_opt_step_ms:.1f}ms | Val Step: {avg_val_step_ms:.1f}ms")
    logger.log(f"Projected Single Epoch (Full 350/75 Dataset): {projected_epoch_duration_s:.1f}s ({projected_epoch_duration_s/60:.2f} min)")
    logger.log(f"Projected 25 Epochs: {projected_25_epochs_min:.1f} min ({projected_25_epochs_min/60:.2f} hrs)")
    logger.log(f"Best Epoch: {best_epoch} | Best Validation Loss: {best_val_loss:.6f}")
    logger.log(f"Best Checkpoint: {best_checkpoint_path}")
    logger.log(f"Latest Checkpoint: {latest_checkpoint_path}")
    logger.log("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="GenVision CS-TransUNet Controlled Training Pipeline")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per step (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM_STEPS, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Reproducibility seed (default: 42)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Execute short smoke test (2 epochs, limited batches)")
    parser.add_argument("--smoke-train-batches", type=int, default=4, help="Number of training batches for smoke test")
    parser.add_argument("--smoke-val-batches", type=int, default=2, help="Number of validation batches for smoke test")

    args = parser.parse_args()

    resume_path = Path(args.resume) if args.resume else None

    run_training(
        epochs=2 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        resume_path=resume_path,
        smoke_test=args.smoke_test,
        smoke_train_batches=args.smoke_train_batches,
        smoke_val_batches=args.smoke_val_batches,
    )


if __name__ == "__main__":
    main()
