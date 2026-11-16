"""
GenVision Phase 4.3: Controlled Held-Out Evaluation & Benchmarking
==================================================================

Conducts a scientifically rigorous, paired held-out evaluation across all three models:
1. Baseline U-Net (models/unet_controlled_best.pth)
2. Baseline Attention U-Net (models/attention_controlled_best.pth)
3. Proposed CS-TransUNet (models/transformer_controlled_best.pth, Epoch 24 checkpoint)

Strict Protocol:
- Evaluates exactly the 75 held-out scenes from results/data_split.json.
- Identical preprocessing: PIL RGB, Resize((256, 256)), ToTensor() -> [0, 1].
- Predictions strictly clamped to [0.0, 1.0].
- Identical metric formulations:
    - PSNR: skimage.metrics.peak_signal_noise_ratio(data_range=1.0)
    - SSIM: skimage.metrics.structural_similarity(channel_axis=0, data_range=1.0)
    - L1: torch.mean(torch.abs(pred - target))
- Latency profiling: mean, median, std inference time per image on CPU.
- Statistical significance: paired Student's t-test, Wilcoxon signed-rank test, 95% CI.
- Visual comparisons: 10 multi-panel figures with error maps.
- Artifact outputs:
    results/phase4/evaluation/per_image_metrics.csv
    results/phase4/evaluation/model_comparison.csv
    results/phase4/evaluation/evaluation_summary.json
    results/phase4/evaluation/visualizations/*.png
    results/phase4/PHASE4_3_HELD_OUT_EVALUATION_REPORT.md
"""

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torchvision.transforms as transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unet import UNet
from attention_unet import AttentionUNet
from models.transformer.cs_transunet import CSTransUNet


DEVICE = torch.device("cpu")
SPLIT_PATH = PROJECT_ROOT / "results" / "data_split.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4" / "evaluation"
VIZ_DIR = OUTPUT_DIR / "visualizations"

MODEL_PATHS = {
    "unet": (PROJECT_ROOT / "models" / "unet_controlled_best.pth", UNet),
    "attention_unet": (PROJECT_ROOT / "models" / "attention_controlled_best.pth", AttentionUNet),
    "cs_transunet": (PROJECT_ROOT / "models" / "transformer_controlled_best.pth", CSTransUNet),
}

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def load_dataset_split() -> Tuple[List[int], List[str]]:
    """Load and strictly verify data split."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Missing data split file at {SPLIT_PATH}")

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    test_indices = split["test"]
    filenames = split["filenames"]

    if len(test_indices) != 75:
        raise ValueError(f"Expected exactly 75 test indices, got {len(test_indices)}")

    train_set = set(split["train"])
    val_set = set(split["validation"])
    test_set = set(test_indices)

    if (test_set & train_set) or (test_set & val_set):
        raise ValueError("Data leakage detected: test indices overlap with train or val!")

    return test_indices, filenames


def load_models() -> Dict[str, Tuple[torch.nn.Module, int]]:
    """Load trained model weights and return model instances with parameter counts."""
    loaded = {}
    for name, (ckpt_path, model_cls) in MODEL_PATHS.items():
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = model_cls()
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        loaded[name] = (model, total_params)
        print(f"Loaded {name.upper()}: {ckpt_path.name} | Parameters: {total_params:,}")

    return loaded


def compute_metrics(pred_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Dict[str, float]:
    """Calculate strictly identical metrics across all models."""
    pred = pred_tensor.detach().cpu().clamp(0.0, 1.0)
    target = target_tensor.detach().cpu().clamp(0.0, 1.0)

    # L1 / MAE
    l1 = torch.mean(torch.abs(pred - target)).item()

    # NumPy arrays (C, H, W) -> float in [0, 1]
    pred_np = pred.numpy()
    target_np = target.numpy()

    # PSNR
    psnr = float(peak_signal_noise_ratio(target_np, pred_np, data_range=1.0))

    # SSIM
    ssim = float(structural_similarity(target_np, pred_np, channel_axis=0, data_range=1.0))

    return {
        "l1_loss": l1,
        "psnr": psnr,
        "ssim": ssim,
    }


def tensor_to_np_img(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor to (H, W, C) numpy uint8 array."""
    arr = t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def calculate_ci95(diffs: np.ndarray) -> Tuple[float, float]:
    """Calculate 95% confidence interval for paired differences using Student's t distribution."""
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    sem = std_diff / math.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return mean_diff - t_crit * sem, mean_diff + t_crit * sem


def create_visual_comparison(
    save_path: Path,
    cloudy_t: torch.Tensor,
    clean_t: torch.Tensor,
    preds_t: Dict[str, torch.Tensor],
    metrics_dict: Dict[str, Dict[str, float]],
    scene_name: str,
):
    """Generate high-resolution multi-panel visual comparison with absolute error heatmaps."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    cloudy_img = tensor_to_np_img(cloudy_t)
    clean_img = tensor_to_np_img(clean_t)

    unet_img = tensor_to_np_img(preds_t["unet"])
    att_img = tensor_to_np_img(preds_t["attention_unet"])
    cs_img = tensor_to_np_img(preds_t["cs_transunet"])

    # Row 1: RGB Reconstructions
    axes[0, 0].imshow(cloudy_img)
    axes[0, 0].set_title(f"Input: Cloudy Scene\n({scene_name})", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    m_u = metrics_dict["unet"]
    axes[0, 1].imshow(unet_img)
    axes[0, 1].set_title(
        f"U-Net Baseline\nPSNR: {m_u['psnr']:.2f} dB | SSIM: {m_u['ssim']:.4f}\nL1: {m_u['l1_loss']:.4f}",
        fontsize=10,
    )
    axes[0, 1].axis("off")

    m_a = metrics_dict["attention_unet"]
    axes[0, 2].imshow(att_img)
    axes[0, 2].set_title(
        f"Attention U-Net Baseline\nPSNR: {m_a['psnr']:.2f} dB | SSIM: {m_a['ssim']:.4f}\nL1: {m_a['l1_loss']:.4f}",
        fontsize=10,
    )
    axes[0, 2].axis("off")

    m_cs = metrics_dict["cs_transunet"]
    axes[0, 3].imshow(cs_img)
    axes[0, 3].set_title(
        f"CS-TransUNet (Ours)\nPSNR: {m_cs['psnr']:.2f} dB | SSIM: {m_cs['ssim']:.4f}\nL1: {m_cs['l1_loss']:.4f}",
        fontsize=10,
        fontweight="bold",
        color="#047857",
    )
    axes[0, 3].axis("off")

    # Row 2: Ground Truth and Error Maps
    axes[1, 0].imshow(clean_img)
    axes[1, 0].set_title("Ground-Truth Clean Image", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Absolute error maps
    err_u = torch.abs(preds_t["unet"] - clean_t).mean(dim=0).cpu().numpy()
    err_a = torch.abs(preds_t["attention_unet"] - clean_t).mean(dim=0).cpu().numpy()
    err_cs = torch.abs(preds_t["cs_transunet"] - clean_t).mean(dim=0).cpu().numpy()

    # Fixed error scale for fair comparison
    vmax = max(float(err_u.max()), float(err_a.max()), float(err_cs.max()), 0.25)
    vmax = min(vmax, 0.5)

    im1 = axes[1, 1].imshow(err_u, cmap="inferno", vmin=0.0, vmax=vmax)
    axes[1, 1].set_title(f"U-Net Absolute Error\nMean L1: {err_u.mean():.4f}", fontsize=10)
    axes[1, 1].axis("off")

    im2 = axes[1, 2].imshow(err_a, cmap="inferno", vmin=0.0, vmax=vmax)
    axes[1, 2].set_title(f"Attention U-Net Absolute Error\nMean L1: {err_a.mean():.4f}", fontsize=10)
    axes[1, 2].axis("off")

    im3 = axes[1, 3].imshow(err_cs, cmap="inferno", vmin=0.0, vmax=vmax)
    axes[1, 3].set_title(f"CS-TransUNet Absolute Error\nMean L1: {err_cs.mean():.4f}", fontsize=10, fontweight="bold", color="#047857")
    axes[1, 3].axis("off")

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.3])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label("Absolute Error Magnitude", fontsize=10)

    fig.suptitle(f"GenVision Held-Out Benchmark Visual Comparison: {scene_name}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.90, 0.95])
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 80)
    print("GENVISION PHASE 4.3: CONTROLLED HELD-OUT EVALUATION & BENCHMARKING")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Split Verification
    test_indices, filenames = load_dataset_split()
    print(f"Loaded {len(test_indices)} held-out test scenes.")

    # 2. Model Loading & Verification
    models_dict = load_models()

    # 3. Warm-up passes
    dummy_x = torch.rand(1, 3, 256, 256).to(DEVICE)
    with torch.inference_mode():
        for name, (model, _) in models_dict.items():
            if name == "cs_transunet":
                _ = model(dummy_x, return_dict=False)
            else:
                _ = model(dummy_x)
    print("Warm-up complete. Commencing paired per-image evaluation...")

    # 4. Evaluation Loop
    per_image_rows = []
    wide_rows = []
    stored_preds = {}

    cloudy_dir = PROJECT_ROOT / "dataset" / "cloudy"
    clean_dir = PROJECT_ROOT / "dataset" / "clean"

    for idx_num, (index, filename) in enumerate(zip(test_indices, [filenames[i] for i in test_indices])):
        cloudy_path = cloudy_dir / filename
        clean_path = clean_dir / filename

        with Image.open(cloudy_path) as img:
            cloudy_t = TRANSFORM(img.convert("RGB"))
        with Image.open(clean_path) as img:
            clean_t = TRANSFORM(img.convert("RGB"))

        input_batch = cloudy_t.unsqueeze(0).to(DEVICE)

        scene_metrics = {}
        scene_preds = {}
        scene_times = {}

        for name, (model, _) in models_dict.items():
            # Timed inference pass
            t0 = time.perf_counter()
            with torch.inference_mode():
                if name == "cs_transunet":
                    pred_batch = model(input_batch, return_dict=False)
                else:
                    pred_batch = model(input_batch)
            t1 = time.perf_counter()
            inf_time_ms = (t1 - t0) * 1000.0

            pred_t = pred_batch.squeeze(0).cpu().clamp(0.0, 1.0)
            m = compute_metrics(pred_t, clean_t)

            scene_metrics[name] = m
            scene_preds[name] = pred_t
            scene_times[name] = inf_time_ms

            # Long format row
            per_image_rows.append({
                "index": index,
                "filename": filename,
                "model": name,
                "psnr": m["psnr"],
                "ssim": m["ssim"],
                "l1_loss": m["l1_loss"],
                "inference_time_ms": inf_time_ms,
            })

        # Wide format row for comparison
        wide_rows.append({
            "index": index,
            "filename": filename,
            "unet_psnr": scene_metrics["unet"]["psnr"],
            "unet_ssim": scene_metrics["unet"]["ssim"],
            "unet_l1_loss": scene_metrics["unet"]["l1_loss"],
            "unet_time_ms": scene_times["unet"],
            "attention_unet_psnr": scene_metrics["attention_unet"]["psnr"],
            "attention_unet_ssim": scene_metrics["attention_unet"]["ssim"],
            "attention_unet_l1_loss": scene_metrics["attention_unet"]["l1_loss"],
            "attention_unet_time_ms": scene_times["attention_unet"],
            "cs_transunet_psnr": scene_metrics["cs_transunet"]["psnr"],
            "cs_transunet_ssim": scene_metrics["cs_transunet"]["ssim"],
            "cs_transunet_l1_loss": scene_metrics["cs_transunet"]["l1_loss"],
            "cs_transunet_time_ms": scene_times["cs_transunet"],
        })

        # Store for visualizations
        stored_preds[filename] = {
            "index": index,
            "cloudy": cloudy_t,
            "clean": clean_t,
            "preds": scene_preds,
            "metrics": scene_metrics,
        }

        if (idx_num + 1) % 15 == 0 or (idx_num + 1) == len(test_indices):
            print(f"Evaluated {idx_num + 1:02d} / {len(test_indices)} test images.")

    # 5. Save per_image_metrics.csv
    per_image_csv_path = OUTPUT_DIR / "per_image_metrics.csv"
    with open(per_image_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "filename", "model", "psnr", "ssim", "l1_loss", "inference_time_ms"])
        writer.writeheader()
        writer.writerows(per_image_rows)

    # 6. Statistical Computations
    models_summary = {}
    for name in ["unet", "attention_unet", "cs_transunet"]:
        psnr_vals = np.array([r["psnr"] for r in per_image_rows if r["model"] == name])
        ssim_vals = np.array([r["ssim"] for r in per_image_rows if r["model"] == name])
        l1_vals = np.array([r["l1_loss"] for r in per_image_rows if r["model"] == name])
        time_vals = np.array([r["inference_time_ms"] for r in per_image_rows if r["model"] == name])

        models_summary[name] = {
            "parameters": models_dict[name][1],
            "psnr": {
                "mean": float(np.mean(psnr_vals)),
                "median": float(np.median(psnr_vals)),
                "std": float(np.std(psnr_vals, ddof=1)),
                "min": float(np.min(psnr_vals)),
                "max": float(np.max(psnr_vals)),
            },
            "ssim": {
                "mean": float(np.mean(ssim_vals)),
                "median": float(np.median(ssim_vals)),
                "std": float(np.std(ssim_vals, ddof=1)),
                "min": float(np.min(ssim_vals)),
                "max": float(np.max(ssim_vals)),
            },
            "l1_loss": {
                "mean": float(np.mean(l1_vals)),
                "median": float(np.median(l1_vals)),
                "std": float(np.std(l1_vals, ddof=1)),
                "min": float(np.min(l1_vals)),
                "max": float(np.max(l1_vals)),
            },
            "inference_time_ms": {
                "mean": float(np.mean(time_vals)),
                "median": float(np.median(time_vals)),
                "std": float(np.std(time_vals, ddof=1)),
                "min": float(np.min(time_vals)),
                "max": float(np.max(time_vals)),
            },
        }

    # 7. Paired Comparisons & Statistical Significance Tests
    cs_psnr = np.array([r["cs_transunet_psnr"] for r in wide_rows])
    cs_ssim = np.array([r["cs_transunet_ssim"] for r in wide_rows])
    cs_l1 = np.array([r["cs_transunet_l1_loss"] for r in wide_rows])

    u_psnr = np.array([r["unet_psnr"] for r in wide_rows])
    u_ssim = np.array([r["unet_ssim"] for r in wide_rows])
    u_l1 = np.array([r["unet_l1_loss"] for r in wide_rows])

    att_psnr = np.array([r["attention_unet_psnr"] for r in wide_rows])
    att_ssim = np.array([r["attention_unet_ssim"] for r in wide_rows])
    att_l1 = np.array([r["attention_unet_l1_loss"] for r in wide_rows])

    # Comparison Rows CSV
    model_comp_rows = []
    for r in wide_rows:
        model_comp_rows.append({
            "index": r["index"],
            "filename": r["filename"],
            "diff_psnr_cs_minus_unet": r["cs_transunet_psnr"] - r["unet_psnr"],
            "diff_ssim_cs_minus_unet": r["cs_transunet_ssim"] - r["unet_ssim"],
            "diff_l1_cs_minus_unet": r["cs_transunet_l1_loss"] - r["unet_l1_loss"],
            "diff_psnr_cs_minus_att": r["cs_transunet_psnr"] - r["attention_unet_psnr"],
            "diff_ssim_cs_minus_att": r["cs_transunet_ssim"] - r["attention_unet_ssim"],
            "diff_l1_cs_minus_att": r["cs_transunet_l1_loss"] - r["attention_unet_l1_loss"],
        })

    model_comp_csv_path = OUTPUT_DIR / "model_comparison.csv"
    with open(model_comp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(model_comp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(model_comp_rows)

    def paired_stats(cs_arr, base_arr, metric_name, lower_is_better=False):
        diff = cs_arr - base_arr
        mean_d = float(np.mean(diff))
        median_d = float(np.median(diff))
        std_d = float(np.std(diff, ddof=1))
        ci_low, ci_high = calculate_ci95(diff)
        base_mean = float(np.mean(base_arr))

        if lower_is_better:
            rel_imp_pct = ((base_mean - float(np.mean(cs_arr))) / base_mean) * 100.0
        else:
            rel_imp_pct = (mean_d / base_mean) * 100.0

        t_res = stats.ttest_rel(cs_arr, base_arr)
        w_res = stats.wilcoxon(diff)

        return {
            "mean_difference": mean_d,
            "median_difference": median_d,
            "std_difference": std_d,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "relative_improvement_percent": rel_imp_pct,
            "paired_t_test": {
                "t_statistic": float(t_res.statistic),
                "p_value": float(t_res.pvalue),
                "significant_0_05": bool(t_res.pvalue < 0.05),
                "significant_0_01": bool(t_res.pvalue < 0.01),
            },
            "wilcoxon_signed_rank": {
                "w_statistic": float(w_res.statistic),
                "p_value": float(w_res.pvalue),
                "significant_0_05": bool(w_res.pvalue < 0.05),
                "significant_0_01": bool(w_res.pvalue < 0.01),
            },
        }

    statistical_comparisons = {
        "cs_transunet_vs_unet": {
            "psnr": paired_stats(cs_psnr, u_psnr, "psnr", lower_is_better=False),
            "ssim": paired_stats(cs_ssim, u_ssim, "ssim", lower_is_better=False),
            "l1_loss": paired_stats(cs_l1, u_l1, "l1_loss", lower_is_better=True),
        },
        "cs_transunet_vs_attention_unet": {
            "psnr": paired_stats(cs_psnr, att_psnr, "psnr", lower_is_better=False),
            "ssim": paired_stats(cs_ssim, att_ssim, "ssim", lower_is_better=False),
            "l1_loss": paired_stats(cs_l1, att_l1, "l1_loss", lower_is_better=True),
        },
    }

    # 8. Visual Comparisons: Select 10 Diverse Scenes
    # Sort test scenes by difficulty/PSNR to ensure balanced representation:
    # 3 high PSNR (thin haze/light clouds), 4 medium PSNR, 3 lower PSNR (dense cloud/challenging textures)
    sorted_scenes = sorted(wide_rows, key=lambda x: x["cs_transunet_psnr"])
    n_total = len(sorted_scenes)

    # 10 evenly spaced representative scenes across the difficulty spectrum
    sample_indices_list = [
        0,  # Hardest test case
        int(n_total * 0.12),
        int(n_total * 0.25),
        int(n_total * 0.38),
        int(n_total * 0.50),  # Median case
        int(n_total * 0.62),
        int(n_total * 0.75),
        int(n_total * 0.85),
        int(n_total * 0.92),
        n_total - 1,  # Easiest test case
    ]

    print("Generating 10 representative visual comparison panels...")
    for comp_idx, s_idx in enumerate(sample_indices_list):
        row = sorted_scenes[s_idx]
        fname = row["filename"]
        data = stored_preds[fname]
        stem = Path(fname).stem
        viz_path = VIZ_DIR / f"comparison_{comp_idx+1:02d}_{stem}.png"

        create_visual_comparison(
            save_path=viz_path,
            cloudy_t=data["cloudy"],
            clean_t=data["clean"],
            preds_t=data["preds"],
            metrics_dict=data["metrics"],
            scene_name=f"{fname} (Scene {data['index']})",
        )

    # 9. Boxplot summary visualizations
    def make_boxplot(metric_name, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(8, 5))
        data_to_plot = [
            [r[f"unet_{metric_name}"] for r in wide_rows],
            [r[f"attention_unet_{metric_name}"] for r in wide_rows],
            [r[f"cs_transunet_{metric_name}"] for r in wide_rows],
        ]
        bp = ax.boxplot(
            data_to_plot,
            tick_labels=["Baseline U-Net", "Attention U-Net", "CS-TransUNet (Ours)"],
            patch_artist=True,
            medianprops={"color": "#111827", "linewidth": 2.2},
        )
        colors = ["#93c5fd", "#c4b5fd", "#6ee7b7"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(fig)

    make_boxplot("psnr", "Held-Out Test Set PSNR Distribution (Higher is Better)", "PSNR (dB)", "test_psnr_boxplot.png")
    make_boxplot("ssim", "Held-Out Test Set SSIM Distribution (Higher is Better)", "SSIM", "test_ssim_boxplot.png")
    make_boxplot("l1_loss", "Held-Out Test Set L1 Reconstruction Error (Lower is Better)", "L1 Error / MAE", "test_l1_boxplot.png")

    # 10. Save evaluation_summary.json
    summary_data = {
        "evaluation_protocol": {
            "test_images_count": len(test_indices),
            "device": str(DEVICE),
            "preprocessing": "PIL RGB conversion, torchvision.transforms.Resize((256, 256)), ToTensor()",
            "prediction_clamp": "[0.0, 1.0]",
            "metrics": {
                "psnr": "skimage.metrics.peak_signal_noise_ratio(data_range=1.0)",
                "ssim": "skimage.metrics.structural_similarity(channel_axis=0, data_range=1.0)",
                "l1_loss": "torch.mean(torch.abs(pred - target))",
            },
            "best_checkpoint_evaluated": {
                "unet": "models/unet_controlled_best.pth",
                "attention_unet": "models/attention_controlled_best.pth",
                "cs_transunet": "models/transformer_controlled_best.pth (Epoch 24 checkpoint)",
            },
        },
        "model_metrics": models_summary,
        "statistical_comparisons": statistical_comparisons,
    }

    with open(OUTPUT_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Per-image metrics: {per_image_csv_path}")
    print(f"Model comparison: {model_comp_csv_path}")
    print(f"Summary JSON: {OUTPUT_DIR / 'evaluation_summary.json'}")
    print(f"Visualizations: {VIZ_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
