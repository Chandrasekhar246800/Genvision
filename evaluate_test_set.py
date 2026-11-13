"""Reproducible held-out evaluation for the selected GenVision checkpoints."""

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from attention_unet import AttentionUNet
from unet import UNet


SPLIT_PATH = ROOT / "results" / "data_split.json"
OUTPUT_DIR = ROOT / "results" / "evaluation"
SAMPLE_DIR = OUTPUT_DIR / "sample_predictions"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (256, 256)
TEST_COUNT = 75
MODELS = {
    "unet": (ROOT / "models" / "unet_controlled_best.pth", UNet),
    "attention_unet": (
        ROOT / "models" / "attention_controlled_best.pth",
        AttentionUNet,
    ),
}
TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)


def load_split() -> tuple[list[int], list[str]]:
    with SPLIT_PATH.open("r", encoding="utf-8") as handle:
        split = json.load(handle)

    test_indices = split["test"]
    filenames = split["filenames"]
    if len(test_indices) != TEST_COUNT:
        raise ValueError(f"Expected {TEST_COUNT} test indices, got {len(test_indices)}.")

    train = set(split["train"])
    validation = set(split["validation"])
    test = set(test_indices)
    if test & train or test & validation:
        raise ValueError("Test indices overlap with train or validation indices.")
    if len(test) != TEST_COUNT:
        raise ValueError("Test indices contain duplicates.")

    return test_indices, filenames


def load_models() -> dict[str, torch.nn.Module]:
    loaded = {}
    for name, (checkpoint_path, model_class) in MODELS.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = model_class()
        state_dict = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        loaded[name] = model
    return loaded


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return TRANSFORM(image.convert("RGB"))


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.detach().cpu().clamp(0, 1)
    target = target.detach().cpu()
    prediction_np = prediction.numpy()
    target_np = target.numpy()
    l1 = torch.mean(torch.abs(prediction - target)).item()
    return {
        "l1_loss": float(l1),
        "psnr": float(
            peak_signal_noise_ratio(target_np, prediction_np, data_range=1.0)
        ),
        "ssim": float(
            structural_similarity(
                target_np,
                prediction_np,
                channel_axis=0,
                data_range=1.0,
            )
        ),
    }


def tensor_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def save_samples(
    samples: dict[str, dict[str, torch.Tensor]], sample_indices: list[int]
) -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for sample_index in sample_indices:
        sample = samples[str(sample_index)]
        filename = sample["filename"]
        stem = Path(filename).stem
        for label, tensor in (
            ("cloudy_input", sample["cloudy"]),
            ("clean_ground_truth", sample["clean"]),
            ("unet_output", sample["unet"]),
            ("attention_unet_output", sample["attention_unet"]),
        ):
            output_path = SAMPLE_DIR / f"{stem}_{label}.png"
            Image.fromarray((tensor_image(tensor) * 255).round().astype(np.uint8)).save(
                output_path
            )


def aggregate(rows: list[dict[str, float]], model: str) -> dict[str, float | int]:
    values = {metric: np.array([row[metric] for row in rows]) for metric in ("l1_loss", "psnr", "ssim")}
    reducers = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }
    return {
        "image_count": len(rows),
        **{
            f"{metric}_{stat}": float(reducers[stat](series))
            for metric, series in values.items()
            for stat in ("mean", "median", "std", "min", "max")
        },
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    filename: str,
    metric: str,
    unit: str,
    unet_values: list[float],
    attention_values: list[float],
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(
        [unet_values, attention_values],
        tick_labels=["U-Net", "Attention U-Net"],
        patch_artist=True,
        boxprops={"facecolor": "#55d6f2", "alpha": 0.65},
        medianprops={"color": "#111827", "linewidth": 2},
    )
    boxes = axis.artists
    if len(boxes) > 1:
        boxes[1].set_facecolor("#a78bfa")
    axis.set_title(f"Held-out test {metric.upper()} comparison")
    axis.set_ylabel(unit)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close(figure)


def main() -> None:
    test_indices, filenames = load_split()
    models = load_models()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    samples = {}
    errors = []
    test_names = [filenames[index] for index in test_indices]
    sample_positions = sorted({0, len(test_indices) // 4, len(test_indices) // 2, (3 * len(test_indices)) // 4, len(test_indices) - 1})
    sample_indices = [test_indices[position] for position in sample_positions]

    for index, filename in zip(test_indices, test_names):
        cloudy_path = ROOT / "dataset" / "cloudy" / filename
        clean_path = ROOT / "dataset" / "clean" / filename
        try:
            cloudy = image_tensor(cloudy_path)
            clean = image_tensor(clean_path)
            input_batch = cloudy.unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                predictions = {
                    name: model(input_batch).squeeze(0).cpu().clamp(0, 1)
                    for name, model in models.items()
                }
            row = {"index": index, "filename": filename}
            for name, prediction in predictions.items():
                row.update({f"{name}_{key}": value for key, value in metrics(prediction, clean).items()})
            if not all(math.isfinite(float(value)) for value in row.values() if isinstance(value, (int, float))):
                raise ValueError("Non-finite metric value.")
            rows.append(row)
            if index in sample_indices:
                samples[str(index)] = {
                    "filename": filename,
                    "cloudy": cloudy,
                    "clean": clean,
                    **predictions,
                }
        except Exception as error:
            errors.append({"index": index, "filename": filename, "error": f"{type(error).__name__}: {error}"})

    if errors:
        error_text = "\n".join(f"{item['filename']}: {item['error']}" for item in errors)
        raise RuntimeError(f"Evaluation failed for {len(errors)} image(s):\n{error_text}")
    if len(rows) != TEST_COUNT or len(samples) != 5:
        raise RuntimeError(f"Expected 75 evaluated rows and 5 samples, got {len(rows)} and {len(samples)}.")

    per_image_fields = [
        "index", "filename", "unet_l1_loss", "unet_psnr", "unet_ssim",
        "attention_unet_l1_loss", "attention_unet_psnr", "attention_unet_ssim",
    ]
    write_csv(OUTPUT_DIR / "per_image_metrics.csv", per_image_fields, rows)

    comparison_rows = []
    for row in rows:
        comparison_rows.append(
            {
                "index": row["index"],
                "filename": row["filename"],
                "l1_attention_minus_unet": row["attention_unet_l1_loss"] - row["unet_l1_loss"],
                "psnr_attention_minus_unet": row["attention_unet_psnr"] - row["unet_psnr"],
                "ssim_attention_minus_unet": row["attention_unet_ssim"] - row["unet_ssim"],
            }
        )
    write_csv(
        OUTPUT_DIR / "model_comparison.csv",
        list(comparison_rows[0].keys()),
        comparison_rows,
    )

    unet_rows = [{metric: row[f"unet_{metric}"] for metric in ("l1_loss", "psnr", "ssim")} for row in rows]
    attention_rows = [{metric: row[f"attention_unet_{metric}"] for metric in ("l1_loss", "psnr", "ssim")} for row in rows]
    aggregate_data = {
        "evaluation": {
            "test_count": TEST_COUNT,
            "evaluated_count": len(rows),
            "device": str(DEVICE),
            "preprocessing": "PIL RGB conversion, torchvision Resize((256, 256)), ToTensor()",
            "prediction_clamp": "[0, 1]",
            "l1_definition": "mean absolute error on CHW tensors in [0, 1]",
            "psnr_definition": "skimage peak_signal_noise_ratio with data_range=1.0",
            "ssim_definition": "skimage structural_similarity with channel_axis=0 and data_range=1.0",
            "test_indices": test_indices,
            "errors": errors,
        },
        "models": {
            "unet": aggregate(unet_rows, "unet"),
            "attention_unet": aggregate(attention_rows, "attention_unet"),
        },
    }
    aggregate_data["comparison"] = {
        "attention_minus_unet": {
            metric: aggregate_data["models"]["attention_unet"][f"{metric}_mean"]
            - aggregate_data["models"]["unet"][f"{metric}_mean"]
            for metric in ("l1_loss", "psnr", "ssim")
        }
    }
    with (OUTPUT_DIR / "aggregate_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_data, handle, indent=2)

    save_samples(samples, sample_indices)
    save_plot("psnr_comparison.png", "PSNR", "dB", [row["unet_psnr"] for row in rows], [row["attention_unet_psnr"] for row in rows])
    save_plot("ssim_comparison.png", "SSIM", "SSIM", [row["unet_ssim"] for row in rows], [row["attention_unet_ssim"] for row in rows])
    save_plot("l1_comparison.png", "L1 loss", "Mean absolute error", [row["unet_l1_loss"] for row in rows], [row["attention_unet_l1_loss"] for row in rows])

    unet_aggregate = aggregate_data["models"]["unet"]
    attention_aggregate = aggregate_data["models"]["attention_unet"]
    comparison = aggregate_data["comparison"]["attention_minus_unet"]
    report = f"""# GenVision held-out test evaluation

Generated: 2026-11-01

## Status

- Completed: **yes**
- Test images evaluated: **{len(rows)} of {TEST_COUNT}**
- Errors: **{len(errors)}**
- Device: `{DEVICE}`
- Test indices: exactly the saved `test` list from `results/data_split.json`
- Train/validation leakage check: passed; test indices are disjoint from both sets

## Method

Both selected checkpoints were evaluated on the identical 75 cloudy/clean RGB
pairs. Each image was converted to RGB, resized to `(256, 256)`, and converted
with `torchvision.transforms.ToTensor()`, matching `dataset.py`. Models were
run in `eval()` mode under `torch.inference_mode()` on `{DEVICE}`. Predictions
were clamped to `[0, 1]` consistently before metric calculation and saving.

- L1: mean absolute error on tensors in `[0, 1]`.
- PSNR: `skimage.metrics.peak_signal_noise_ratio`, `data_range=1.0`.
- SSIM: `skimage.metrics.structural_similarity`, `channel_axis=0`,
  `data_range=1.0`.

## Selected checkpoints

- U-Net: `models/unet_controlled_best.pth`
- Attention U-Net: `models/attention_controlled_best.pth`

## Aggregate metrics

| Model | L1 mean | L1 median | L1 std | PSNR mean | PSNR median | PSNR std | SSIM mean | SSIM median | SSIM std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | {unet_aggregate['l1_loss_mean']:.6f} | {unet_aggregate['l1_loss_median']:.6f} | {unet_aggregate['l1_loss_std']:.6f} | {unet_aggregate['psnr_mean']:.4f} | {unet_aggregate['psnr_median']:.4f} | {unet_aggregate['psnr_std']:.4f} | {unet_aggregate['ssim_mean']:.6f} | {unet_aggregate['ssim_median']:.6f} | {unet_aggregate['ssim_std']:.6f} |
| Attention U-Net | {attention_aggregate['l1_loss_mean']:.6f} | {attention_aggregate['l1_loss_median']:.6f} | {attention_aggregate['l1_loss_std']:.6f} | {attention_aggregate['psnr_mean']:.4f} | {attention_aggregate['psnr_median']:.4f} | {attention_aggregate['psnr_std']:.4f} | {attention_aggregate['ssim_mean']:.6f} | {attention_aggregate['ssim_median']:.6f} | {attention_aggregate['ssim_std']:.6f} |

## Difference

Attention U-Net minus U-Net mean:

- L1: `{comparison['l1_loss']:+.6f}` (lower is better)
- PSNR: `{comparison['psnr']:+.4f}` dB (higher is better)
- SSIM: `{comparison['ssim']:+.6f}` (higher is better)

These are actual held-out test-set results. Any claim that one model is better
should be based on the direction of all relevant metrics and should acknowledge
that this is one fixed 75-image split.

## Samples

Five deterministic test examples were saved under
`results/evaluation/sample_predictions/`. Each contains the cloudy input, clean
ground truth, U-Net output, and Attention U-Net output for the same scene.

## Output files

- `per_image_metrics.csv`
- `aggregate_metrics.json`
- `model_comparison.csv`
- `evaluation_report.md`
- `psnr_comparison.png`
- `ssim_comparison.png`
- `l1_comparison.png`
- `sample_predictions/` (20 PNG files)
"""
    (OUTPUT_DIR / "evaluation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"evaluated": len(rows), "errors": errors, "device": str(DEVICE)}, indent=2))


if __name__ == "__main__":
    main()
