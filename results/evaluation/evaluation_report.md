# GenVision held-out test evaluation

Generated: 2026-11-01

## Status

- Completed: **yes**
- Test images evaluated: **75 of 75**
- Errors: **0**
- Device: `cpu`
- Test indices: exactly the saved `test` list from `results/data_split.json`
- Train/validation leakage check: passed; test indices are disjoint from both sets

## Method

Both selected checkpoints were evaluated on the identical 75 cloudy/clean RGB
pairs. Each image was converted to RGB, resized to `(256, 256)`, and converted
with `torchvision.transforms.ToTensor()`, matching `dataset.py`. Models were
run in `eval()` mode under `torch.inference_mode()` on `cpu`. Predictions
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
| U-Net | 0.043854 | 0.042028 | 0.024963 | 26.6202 | 25.5174 | 5.2785 | 0.885598 | 0.888321 | 0.051277 |
| Attention U-Net | 0.082345 | 0.077231 | 0.030616 | 20.7718 | 21.0454 | 2.2938 | 0.797573 | 0.836479 | 0.111968 |

## Difference

Attention U-Net minus U-Net mean:

- L1: `+0.038491` (lower is better)
- PSNR: `-5.8483` dB (higher is better)
- SSIM: `-0.088025` (higher is better)

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
