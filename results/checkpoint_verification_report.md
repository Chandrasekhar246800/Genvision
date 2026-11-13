# GenVision checkpoint verification report

Generated: 2026-11-01  
Project: `C:\Users\mcsr8\Desktop\cloud-removal-poc`  
Verifier: [`results/verify_checkpoints.py`](./verify_checkpoints.py)

## Scope and method

This was a non-destructive verification. Every `models/*.pth` file was inspected
without writing to, replacing, deleting, or retraining any artifact.

For each non-empty checkpoint, the verifier:

1. Selected U-Net or Attention U-Net from the checkpoint filename.
2. Loaded the state dictionary with `torch.load(..., weights_only=True)` on CPU.
3. Ran a dummy tensor of shape `(1, 3, 256, 256)`.
4. Checked the output shape and verified that all output values were finite.
5. Recorded file size and SHA-256.

## Checkpoints inspected

| Checkpoint | Architecture | Role | Size | Status |
|---|---|---:|---:|---|
| `models/attention_controlled_best.pth` | Attention U-Net | Best | 125,682,635 bytes | Verified |
| `models/attention_controlled_final.pth` | Attention U-Net | Final | 125,682,843 bytes | Verified |
| `models/attention_unet.pth` | Attention U-Net | Final | 125,680,219 bytes | Verified |
| `models/best_attention_unet.pth` | Attention U-Net | Best | 125,681,323 bytes | Verified |
| `models/unet.pth` | U-Net | Final | 30,800,505 bytes | Verified |
| `models/unet_baseline.pth` | U-Net | Baseline/unknown | 0 bytes | Invalid: empty |
| `models/unet_controlled_best.pth` | U-Net | Best | 30,803,417 bytes | Verified |
| `models/unet_controlled_final.pth` | U-Net | Final | 30,803,459 bytes | Verified |

## Loading and forward-pass results

Seven of eight checkpoints loaded successfully. Every successfully loaded
checkpoint produced:

```text
Input:  (1, 3, 256, 256)
Output: (1, 3, 256, 256)
Finite: True
```

`models/unet_baseline.pth` is confirmed to be empty. Its SHA-256 is the standard
empty-file hash:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

It was not loaded or modified.

Complete hashes and machine-readable verification data are in
[`results/experiment_manifest.json`](./experiment_manifest.json).

## Training configurations

### `train.py`

- Architecture: U-Net
- Optimizer: Adam
- Learning rate: `0.0001`
- Batch size: `2`
- Epochs: `20`
- Loss: `L1Loss`
- Seed: `42` for Python, NumPy, and PyTorch
- Scheduler: none
- Split: all 500 images; no explicit validation split
- Output: `models/unet.pth`

### `train_attention.py`

- Architecture: Attention U-Net
- Optimizer: Adam
- Learning rate: `0.0001`
- Batch size: `2`
- Epochs: `20`
- Loss: `L1Loss`
- Seed: `42`
- Scheduler: none
- Split: random 80/20 split, 400 train and 100 validation
- Outputs: `models/best_attention_unet.pth`, `models/attention_unet.pth`

### `train_attention_v2.py`

- Architecture: Attention U-Net
- Optimizer: Adam
- Learning rate: `0.0001`
- Batch size: `2`
- Epochs: `20`
- Loss: `L1Loss`
- Seed: `42`
- Scheduler: `ReduceLROnPlateau`, factor `0.5`, patience `3`, minimum
  learning rate `1e-6`
- Split: random 80/20 split, 400 train and 100 validation
- Outputs: same original Attention U-Net filenames
- Important: executing this script would overwrite the original Attention
  checkpoint filenames; it was not executed.

### Controlled scripts

`train_unet_controlled.py` and `train_attention_controlled.py` share:

- Optimizer: Adam
- Learning rate: `0.0001`
- Batch size: `2`
- Epochs: `20`
- Loss: `L1Loss`
- Seed: `42`
- Scheduler: `ReduceLROnPlateau`, factor `0.5`, patience `3`, minimum
  learning rate `1e-6`
- Split: `results/data_split.json`, with 350 train, 75 validation, and 75 test

They write separate controlled best/final checkpoints and JSON histories.

## Training-history evidence

Available controlled histories provide the only direct checkpoint-selection
evidence tied to a saved best checkpoint:

| Architecture | Best checkpoint | Best epoch | Best validation L1 |
|---|---|---:|---:|
| U-Net | `models/unet_controlled_best.pth` | 19 | 0.041781690045891506 |
| Attention U-Net | `models/attention_controlled_best.pth` | 18 | 0.08260762877762318 |

The older `models/unet.pth`, `models/attention_unet.pth`, and
`models/best_attention_unet.pth` files are loadable, but no unambiguous history
file was found that proves which exact run produced each older file. The
`results/baseline_metrics.txt` file contains single-image metrics, but it does
not establish a held-out test evaluation and was not used as definitive
checkpoint-selection evidence.

## Recommended checkpoints

### Recommended U-Net

**Provisional:** `models/unet_controlled_best.pth`

This is the only U-Net checkpoint directly tied to available controlled
training-history evidence, and the history identifies it as the minimum
validation-loss checkpoint. This is not a claim that it is best on unseen test
data.

### Recommended Attention U-Net

**Provisional:** `models/attention_controlled_best.pth`

This is the only Attention U-Net checkpoint directly tied to available
controlled training-history evidence, and the history identifies it as the
minimum validation-loss checkpoint. This is not a claim that it is best on
unseen test data.

## `unet_baseline.pth` investigation

- The file exists in `models/`.
- File size is `0` bytes.
- Loading was not attempted because the file is empty.
- The only source reference found is the legacy path in
  `check_checkpoints.py`; no training script writes this filename.
- No matching history, manifest, or checkpoint provenance was found.
- The file was not recreated, overwritten, or deleted.

## Missing experiment information

- No aggregate evaluation over the 75-image held-out test split.
- No per-checkpoint PSNR/SSIM/L1 results on the test split.
- No confidence intervals or repeated-run variance.
- No complete lineage for the original non-controlled checkpoint files.
- No Python dependency manifest.
- No documented rationale for the empty baseline artifact.
- The architecture constructor in `attention_unet.py` accepts channel arguments,
  but its encoder is hardcoded for three input channels; this was observed but
  not changed under the task restrictions.

## Validation and artifact preservation

- Verification script: passed for seven checkpoints; correctly reported one
  empty checkpoint.
- Existing model sizes and SHA-256 hashes were recorded before final reporting.
- No existing checkpoint, architecture, frontend file, dataset, or training
  history was modified.
- New files created:
  - `results/verify_checkpoints.py`
  - `results/experiment_manifest.json`
  - `results/checkpoint_verification_report.md`

## Recommended next step

Implement a read-only, deterministic evaluation over all 75 held-out test
pairs, using the saved split and the two provisional recommended checkpoints.
Report aggregate and per-image L1, PSNR, and SSIM without overwriting existing
outputs or checkpoints.
