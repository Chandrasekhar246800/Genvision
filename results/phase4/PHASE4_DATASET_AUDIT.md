# GenVision Phase 4 — Comprehensive Dataset Audit & Transformer Suitability Analysis

**Project Title:** GENVISION: A CLOUD SEVERITY-AWARE TRANSFORMER-DIFFUSION FRAMEWORK FOR CLOUD-FREE SATELLITE IMAGE RECONSTRUCTION AND INTELLIGENT AGRICULTURAL ANALYSIS  
**Document ID:** `results/phase4/PHASE4_DATASET_AUDIT.md`  
**Phase:** 4 — Dataset Audit & Preprocessing Specification  
**Date:** 16 November 2026  
**Status:** **AUDIT VERIFIED & VALIDATED**  
**Authors:** Senior AI Research Engineer, Computer Vision Architect & Remote Sensing Scientist  

---

## 1. Executive Summary

This audit evaluates the experimental datasets within GenVision to establish strict data boundaries, eliminate data leakage risks, and assess suitability for training the proposed **Cloud Severity-Aware Hybrid Transformer (CS-TransUNet)**.

### Key Audit Findings:
1. **Paired RGB Dataset Integrity:** Exactly **500 paired image files** exist in `dataset/cloudy/` and `dataset/clean/` with **100% 1-to-1 filename correspondence**.
2. **Native Resolution & Color Space:** All 500 pairs are original **$512 \times 512$ PNG images in 3-channel 8-bit RGB** ($24\text{ bits/pixel}$).
3. **Data Split Verification:** The fixed split in `results/data_split.json` (Seed 42) partitions the data into **350 training pairs (70%)**, **75 validation pairs (15%)**, and **75 held-out test pairs (15%)** with zero set overlap or contamination.
4. **Scientific Demarcation:** The paired RGB dataset is **strictly distinct** from the genuine Sentinel-2 Level-2A agricultural scene (`dataset/sentinal2/pulivendula_scene_01/`), which contains real clouds and shadows with no paired cloud-free ground truth.

---

## 2. Quantitative Dataset Inventory

```
cloud-removal-poc/dataset/
├── clean/               # 500 paired ground-truth cloud-free satellite scenes (PNG, 512×512)
├── cloudy/              # 500 paired cloud-obscured satellite scenes (PNG, 512×512)
└── sentinal2/           # [SEPARATE] Real-world Sentinel-2C Level-2A agricultural scene
    └── pulivendula_scene_01/
        ├── B02.jp2 (Blue, 10m)
        ├── B03.jp2 (Green, 10m)
        ├── B04.jp2 (Red, 10m)
        ├── B08.jp2 (NIR, 10m)
        ├── B11.jp2 (SWIR-1, 20m)
        └── SCL.jp2 (Scene Classification Layer, 20m)
```

### Statistical Verification:
- **Total Paired Files:** 500 cloudy + 500 clean = 1,000 files.
- **Missing Pairs:** 0 (Both directories contain identical filename sets).
- **Corrupted Images:** 0 (All files passed `PIL.Image.verify()` without exceptions).
- **Native Dimensions:** $512 \times 512$ pixels across all 1,000 images.
- **Dynamic Range:** Unnormalized raw integer pixel range $[0, 255]$; normalized float tensor range $[0.0, 1.0]$.

---

## 3. Data Split Audit & Data Leakage Assessment

The train/validation/test split is defined in `results/data_split.json` generated using `create_data_split.py` (fixed random seed $42$):

| Split Subset | Count | Percentage | Purpose | Leakage Status |
|---|---|---|---|---|
| **Training Set** | 350 pairs | 70.0% | Model parameter gradient optimization | Strictly disjoint |
| **Validation Set** | 75 pairs | 15.0% | Checkpoint selection, learning rate decay, early stopping | Strictly disjoint |
| **Held-Out Test Set** | 75 pairs | 15.0% | Final benchmark evaluation (PSNR, SSIM, L1) | Strictly disjoint |
| **Total** | **500 pairs** | **100.0%** | Comprehensive research pool | Zero overlap |

### Set Disjoint Verification:
$$\text{Train} \cap \text{Validation} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Validation} \cap \text{Test} = \emptyset$$
$$\text{Train} \cup \text{Validation} \cup \text{Test} = \{0, 1, \dots, 499\} \quad (|S| = 500)$$

### Data Leakage Safeguards:
1. **No Shared Augmentation:** Validation and test splits never undergo stochastic augmentations.
2. **Fixed Normalization:** Transformations use deterministic min-max scaling ($x / 255.0 \to [0.0, 1.0]$) rather than batch-dependent or running dataset statistics.
3. **No Test-Set Peeking:** Test metrics are only computed during final evaluation using `evaluate_test_set.py`.

---

## 4. Scientific Demarcation: Training Dataset vs. Sentinel-2 Agricultural Scene

A vital scientific principle in remote sensing research is distinguishing between **supervised paired benchmark datasets** and **unpaired real-world multispectral satellite acquisitions**:

| Dimension | Dataset A: Paired RGB Training Dataset | Dataset B: Sentinel-2C Agricultural Scene |
|---|---|---|
| **Location** | `dataset/cloudy/` and `dataset/clean/` | `dataset/sentinal2/pulivendula_scene_01/` |
| **Role in GenVision** | Model training, validation, and held-out quantitative benchmark evaluation. | Real-world agricultural deployment, vegetation indices (NDVI/NDWI/NDMI), and SCL cloud masking demonstration. |
| **Bands Available** | 3 RGB channels ($B04, B03, B02$ optical rendering). | 6 multispectral bands ($B02, B03, B04, B08, B11, SCL$). |
| **Paired Ground Truth** | **Yes** (Every cloudy image has an exact clean reference image). | **No** (Single temporal snapshot on 14 October 2026; no simultaneous cloud-free reference exists). |
| **Cloud Characteristics** | Mixed synthetic/semi-synthetic clouds with controlled spatial distribution. | Genuine atmospheric clouds ($30.40\%$) and topographic cast shadows ($13.09\%$). |
| **Supervised Training** | Supervised pair loss: $\mathcal{L}_1(\hat{\mathbf{Y}}, \mathbf{Y}_{\text{clean}})$. | Unsupervised / Mask-filtered inference. Cannot be trained with paired pixel-wise L1 loss. |

---

## 5. Suitability of the 350-Pair Training Set for Transformer Training

### Challenge: Sample Efficiency in Vision Transformers
Standard Vision Transformers (ViT) lack 2D translational equivariance and local spatial inductive biases. Pure ViT architectures typically require $10^5$ to $10^7$ training images to learn basic edge and texture filters from scratch without catastrophic overfitting.

With an available training budget of **350 image pairs ($N=350$)**:
- **Pure ViT (e.g., ViT-B/16):** Severe risk of overfitting; memorization of cloud positions rather than learning atmospheric de-scattering.
- **Heavy Hybrid (e.g., TransUNet with ResNet-50 stem, 105M params):** Parameter-to-data ratio is $\sim 300,000 : 1$, which is severely overparameterized for 350 scenes.

### Solution: Designing CS-TransUNet for Small-N Remote Sensing Regimes
CS-TransUNet resolves the sample-efficiency bottleneck through specific architectural constraints:
1. **Convolutional Inductive Bias in Early Stages:** Convolutions in Stages 1 and 2 ($256 \times 256$ and $128 \times 128$) constrain spatial feature extraction to localized $3 \times 3$ receptive fields, eliminating the need to learn 2D geometry from scratch.
2. **Compact Parameter Budget (~7.8M parameters):** Parameters are right-sized to match the information capacity of 350 training scenes, directly preventing overparameterization.
3. **Multi-Dconv Transposed Channel Attention:** Rather than spatial all-to-all attention ($N \times N$), attention is computed across feature channels ($C \times C$). This prevents spatial overfitting and forces the network to learn spectral cross-channel correlations.
4. **Global Residual Learning ($\Delta \mathbf{X}$):** By learning to predict only the additive/subtractive difference between cloudy and clean pixels, the target function space is constrained, enabling fast generalization.

---

## 6. Preprocessing & Data Pipeline Specification

To prepare batches for CS-TransUNet training without modifying the existing codebase:
- **Input Resolution:** Resized from $512 \times 512 \to 256 \times 256$ using bilinear interpolation (preserving consistency with U-Net and Attention U-Net baselines).
- **Pixel Tensor Scaling:** Normalized to $[0.0, 1.0]$.
- **Data Augmentation Protocol for Training (Offline / On-the-Fly):**
  - Random Horizontal Flip ($p = 0.5$)
  - Random Vertical Flip ($p = 0.5$)
  - Random $90^\circ$ Orthogonal Rotation ($p = 0.5$)
  - *Constraint:* Photometric jitter is **omitted** to preserve radiometric reflectance fidelity for subsequent agricultural index calculations.
- **PyTorch DataLoader Settings:**
  - Batch size: $2$ (or $4$ with gradient accumulation).
  - Shuffle: `True` for training with fixed seed generator; `False` for validation and test.
  - Pin Memory: `True` (accelerates host-to-tensor transfers on 32GB RAM).
  - Worker Threads: `0` or `2` on Windows to avoid inter-process multiprocessing overhead.
