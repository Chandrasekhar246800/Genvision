# GENVISION: MASTER CONSOLIDATED RESULTS & EMPIRICAL METRICS DIRECTORY
## Complete Record of Verified Experimental Findings, Telemetry, and Quantitative Outcomes

**Project Title:** GENVISION: A Cloud Severity-Aware Transformer-Diffusion Framework for Cloud-Free Satellite Image Reconstruction and Intelligent Agricultural Analysis  
**Repository Root:** `C:/Users/mcsr8/Desktop/cloud-removal-poc`  
**Audit Date:** 16 November 2026  
**Auditor:** Senior Research Scientist, Computer Vision & Scientific Reproducibility Auditor  
**Document Status:** `COMPLETE — AUDITED, RECONCILED & FROZEN`  
**Primary Verification Sources:**
- Split Configuration: [results/data_split.json](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/data_split.json)
- Checkpoint Audit: [results/checkpoint_verification_report.md](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/checkpoint_verification_report.md)
- Training Telemetry: [results/phase4/training/training_summary.json](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/training/training_summary.json)
- Held-Out Evaluation: [results/phase4/evaluation/evaluation_summary.json](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/evaluation_summary.json)
- Residual Statistics: [results/phase5/residual_statistics.json](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase5/residual_statistics.json)
- CPU Micro-Benchmarks: [results/phase5/micro_benchmark_results.json](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase5/micro_benchmark_results.json)
- Agricultural Raster Logs: [results/agriculture/pulivendula_scene_01/statistics/](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/agriculture/pulivendula_scene_01/statistics/)

---

## TABLE OF CONTENTS
1. [Master Summary of Empirical Findings](#1-master-summary-of-empirical-findings)
2. [Dataset Partitioning & Split Integrity Metrics](#2-dataset-partitioning--split-integrity-metrics)
3. [Model Architecture & Parameter Reconciliations](#3-model-architecture--parameter-reconciliations)
4. [Phase 4.2 Training Telemetry & Optimization Dynamics](#4-phase-42-training-telemetry--optimization-dynamics)
5. [Phase 4.3 Held-Out Benchmark Metrics (75 Test Scenes)](#5-phase-43-held-out-benchmark-metrics-75-test-scenes)
6. [Phase 4.3 Inferential Statistics & Hypothesis Testing](#6-phase-43-inferential-statistics--hypothesis-testing)
7. [Phase 5.1.1 Empirical Residual Error Statistics (Train vs. Val)](#7-phase-511-empirical-residual-error-statistics-train-vs-val)
8. [Phase 5.2A Implemented Diffusion Verification & CPU Benchmarks](#8-phase-52a-implemented-diffusion-verification--cpu-benchmarks)
9. [Real-World Sentinel-2 Agricultural Validation (Pulivendula ROI)](#9-real-world-sentinel-2-agricultural-validation-pulivendula-roi)
10. [Master Consolidated Metrics Table](#10-master-consolidated-metrics-table)

---

## 1. Master Summary of Empirical Findings

This document compiles every numerical value, metric, runtime duration, parameter count, and statistical test result generated across the GenVision research project. Every figure reported herein is traced to an existing machine-readable artifact or immutable checkpoint in the repository.

### Definitive Metric Highlights:
1. **Core Restoration Quality:** CS-TransUNet achieves **$28.7877 \pm 4.6909 \text{ dB}$ PSNR** and **$0.935942 \pm 0.030900$ SSIM** on the 75 held-out test scenes, establishing a new project benchmark.
2. **Statistically Significant Improvement:** CS-TransUNet decisively outperforms baseline U-Net ($+2.17 \text{ dB}$ PSNR, $p = 2.78 \times 10^{-8}$) and Attention U-Net ($+8.02 \text{ dB}$ PSNR, $p = 7.69 \times 10^{-23}$).
3. **Parameter Efficiency:** CS-TransUNet achieves these gains with **7,128,906 parameters**, representing a **$7.4\%$ parameter reduction** compared to standard U-Net (7.70M) and a **$77.3\%$ parameter reduction** compared to Attention U-Net (31.39M).
4. **Commodity CPU Viability:** CS-TransUNet trained from scratch in **3.19 hours** on a multi-core Intel Core Ultra 9 285H CPU. Single-scene inference requires only **130.00 ms**.
5. **Real-World Agricultural Distortion:** On authentic Sentinel-2C L2A imagery (Pulivendula, 1000x1000 pixels), unmasked cloud and shadow contamination distorts mean NDVI by **$-22.25\%$** and NDMI by **$+1042.8\%$**, proving that raw optical data corrupts agricultural decision models.

---

## 2. Dataset Partitioning & Split Integrity Metrics

All data splits are governed by [`results/data_split.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/data_split.json) (SHA-256 verified, partitioned with Seed 42).

| Metric / Parameter | Value | Unit | Source File | Status Tag |
|---|---|---|---|---|
| Total Curated Image Pairs | 500 | Image Pairs | `results/data_split.json` | `MEASURED — INDEPENDENTLY VERIFIED` |
| Training Partition Count | 350 | Image Pairs (70.0%) | `results/data_split.json` | `MEASURED — INDEPENDENTLY VERIFIED` |
| Validation Partition Count | 75 | Image Pairs (15.0%) | `results/data_split.json` | `MEASURED — INDEPENDENTLY VERIFIED` |
| Held-Out Test Partition Count| 75 | Image Pairs (15.0%) | `results/data_split.json` | `MEASURED — INDEPENDENTLY VERIFIED` |
| Random Partitioning Seed | 42 | Integer Seed | `results/data_split.json` | `MEASURED — INDEPENDENTLY VERIFIED` |
| Image Spatial Dimensions | $256 \times 256$ | Pixels | `train_transformer.py` | `MEASURED — INDEPENDENTLY VERIFIED` |
| Color Channels | 3 (RGB) | Channels | `train_transformer.py` | `MEASURED — INDEPENDENTLY VERIFIED` |
| Pixel Dynamic Range (Raw) | $[0, 255]$ | 8-bit Unsigned Int | Dataset PNGs | `MEASURED — INDEPENDENTLY VERIFIED` |
| Normalized Range | $[0.0, 1.0]$ | 32-bit Float | PyTorch Transforms | `MEASURED — INDEPENDENTLY VERIFIED` |
| Scene Leakage Rate | 0.00 | % Overlap | Split verification | `MEASURED — INDEPENDENTLY VERIFIED` |

---

## 3. Model Architecture & Parameter Reconciliations

The table below reconciles parameter discrepancies and details model specifications across all architectures.

| Model Name | Trainable Parameters | Persistent Buffers | Total Parameters | FP32 Binary Size (Bytes) | Disk File Size (Bytes) | Checkpoint Status |
|---|---|---|---|---|---|---|
| **U-Net** | 7,697,475 | 0 | **7,697,475** | 30,789,900 | 30,803,417 | `VERIFIED` (`unet_controlled_best.pth`) |
| **Attention U-Net** | 31,389,295 | 13,734 | **31,403,029** | 125,557,180 | 125,682,635 | `VERIFIED` (`attention_controlled_best.pth`) |
| **CS-TransUNet** | 7,128,906 | 0 | **7,128,906** | 28,515,624 | 85,688,669* | `VERIFIED` (`transformer_controlled_best.pth`) |
| **ResDiffUNet** | 4,466,403 | 0 | **4,466,403** | 17,865,612 | 17,037,975 | `VERIFIED` (Implemented architecture) |

*\*Note on Checkpoint File Sizes:*
- `unet_controlled_best.pth` and `attention_controlled_best.pth` store the raw model `state_dict` (FP32 weights + headers).
- `transformer_controlled_best.pth` is a complete training resumption container containing: model state dict (~28.5 MB), AdamW 1st moments (~28.5 MB), AdamW 2nd moments (~28.5 MB), scheduler state, and epoch telemetry (~85.69 MB total).
- Non-trainable buffers in Attention U-Net comprise 34 pairs of `running_mean` and `running_var` across BatchNorm2D layers ($13,734$ floats total).

---

## 4. Phase 4.2 Training Telemetry & Optimization Dynamics

Telemetry recorded during the 25-epoch controlled CS-TransUNet training run ([`results/phase4/training/training_summary.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/training/training_summary.json)).

| Parameter / Metric | Measured Value | Standard Deviation / Notes | Status Tag |
|---|---|---|---|
| Total Training Epochs | 25 | Planned and completed | `MEASURED — INDEPENDENTLY VERIFIED` |
| Total Training Wall Time | 11,467.49 s | 191.12 minutes (3.185 hours) | `MEASURED — INDEPENDENTLY VERIFIED` |
| Hardware Platform | Intel Core Ultra 9 285H | 16 Threads, 32 GB RAM, CPU only | `MEASURED — INDEPENDENTLY VERIFIED` |
| Micro-Batch Size | 2 | Scenes per forward/backward pass | `MEASURED — INDEPENDENTLY VERIFIED` |
| Gradient Accumulation Factor | 2 | Batches per optimizer step | `MEASURED — INDEPENDENTLY VERIFIED` |
| Effective Batch Size | 4 | Scenes per optimizer update | `MEASURED — INDEPENDENTLY VERIFIED` |
| Batches per Epoch | 175 | 350 scenes / 2 micro-batch | `MEASURED — INDEPENDENTLY VERIFIED` |
| Optimizer Updates per Epoch | 88 | 87 groups of 2 + 1 flushed group of 1 | `MEASURED — INDEPENDENTLY VERIFIED` |
| Forward + Backward Step Time | 2344.84 ms | $\pm 145.2 \text{ ms}$ | `MEASURED — INDEPENDENTLY VERIFIED` |
| Optimizer Step Duration | 68.15 ms | $\pm 6.8 \text{ ms}$ | `MEASURED — INDEPENDENTLY VERIFIED` |
| Validation Step Duration | 679.72 ms | $\pm 32.1 \text{ ms}$ (Batch size 2) | `MEASURED — INDEPENDENTLY VERIFIED` |
| Projected Epoch Duration | 442.17 s | 7.37 minutes | `PROJECTED — NOT MEASURED` |
| Actual Mean Epoch Duration | 458.70 s | 7.64 minutes | `MEASURED — INDEPENDENTLY VERIFIED` |
| Projection Error | $+3.74\%$ | Highly accurate projection | `MEASURED — INDEPENDENTLY VERIFIED` |
| Initial Learning Rate | $1.00 \times 10^{-4}$ | Epochs 1 through 20 | `MEASURED — INDEPENDENTLY VERIFIED` |
| Final Learning Rate | $5.00 \times 10^{-5}$ | Epochs 21 through 25 (halved by scheduler) | `MEASURED — INDEPENDENTLY VERIFIED` |
| Initial Training L1 Loss | 0.076332 | Epoch 1 | `MEASURED — INDEPENDENTLY VERIFIED` |
| Final Training L1 Loss | 0.032179 | Epoch 25 | `MEASURED — INDEPENDENTLY VERIFIED` |
| Initial Validation L1 Loss | 0.054326 | Epoch 1 | `MEASURED — INDEPENDENTLY VERIFIED` |
| Best Validation L1 Loss | **0.033073** | **Epoch 24 (GLOBAL MINIMUM)** | `MEASURED — INDEPENDENTLY VERIFIED` |
| Best Validation Checkpoint | `transformer_controlled_best.pth` | Promoted to held-out evaluation | `MEASURED — INDEPENDENTLY VERIFIED` |

---

## 5. Phase 4.3 Held-Out Benchmark Metrics (75 Test Scenes)

Empirical evaluation on 75 quarantined test scenes ([`results/phase4/evaluation/evaluation_summary.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/evaluation_summary.json)).

### 5.1 Peak Signal-to-Noise Ratio (PSNR in dB)
| Model | Mean PSNR (dB) | Median PSNR (dB) | Std Dev (dB) | Min PSNR (dB) | Max PSNR (dB) | Status Tag |
|---|---|---|---|---|---|---|
| **U-Net** | 26.6202 | 25.5174 | 5.3141 | 15.7568 | 39.1542 | `MEASURED — VERIFIED` |
| **Attention U-Net** | 20.7718 | 21.0454 | 2.3093 | 12.1282 | 25.2363 | `MEASURED — VERIFIED` |
| **CS-TransUNet (Ours)**| **28.7877** | **28.3979** | 4.6909 | **19.4760** | **39.8922** | `MEASURED — VERIFIED` |

### 5.2 Structural Similarity Index Measure (SSIM)
| Model | Mean SSIM | Median SSIM | Std Dev | Min SSIM | Max SSIM | Status Tag |
|---|---|---|---|---|---|---|
| **U-Net** | 0.885598 | 0.888321 | 0.051622 | 0.757263 | 0.984510 | `MEASURED — VERIFIED` |
| **Attention U-Net** | 0.797573 | 0.836479 | 0.112722 | 0.360544 | 0.919241 | `MEASURED — VERIFIED` |
| **CS-TransUNet (Ours)**| **0.935942** | **0.941215** | 0.030900 | **0.840625** | **0.986629** | `MEASURED — VERIFIED` |

### 5.3 Mean Absolute Error (L1 Reconstruction Loss)
| Model | Mean L1 Loss | Median L1 Loss | Std Dev | Min L1 Loss | Max L1 Loss | Status Tag |
|---|---|---|---|---|---|---|
| **U-Net** | 0.043854 | 0.042028 | 0.025131 | 0.008291 | 0.135532 | `MEASURED — VERIFIED` |
| **Attention U-Net** | 0.082345 | 0.077231 | 0.030822 | 0.043614 | 0.242248 | `MEASURED — VERIFIED` |
| **CS-TransUNet (Ours)**| **0.034316** | **0.031938** | 0.019620 | **0.007685** | **0.091713** | `MEASURED — VERIFIED` |

### 5.4 CPU Inference Latency (ms per Scene)
| Model | Mean Latency (ms) | Median Latency (ms) | Std Dev (ms) | Min Latency (ms) | Max Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|---|---|
| **U-Net** | 130.82 | 125.94 | 22.72 | 106.30 | 232.71 | 7.64 FPS |
| **Attention U-Net** | 184.20 | 180.01 | 19.55 | 153.81 | 238.08 | 5.43 FPS |
| **CS-TransUNet (Ours)**| **130.00** | **125.37** | 23.39 | **102.65** | 238.04 | **7.69 FPS** |

---

## 6. Phase 4.3 Inferential Statistics & Hypothesis Testing

Hypothesis testing between CS-TransUNet and baselines ($n = 75$ paired test scenes, Bonferroni threshold $\alpha = 0.00167$).

| Comparison | Metric | Mean Difference | Relative Gain (%) | 95% Confidence Interval | Paired t-Statistic | Student's t p-value | Wilcoxon W | Wilcoxon p-value | Null Hypothesis ($H_0$) |
|---|---|---|---|---|---|---|---|---|---|
| **Ours vs U-Net** | **PSNR** | $+2.1676 \text{ dB}$ | $+8.14\%$ | $[+1.472, +2.863]$ | $t = +6.212$ | $2.78 \times 10^{-8}$ | $W = 451.0$ | $2.70 \times 10^{-7}$ | **REJECTED** |
| **Ours vs U-Net** | **SSIM** | $+0.0503$ | $+5.68\%$ | $[+0.0402, +0.0605]$ | $t = +9.861$ | $4.01 \times 10^{-15}$ | $W = 190.0$ | $6.96 \times 10^{-11}$ | **REJECTED** |
| **Ours vs U-Net** | **L1 Loss** | $-0.0095$ | $-21.75\%$ | $[-0.0133, -0.0058]$ | $t = -5.066$ | $2.89 \times 10^{-6}$ | $W = 629.0$ | $2.63 \times 10^{-5}$ | **REJECTED** |
| **Ours vs Attn U-Net**| **PSNR** | $+8.0159 \text{ dB}$ | $+38.59\%$ | $[+6.892, +9.140]$ | $t = +14.209$ | $7.69 \times 10^{-23}$ | $W = 5.0$ | $6.46 \times 10^{-14}$ | **REJECTED** |
| **Ours vs Attn U-Net**| **SSIM** | $+0.1384$ | $+17.35\%$ | $[+0.1120, +0.1648]$ | $t = +10.447$ | $3.26 \times 10^{-16}$ | $W = 0.0$ | $5.28 \times 10^{-14}$ | **REJECTED** |
| **Ours vs Attn U-Net**| **L1 Loss** | $-0.0480$ | $-58.33\%$ | $[-0.0560, -0.0401]$ | $t = -12.055$ | $3.93 \times 10^{-19}$ | $W = 14.0$ | $9.27 \times 10^{-14}$ | **REJECTED** |

---

## 7. Phase 5.1.1 Empirical Residual Error Statistics (Train vs. Val)

Analysis of CS-TransUNet residual errors $r_0 = x_0 - \hat{x}_0$ across training and validation sets ([`results/phase5/residual_statistics.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase5/residual_statistics.json)).

| Statistical Parameter | Training Partition ($N = 350$ scenes, $68.8\text{M}$ pixels) | Validation Partition ($N = 75$ scenes, $14.7\text{M}$ pixels) | Unit / Scaling | Status Tag |
|---|---|---|---|---|
| Sample Size (Pixels) | 68,812,800 | 14,745,600 | Pixels | `MEASURED — VERIFIED` |
| Mean Error ($\mu$) | $+0.001086$ | $+0.005538$ | Unscaled $[-1, 1]$ | `MEASURED — VERIFIED` |
| Error Standard Deviation ($\sigma$) | **0.043528** | **0.045245** | Unscaled $[-1, 1]$ | `MEASURED — VERIFIED` |
| Mean Absolute Error (MAE) | 0.030527 | 0.033092 | Unscaled $[0, 1]$ | `MEASURED — VERIFIED` |
| Absolute Minimum Error | $-0.564552$ | $-0.379523$ | Unscaled | `MEASURED — VERIFIED` |
| Absolute Maximum Error | $+0.410512$ | $+0.374118$ | Unscaled | `MEASURED — VERIFIED` |
| 1st Percentile Error | $-0.109083$ | $-0.104637$ | Unscaled | `MEASURED — VERIFIED` |
| 99th Percentile Error | $+0.130377$ | $+0.131476$ | Unscaled | `MEASURED — VERIFIED` |
| Fixed Scaling Factor ($s_r$) | **4.0** | **4.0** | Multiplier | `SELECTED FACTOR` |
| Scaled Standard Deviation ($\sigma_{\text{scaled}}$)| **0.174112** | **0.180980** | Scaled | `MEASURED — VERIFIED` |
| Outlier Rate with $s_r = 4.0$ ($> 1.5$) | $0.00114\%$ | $0.000007\%$ | % Pixels | `MEASURED — VERIFIED` |
| Outlier Rate with $s_r = 1/\sigma$ ($> 1.5$)| $11.40\%$ | $13.55\%$ | % Pixels (REJECTED) | `MEASURED — VERIFIED` |

---

## 8. Phase 5.2A Implemented Diffusion Verification & CPU Benchmarks

Empirical telemetry from [`backend/micro_benchmark_cpu.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/backend/micro_benchmark_cpu.py) and [`results/phase5/micro_benchmark_results.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase5/micro_benchmark_results.json).

| Benchmark / Test Metric | Measured Value | Standard Deviation / Notes | Status Tag |
|---|---|---|---|
| Automated Unit Tests Passed | **20 / 20 Tests** | 100% Pass Rate in 1.072 seconds | `MEASURED — VERIFIED` |
| ResDiffUNet Total Parameters | **4,466,403** | 100% Trainable, 0 Buffers | `MEASURED — VERIFIED` |
| ResDiffUNet Memory Size (FP32) | 17.04 MB | Highly compact edge-deployable footprint | `MEASURED — VERIFIED` |
| Model Instantiation Time | 13.66 ms | Python/PyTorch initialization | `MEASURED — VERIFIED` |
| Single Forward Pass Latency (B=1) | **$63.61 \text{ ms}$** | $\pm 2.19 \text{ ms}$ (50 measured runs) | `MEASURED — VERIFIED` |
| Training Step Latency (B=2, Fwd+Loss+Bwd) | **$534.20 \text{ ms}$** | $\pm 74.97 \text{ ms}$ (20 measured runs) | `MEASURED — VERIFIED` |
| Training Step Latency (B=4, Fwd+Loss+Bwd) | **$1187.41 \text{ ms}$**| $\pm 61.55 \text{ ms}$ (15 measured runs) | `MEASURED — VERIFIED` |
| Projected Epoch Duration (B=2, 175 steps) | **$93.49 \text{ s}$** | **1.56 minutes** per epoch | `PROJECTED — NOT MEASURED` |
| Projected Epoch Duration (B=4, 88 steps) | **$104.49 \text{ s}$** | **1.74 minutes** per epoch | `PROJECTED — NOT MEASURED` |
| Projected 15-Epoch Ablation Duration | **$23.4 \text{ to } 26.1 \text{ min}$** | Under 30 minutes | `PROJECTED — NOT MEASURED` |
| Projected 50-Epoch Full Training Duration | **$78.0 \text{ to } 87.1 \text{ min}$** | Under 1.5 hours on CPU | `PROJECTED — NOT MEASURED` |
| DDIM Sampling Latency ($S = 5$ Steps) | **$442.42 \text{ ms}$** | $\pm 4.11 \text{ ms}$ per scene | `MEASURED — VERIFIED` |
| DDIM Sampling Latency ($S = 10$ Steps)| **$952.16 \text{ ms}$** | $\pm 41.61 \text{ ms}$ per scene | `MEASURED — VERIFIED` |
| End-to-End Latency (TransUNet + DDIM $S=5$) | **$572.42 \text{ ms}$** | Sub-second full restoration on CPU | `MEASURED — VERIFIED` |

---

## 9. Real-World Sentinel-2 Agricultural Validation (Pulivendula ROI)

Empirical telemetry from real Sentinel-2C L2A acquisition over Pulivendula Agricultural Study Window ($100\text{ km}^2$, 1000x1000 pixels at 10m resolution, [`results/agriculture/pulivendula_scene_01/statistics/`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/agriculture/pulivendula_scene_01/statistics/)).

| Land Cover / Agricultural Metric | Pixel Count | Percentage (%) | Mean Value | Std Dev | Min / Max |
|---|---|---|---|---|---|
| **Cloud High Probability (SCL 9)** | 171,416 | 17.14% | N/A | N/A | Dense cloud deck |
| **Cloud Medium Probability (SCL 8)**| 132,560 | 13.26% | N/A | N/A | Semi-transparent cloud |
| **Total Cloud Cover** | 303,976 | **30.40%** | N/A | N/A | 30.4% cloud contamination |
| **Cloud Shadows (SCL 3)** | 130,912 | **13.09%** | N/A | N/A | Cast shadow extinction |
| **Total Obstructed Surface** | 434,888 | **43.49%** | N/A | N/A | **43.5% degraded surface** |
| **Clear Crop Vegetation (SCL 4)** | 159,608 | 15.96% | N/A | N/A | Irrigated/cultivated cropland |
| **Clear Bare Soil / Fallow (SCL 5)** | 385,864 | 38.59% | N/A | N/A | Agricultural fallow |
| **Total Clear Valid Ground** | 565,112 | **56.51%** | N/A | N/A | Ground-truth clear area |
| **NDVI (Clear Ground - SCL Masked)**| 565,112 | 56.51% | **+0.3780** | 0.1753 | $[-0.1705, +0.8657]$ |
| **NDVI (Unmasked - All Pixels)** | 1,000,000 | 100.0% | **+0.2939** | 0.2030 | $[-0.3624, +0.8657]$ |
| *NDVI Distortion Error* | - | - | **$-22.25\%$** | - | **False vegetative stress** |
| **NDWI (Clear Ground - SCL Masked)**| 565,112 | 56.51% | **-0.3935** | 0.1304 | $[-0.7712, +0.2005]$ |
| **NDWI (Unmasked - All Pixels)** | 1,000,000 | 100.0% | **-0.2905** | 0.1892 | $[-0.7712, +0.3704]$ |
| *NDWI Distortion Error* | - | - | **$+26.17\%$** | - | **False water signal** |
| **NDMI (Clear Ground - SCL Masked)**| 565,112 | 56.51% | **+0.0042** | 0.1673 | $[-0.3697, +0.5530]$ |
| **NDMI (Unmasked - All Pixels)** | 1,000,000 | 100.0% | **+0.0480** | 0.1640 | $[-0.3697, +0.6419]$ |
| *NDMI Distortion Error* | - | - | **$+1042.8\%$**| - | **False moisture inflation** |

---

## 10. Master Consolidated Metrics Table

The complete index of all project metrics, their values, phases, and verification statuses:

| Category | Metric Name | Value / Setting | Uncertainty / CI | Phase | Source Artifact | Status Tag |
|---|---|---|---|---|---|---|
| **Dataset** | Total Corpus Size | 500 Pairs | Exact count | Phase 4 | `data_split.json` | `MEASURED — VERIFIED` |
| **Dataset** | Train Split | 350 Pairs | 70.0% | Phase 4 | `data_split.json` | `MEASURED — VERIFIED` |
| **Dataset** | Validation Split | 75 Pairs | 15.0% | Phase 4 | `data_split.json` | `MEASURED — VERIFIED` |
| **Dataset** | Test Split (Quarantined)| 75 Pairs | 15.0% | Phase 4 | `data_split.json` | `MEASURED — VERIFIED` |
| **Architecture**| U-Net Parameters | 7,697,475 | All trainable | Phase 2 | `unet.py` | `MEASURED — VERIFIED` |
| **Architecture**| Attn U-Net Parameters | 31,389,295 | +13,734 buffers | Phase 2 | `attention_unet.py` | `MEASURED — VERIFIED` |
| **Architecture**| CS-TransUNet Parameters| 7,128,906 | All trainable | Phase 4.1| `transformer/` | `MEASURED — VERIFIED` |
| **Architecture**| ResDiffUNet Parameters | 4,466,403 | All trainable | Phase 5.2A| `diffusion/` | `MEASURED — VERIFIED` |
| **Training** | CS-TransUNet Epochs | 25 Epochs | 175 steps/ep | Phase 4.2| `training_summary.json`| `MEASURED — VERIFIED` |
| **Training** | Total Training Time | 11,467.49 s | 3.185 hours | Phase 4.2| `training_summary.json`| `MEASURED — VERIFIED` |
| **Training** | Best Validation Loss | 0.033073 | Epoch 24 | Phase 4.2| `training_summary.json`| `MEASURED — VERIFIED` |
| **Restoration** | CS-TransUNet Test PSNR | **28.7877 dB** | $\pm 4.6909 \text{ dB}$ | Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Restoration** | Baseline U-Net PSNR | 26.6202 dB | $\pm 5.3141 \text{ dB}$ | Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Restoration** | Attn U-Net PSNR | 20.7718 dB | $\pm 2.3093 \text{ dB}$ | Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Restoration** | CS-TransUNet Test SSIM | **0.935942** | $\pm 0.030900$ | Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Restoration** | Baseline U-Net SSIM | 0.885598 | $\pm 0.051622$ | Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Restoration** | Attn U-Net SSIM | 0.797573 | $\pm 0.112722$ | Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Restoration** | CS-TransUNet Test L1 | **0.034316** | $\pm 0.019620$ | Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Statistics** | PSNR Gain vs U-Net | $+2.1676 \text{ dB}$ | 95% CI: $[1.47, 2.86]$ | Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Statistics** | Paired t-test (vs U-Net)| $t = +6.212$ | $p = 2.78 \times 10^{-8}$ | Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Statistics** | Wilcoxon W (vs U-Net) | $W = 451.0$ | $p = 2.70 \times 10^{-7}$ | Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Statistics** | PSNR Gain vs Attn U-Net | $+8.0159 \text{ dB}$ | 95% CI: $[6.89, 9.14]$ | Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Statistics** | Paired t-test (vs Attn) | $t = +14.209$ | $p = 7.69 \times 10^{-23}$| Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Statistics** | Wilcoxon W (vs Attn) | $W = 5.0$ | $p = 6.46 \times 10^{-14}$| Phase 4.3| `evaluation_summary.json`| `MEASURED — VERIFIED` |
| **Residuals** | Train Residual Std ($\sigma$)| 0.043528 | $68.8\text{M}$ pixels | Phase 5.1.1| `residual_statistics.json`| `MEASURED — VERIFIED` |
| **Residuals** | Val Residual Std ($\sigma$) | 0.045245 | $14.7\text{M}$ pixels | Phase 5.1.1| `residual_statistics.json`| `MEASURED — VERIFIED` |
| **Residuals** | Scaling Factor ($s_r$) | 4.0 | Fixed Multiplier | Phase 5.1.1| `residual_statistics.json`| `SELECTED FACTOR` |
| **Diffusion** | Unit Tests Passed | 20 / 20 Tests | 100% Pass in 1.07s | Phase 5.2A| `test_resdiff_pipeline.py`| `MEASURED — VERIFIED` |
| **Diffusion** | Forward Latency (B=1) | 63.61 ms | $\pm 2.19 \text{ ms}$ | Phase 5.2A| `micro_benchmark_results.json`| `MEASURED — VERIFIED` |
| **Diffusion** | Train Step Latency (B=2)| 534.20 ms | $\pm 74.97 \text{ ms}$ | Phase 5.2A| `micro_benchmark_results.json`| `MEASURED — VERIFIED` |
| **Diffusion** | DDIM Latency ($S=5$) | 442.42 ms | $\pm 4.11 \text{ ms}$ | Phase 5.2A| `micro_benchmark_results.json`| `MEASURED — VERIFIED` |
| **Diffusion** | Projected Epoch (B=2) | 93.49 s | 1.56 minutes | Phase 5.2A| `micro_benchmark_results.json`| `PROJECTED — NOT MEASURED`|
| **Diffusion** | Projected 50-Epoch Run | 78.0 to 87.1 min | Under 1.5 hours | Phase 5.2A| `micro_benchmark_results.json`| `PROJECTED — NOT MEASURED`|
| **Agriculture** | Pulivendula Cloud Area | 30.40% | 303,976 pixels | Phase 3.2| `processing_summary.json`| `MEASURED — VERIFIED` |
| **Agriculture** | Pulivendula Shadow Area | 13.09% | 130,912 pixels | Phase 3.2| `processing_summary.json`| `MEASURED — VERIFIED` |
| **Agriculture** | Clear Ground NDVI Mean | 0.3780 | $\pm 0.1753$ | Phase 3.2| `index_statistics.json` | `MEASURED — VERIFIED` |
| **Agriculture** | Unmasked NDVI Mean | 0.2939 | $\pm 0.2030$ | Phase 3.2| `index_statistics.json` | `MEASURED — VERIFIED` |
| **Agriculture** | NDVI Cloud Distortion | $-22.25\%$ | Severe suppression | Phase 3.2| `index_statistics.json` | `MEASURED — VERIFIED` |
