# GenVision Phase 4 — Controlled Experiment Plan: Transformer vs. Baselines

**Project Title:** GENVISION: A CLOUD SEVERITY-AWARE TRANSFORMER-DIFFUSION FRAMEWORK FOR CLOUD-FREE SATELLITE IMAGE RECONSTRUCTION AND INTELLIGENT AGRICULTURAL ANALYSIS  
**Document ID:** `results/phase4/PHASE4_EXPERIMENT_PLAN.md`  
**Phase:** 4 — Experiment Design & Benchmarking Protocol  
**Date:** 16 November 2026  
**Status:** **EXPERIMENTAL PROTOCOL SPECIFIED**  
**Authors:** Senior AI Research Engineer, Computer Vision Architect & Remote Sensing Scientist  

---

## 1. Experimental Overview & Objectives

The primary research objective of Phase 4 is to rigorously evaluate whether the proposed **Cloud Severity-Aware Hybrid Transformer (CS-TransUNet)** outperforms the existing convolutional baselines in satellite image cloud removal while maintaining strict computational efficiency on CPU hardware.

### Established Experimental Baselines (Held-Out Test Set, $N=75$ scenes):
1. **Standard U-Net:**
   - Parameters: 7,697,475 (~7.7M)
   - PSNR: **26.6202 dB**
   - SSIM: **0.8856**
   - L1 Loss: **0.043854**
   - Mean CPU Inference Time: **140.60 ms**
2. **Attention U-Net:**
   - Parameters: 31,389,295 (~31.4M)
   - PSNR: **20.7718 dB**
   - SSIM: **0.7976**
   - L1 Loss: **0.082345**
   - Mean CPU Inference Time: **189.99 ms**

### Hypothesis
Due to its global transposed attention bottleneck and adaptive cloud severity modulation, **CS-TransUNet** will:
1. Surpass U-Net in both **PSNR (>27.5 dB)** and **SSIM (>0.900)** by better reconstructing semantic ground context occluded by semi-transparent and thick cloud cover.
2. Require only **~7.8M parameters** (comparable to U-Net and $4.0\times$ fewer parameters than Attention U-Net).
3. Maintain an interactive CPU inference latency of **<160 ms**.

---

## 2. Experimental Control & Fixed Dataset Split

To guarantee strict scientific comparability and avoid data leakage:
- **Dataset Split:** Exactly matches `results/data_split.json` (Seed 42).
  - **Train:** 350 pairs
  - **Validation:** 75 pairs
  - **Test (Held-Out):** 75 pairs
- **Zero Test Contamination:** The 75 test scenes remain completely untouched during training, hyperparameter tuning, and checkpoint selection.
- **Identical Image Resolution:** Input: $256 \times 256 \times 3$, Output: $256 \times 256 \times 3$, normalized to $[0.0, 1.0]$.
- **Hardware Device:** Intel Core Ultra 9 285H CPU, 24 threads, deterministic seeding.

---

## 3. Loss Function Formulation

Unlike the baselines which trained purely on scalar $\mathcal{L}_1$ pixel loss, CS-TransUNet utilizes a **Perceptually & Severity-Aware Multi-Task Compound Loss**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{1}(\hat{\mathbf{Y}}, \mathbf{Y}) + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}}(\hat{\mathbf{Y}}, \mathbf{Y}) + \lambda_{\text{sev}} \mathcal{L}_{\text{sev}}(\hat{\mathbf{s}}, \mathbf{s}_{\text{target}})$$

Where:
1. **Pixel-Wise Reconstruction Loss ($\mathcal{L}_1$):**
   $$\mathcal{L}_1(\hat{\mathbf{Y}}, \mathbf{Y}) = \frac{1}{C \times H \times W} \sum_{c, h, w} |\hat{\mathbf{Y}}_{c,h,w} - \mathbf{Y}_{c,h,w}|$$
2. **Structural Similarity Loss ($\mathcal{L}_{\text{SSIM}}$):**
   $$\mathcal{L}_{\text{SSIM}}(\hat{\mathbf{Y}}, \mathbf{Y}) = 1 - \text{SSIM}(\hat{\mathbf{Y}}, \mathbf{Y})$$
   Forces the network to maintain contrast, edge definitions, and parcel boundary structures.
3. **Cloud Severity Auxiliary Loss ($\mathcal{L}_{\text{sev}}$):**
   $$\mathcal{L}_{\text{sev}}(\hat{\mathbf{s}}, \mathbf{s}_{\text{target}}) = \text{CrossEntropy}(\hat{\mathbf{s}}, \mathbf{s}_{\text{target}})$$
   Supervises the Cloud Severity Estimation Branch (CSEB) to correctly distinguish thin, medium, and thick cloud covers.
4. **Loss Weighting Hyperparameters:**
   - $\lambda_{\text{SSIM}} = 0.2$
   - $\lambda_{\text{sev}} = 0.05$

---

## 4. Controlled Training Hyperparameters

| Hyperparameter | Baseline U-Net | Baseline Attention U-Net | Proposed CS-TransUNet |
|---|---|---|---|
| **Optimizer** | Adam | Adam | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight_decay=$10^{-4}$) |
| **Initial Learning Rate** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) | ReduceLROnPlateau (factor=0.5, patience=3) | CosineAnnealingWarmRestarts ($T_0=10, T_{\text{mult}}=2, \eta_{\text{min}}=10^{-6}$) |
| **Batch Size** | 2 | 2 | 2 (with gradient accumulation step = 2 $\to$ effective batch size 4) |
| **Epochs** | 20 | 20 | 25 |
| **Input Dimensions** | $256 \times 256 \times 3$ | $256 \times 256 \times 3$ | $256 \times 256 \times 3$ |
| **Device** | CPU (Intel Core Ultra 9 285H) | CPU (Intel Core Ultra 9 285H) | CPU (Intel Core Ultra 9 285H, multi-threaded) |
| **Checkpoint Selection** | Lowest validation $\mathcal{L}_1$ | Lowest validation $\mathcal{L}_1$ | Lowest validation compound loss |

---

## 5. Planned Series of Experiments

### Experiment 1: Primary Model Benchmark Comparison
- **Models:** Baseline U-Net vs. Baseline Attention U-Net vs. Proposed CS-TransUNet.
- **Evaluation Set:** 75 held-out paired satellite images.
- **Reporting Metrics:**
  - PSNR mean, median, min, max, std (data_range=1.0)
  - SSIM mean, median, min, max, std (channel_axis=0, data_range=1.0)
  - L1 loss mean, median, std
  - Parameter count (M)
  - Model file size (MB)
  - Average CPU inference latency (ms)

### Experiment 2: Cloud Severity Modulation Ablation
- **Condition A:** CS-TransUNet without Cloud Severity Branch (Standard LayerNorm).
- **Condition B:** CS-TransUNet with Cloud Severity Modulation (AdaLN scaling and shifting).
- **Goal:** Quantify the exact contribution of severity awareness across different cloud opacity levels (thin haze vs. thick cumulus).

### Experiment 3: Bottleneck Attention Mechanism Ablation
- **Condition A:** Standard Spatial Multi-Head Self-Attention ($O(N^2)$ complexity).
- **Condition B:** Multi-Dconv Head Transposed Attention (MDTA, $O(C^2)$ complexity).
- **Goal:** Compare wall-clock CPU training time, memory consumption, and reconstruction fidelity.

### Experiment 4: Loss Formulation Ablation
- **Condition A:** Trained with $\mathcal{L}_1$ loss only.
- **Condition B:** Trained with Compound Loss ($\mathcal{L}_1 + 0.2 \cdot \mathcal{L}_{\text{SSIM}} + 0.05 \cdot \mathcal{L}_{\text{sev}}$).
- **Goal:** Assess texture preservation and edge sharpness on agricultural field boundaries.

---

## 6. Evaluation Protocol & Software Tooling

To maintain continuity with Phase 2 evaluation standards, all models will be benchmarked using the established evaluation script `evaluate_test_set.py`:
- **Image Metrics:**
  - `skimage.metrics.peak_signal_noise_ratio(target_np, prediction_np, data_range=1.0)`
  - `skimage.metrics.structural_similarity(target_np, prediction_np, channel_axis=0, data_range=1.0)`
  - `torch.mean(torch.abs(prediction - target))`
- **Sample Visualizations:**
  - Generate qualitative comparison panels displaying `Cloudy Input`, `Clean Ground Truth`, `U-Net Prediction`, `Attention U-Net Prediction`, and `CS-TransUNet Prediction` for 10 representative test scenes (thin, medium, and heavy cloud cover).
- **Aggregation:**
  - Metrics exported to structured JSON format (`results/evaluation/transformer_benchmark_metrics.json`) and CSV summaries.
