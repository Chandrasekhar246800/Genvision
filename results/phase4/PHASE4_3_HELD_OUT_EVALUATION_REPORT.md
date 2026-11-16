# GenVision Phase 4.3: Controlled Held-Out Evaluation & Benchmarking Report

**Project:** GenVision — A Cloud Severity-Aware Transformer-Diffusion Framework for Cloud-Free Satellite Image Reconstruction and Intelligent Agricultural Analysis  
**Repository Root:** `C:\Users\mcsr8\Desktop\cloud-removal-poc`  
**Date:** 16 November 2026  
**Final Evaluation Status:** **CS-TRANSUNET OUTPERFORMS BASELINES**  
**Lead Auditor & Scientist:** Senior AI Research & Computer Vision Specialist  

---

## 1. Executive Summary

In Phase 4.3, we executed a scientifically controlled, statistically sound held-out evaluation benchmarking the proposed **Cloud Severity-Aware Hybrid Transformer (`CS-TransUNet`)** against the two established baselines: standard **U-Net** and **Attention U-Net**.

All three models were evaluated under identical conditions on the **75 strictly held-out test scenes** defined in [`results/data_split.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/data_split.json) (Seed 42), with zero data leakage or prior exposure during training or checkpoint selection.

### Key Empirical Findings

1. **Reconstruction Fidelity (PSNR):**
   - **CS-TransUNet:** **$28.7877 \pm 4.6909\text{ dB}$** (Median: $28.3979\text{ dB}$)
   - **U-Net Baseline:** $26.6202 \pm 5.3141\text{ dB}$ (Median: $25.5174\text{ dB}$)
   - **Attention U-Net Baseline:** $20.7718 \pm 2.3093\text{ dB}$ (Median: $21.0454\text{ dB}$)
   - *CS-TransUNet achieves an absolute mean improvement of **$+2.1676\text{ dB}$ ($+8.14\%$)** over U-Net and **$+8.0159\text{ dB}$ ($+38.59\%$)** over Attention U-Net.*
   - Both improvements are statistically significant at $p < 0.0001$ ($p = 2.78 \times 10^{-8}$ vs U-Net; $p = 7.69 \times 10^{-23}$ vs Attention U-Net).

2. **Structural Similarity (SSIM):**
   - **CS-TransUNet:** **$0.935942 \pm 0.030900$** (Median: $0.941215$)
   - **U-Net Baseline:** $0.885598 \pm 0.051622$ (Median: $0.888321$)
   - **Attention U-Net Baseline:** $0.797573 \pm 0.112722$ (Median: $0.836479$)
   - *CS-TransUNet achieves an absolute mean improvement of **$+0.050344$ ($+5.68\%$)** over U-Net and **$+0.138369$ ($+17.35\%$)** over Attention U-Net ($p = 4.01 \times 10^{-15}$ and $p = 3.26 \times 10^{-16}$).*

3. **Reconstruction Error ($L_1$ / MAE):**
   - **CS-TransUNet:** **$0.034316 \pm 0.019620$** (Median: $0.031938$)
   - **U-Net Baseline:** $0.043854 \pm 0.025131$ (Median: $0.042028$)
   - **Attention U-Net Baseline:** $0.082345 \pm 0.030822$ (Median: $0.077231$)
   - *CS-TransUNet achieves an absolute error reduction of **$-0.009538$ ($-21.75\%$ relative error reduction)** vs U-Net and **$-0.048029$ ($-58.33\%$ relative error reduction)** vs Attention U-Net ($p = 2.89 \times 10^{-6}$ and $p = 3.93 \times 10^{-19}$).*

4. **Computational & Parameter Efficiency:**
   - CS-TransUNet has **$7,128,906$ parameters** (~$7.13\text{M}$), making it **$7.39\%$ lighter than U-Net** ($7.70\text{M}$) and **$77.29\%$ lighter than Attention U-Net** ($31.39\text{M}$).
   - On the Intel Core Ultra 9 285H CPU, CS-TransUNet demonstrates an average inference latency of **$130.00\text{ ms}$** per image, matching U-Net ($130.82\text{ ms}$) while running **$29.42\%$ faster than Attention U-Net** ($184.20\text{ ms}$).

---

## 2. Pre-Evaluation Audit & Checkpoint Verification

### 2.1 Dataset Partitioning & Leakage Audit
- **Split Path:** [`results/data_split.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/data_split.json) (Seed 42)
- **Total Dataset Size:** 500 paired images.
- **Training Set:** 350 pairs ($70\%$).
- **Validation Set:** 75 pairs ($15\%$).
- **Held-Out Test Set:** Exactly 75 pairs ($15\%$).
- **Zero Leakage Verification:**
  $$\text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
  All 75 test scenes were strictly withheld during all phases of architecture design, hyperparameter selection, model training, and validation checkpoint selection.

### 2.2 Evaluated Checkpoints
1. **U-Net Baseline:**
   - Path: [`models/unet_controlled_best.pth`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/unet_controlled_best.pth)
   - Size: $30.80\text{ MB}$ (Timestamp: Oct 13 2026, unmodified)
2. **Attention U-Net Baseline:**
   - Path: [`models/attention_controlled_best.pth`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/attention_controlled_best.pth)
   - Size: $125.68\text{ MB}$ (Timestamp: Oct 13 2026, unmodified)
3. **CS-TransUNet (Proposed Architecture):**
   - Path: [`models/transformer_controlled_best.pth`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/transformer_controlled_best.pth)
   - Size: $85.69\text{ MB}$ (Timestamp: Nov 16 2026 04:39)
   - Selected Checkpoint: **Epoch 24** (Minimum validation loss $0.033073$, not the later Epoch 25 checkpoint).

---

## 3. Evaluation Methodology & Metric Formulations

All models were evaluated under identical conditions:
1. **Input Preprocessing:**
   - High-fidelity PIL RGB conversion.
   - Resized to $(256, 256)$ via bilinear interpolation.
   - Converted to PyTorch tensor normalized in $[0.0, 1.0]$ via `torchvision.transforms.ToTensor()`.
2. **Output Handling:**
   - Model predictions generated under `torch.inference_mode()` on CPU.
   - Predictions clamped strictly to $[0.0, 1.0]$.
3. **Metric Definitions:**
   - **Peak Signal-to-Noise Ratio (PSNR):**
     $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) = 10 \cdot \log_{10}\left(\frac{1.0}{\frac{1}{3HW}\sum (I - \hat{I})^2}\right)$$
     Computed via `skimage.metrics.peak_signal_noise_ratio` with `data_range=1.0`.
   - **Structural Similarity Index Measure (SSIM):**
     $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
     Computed via `skimage.metrics.structural_similarity` with `channel_axis=0`, uniform box window size $7 \times 7$ (`win_size=7`, `gaussian_weights=False`, `use_sample_covariance=True`), stabilization constants $c_1 = 10^{-4}, c_2 = 9 \times 10^{-4}$, and `data_range=1.0`.
   - **Mean Absolute Error ($L_1$ Loss):**
     $$\mathcal{L}_{L1} = \frac{1}{3HW}\sum_{c=1}^3 \sum_{h=1}^H \sum_{w=1}^W \left| \hat{I}_{c,h,w} - I_{c,h,w} \right|$$
   - **Per-Image Aggregation:** Metrics were computed on each individual held-out scene before calculating summary statistics (mean, median, standard deviation, min, max) across the 75 test scenes.

---

## 4. Aggregate Held-Out Benchmark Results

### 4.1 Master Model Comparison Table

| Metric | Baseline U-Net | Baseline Attention U-Net | CS-TransUNet (Ours) | Superior Model |
| :--- | :---: | :---: | :---: | :---: |
| **Model Parameters** | 7,697,475 (7.70M) | 31,389,295 (31.39M) | **7,128,906 (7.13M)** | **CS-TransUNet (-7.4% vs U-Net, -77.3% vs Att)** |
| **Model Size on Disk** | 30.80 MB | 125.68 MB | **85.69 MB** | U-Net (weights-only vs full state dict) |
| **PSNR Mean (dB) ↑** | $26.6202 \pm 5.3141$ | $20.7718 \pm 2.3093$ | **28.7877 ± 4.6909** | **CS-TransUNet (+2.17 dB vs U-Net)** |
| **PSNR Median (dB) ↑** | 25.5174 | 21.0454 | **28.3979** | **CS-TransUNet (+2.88 dB vs U-Net)** |
| **PSNR Min / Max (dB)** | 15.7568 / 39.1542 | 12.1282 / 25.2363 | **19.4760 / 39.8922** | **CS-TransUNet (+3.72 dB higher floor)** |
| **SSIM Mean ↑** | $0.885598 \pm 0.051622$ | $0.797573 \pm 0.112722$ | **0.935942 ± 0.030900** | **CS-TransUNet (+0.0503 vs U-Net)** |
| **SSIM Median ↑** | 0.888321 | 0.836479 | **0.941215** | **CS-TransUNet (+0.0529 vs U-Net)** |
| **SSIM Min / Max** | 0.757263 / 0.984510 | 0.360544 / 0.919241 | **0.840625 / 0.986629** | **CS-TransUNet (+0.0834 higher floor)** |
| **$L_1$ Error Mean ↓** | $0.043854 \pm 0.025131$ | $0.082345 \pm 0.030822$ | **0.034316 ± 0.019620** | **CS-TransUNet (-21.75% error reduction)** |
| **$L_1$ Error Median ↓** | 0.042028 | 0.077231 | **0.031938** | **CS-TransUNet (-24.01% error reduction)** |
| **$L_1$ Error Min / Max** | 0.008291 / 0.135532 | 0.043614 / 0.242248 | **0.007685 / 0.091713** | **CS-TransUNet (-32.33% lower max error)** |
| **Inference Mean (ms)** | $130.82 \pm 22.72$ | $184.20 \pm 19.55$ | **130.00 ± 23.39** | **CS-TransUNet (Parity with U-Net, -29.4% vs Att)** |
| **Inference Median (ms)** | 125.94 | 180.01 | **125.37** | **CS-TransUNet** |

---

## 5. Statistical Significance Analysis

To establish whether the observed performance gains of CS-TransUNet are statistically significant or attributable to sample variance, we conducted two-sided paired Student's t-tests and non-parametric Wilcoxon signed-rank tests across the 75 paired held-out test scenes.

### 5.1 CS-TransUNet vs Baseline U-Net

| Metric | Mean Paired Difference ($\bar{d}$) | Median Paired Difference | 95% Confidence Interval | Relative Improvement | Paired t-test ($t$, $p$) | Wilcoxon Signed-Rank ($W$, $p$) | Significance ($\alpha = 0.01$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** | **$+2.1676\text{ dB}$** | $+2.3229\text{ dB}$ | $[+1.4724, +2.8628]\text{ dB}$ | **$+8.14\%$** | $t = 6.212$, $p = 2.78 \times 10^{-8}$ | $W = 451.0$, $p = 2.70 \times 10^{-7}$ | **STATISTICALLY SIGNIFICANT** |
| **SSIM** | **$+0.050344$** | $+0.055595$ | $[+0.040172, +0.060517]$ | **$+5.68\%$** | $t = 9.861$, $p = 4.01 \times 10^{-15}$ | $W = 190.0$, $p = 6.96 \times 10^{-11}$ | **STATISTICALLY SIGNIFICANT** |
| **$L_1$ Error** | **$-0.009538$** | $-0.005666$ | $[-0.013289, -0.005786]$ | **$-21.75\%$** | $t = -5.066$, $p = 2.89 \times 10^{-6}$ | $W = 629.0$, $p = 2.63 \times 10^{-5}$ | **STATISTICALLY SIGNIFICANT** |

### 5.2 CS-TransUNet vs Baseline Attention U-Net

| Metric | Mean Paired Difference ($\bar{d}$) | Median Paired Difference | 95% Confidence Interval | Relative Improvement | Paired t-test ($t$, $p$) | Wilcoxon Signed-Rank ($W$, $p$) | Significance ($\alpha = 0.01$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** | **$+8.0159\text{ dB}$** | $+6.6215\text{ dB}$ | $[+6.8918, +9.1400]\text{ dB}$ | **$+38.59\%$** | $t = 14.209$, $p = 7.69 \times 10^{-23}$ | $W = 5.0$, $p = 6.46 \times 10^{-14}$ | **STATISTICALLY SIGNIFICANT** |
| **SSIM** | **$+0.138369$** | $+0.093636$ | $[+0.111979, +0.164759]$ | **$+17.35\%$** | $t = 10.447$, $p = 3.26 \times 10^{-16}$ | $W = 0.0$, $p = 5.28 \times 10^{-14}$ | **STATISTICALLY SIGNIFICANT** |
| **$L_1$ Error** | **$-0.048029$** | $-0.037921$ | $[-0.055967, -0.040090]$ | **$-58.33\%$** | $t = -12.055$, $p = 3.93 \times 10^{-19}$ | $W = 14.0$, $p = 9.27 \times 10^{-14}$ | **STATISTICALLY SIGNIFICANT** |

### 5.3 Per-Image Dominance Rates
Across the 75 held-out scenes:
- **CS-TransUNet beats U-Net on SSIM:** **68 out of 75 scenes (90.67%)**.
- **CS-TransUNet beats U-Net on PSNR:** **59 out of 75 scenes (78.67%)**.
- **CS-TransUNet beats U-Net on $L_1$ Error:** **59 out of 75 scenes (78.67%)**.
- **CS-TransUNet beats Attention U-Net on all metrics:** **74 out of 75 scenes (98.67%)**.

---

## 6. Qualitative & Visual Comparison Analysis

Ten diverse test scenes were rendered under [`results/phase4/evaluation/visualizations/`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/visualizations):

```
results/phase4/evaluation/visualizations/
├── comparison_01_257.png    # High-difficulty case: Heavy cloud occlusion
├── comparison_02_255.png    # Heavy cloud occlusion with fine agricultural boundaries
├── comparison_03_47.png     # Dense cloud patch over high-contrast vegetation
├── comparison_04_273.png    # Moderate cloud cover over structured field plots
├── comparison_05_100.png    # Median test difficulty case
├── comparison_06_198.png    # Moderate cloud cover with subtle road and path networks
├── comparison_07_377.png    # Thin cirrus haze with spectral distortion
├── comparison_08_171.png    # Thin cloud veil over homogeneous canopy
├── comparison_09_345.png    # Light cloud scattering
└── comparison_10_293.png    # Lightest cloud haze baseline
```

### Visual Observations & Insights
1. **Cloud Edge Artifacts in Baselines:**
   - Standard U-Net frequently leaves residual cloud silhouettes, manifesting as milky halos around dense cloud cores.
   - Attention U-Net exhibits oversmoothing in dense cloud regions, washing out field boundaries and causing severe spectral desaturation.
2. **CS-TransUNet Structural Restoration:**
   - The combination of **MDTA (Multi-Dconv Head Transposed Attention)** and **AdaLN severity conditioning** allows CS-TransUNet to dynamically identify the spatial extent and density of cloud cover.
   - Ground textures beneath thin and moderate cloud layers are reconstructed with sharp geometric edge definitions.
3. **Absolute Error Heatmaps:**
   - Examination of the lower panels in each visualization reveals that CS-TransUNet's absolute error maps remain consistently darker (lower error magnitude) across both cloud-covered and clear pixels, demonstrating that clear pixels are preserved without distortion while cloudy pixels are accurately inpainted.

---

## 7. Distributional Plots & Boxplots

Boxplots illustrating the distribution of held-out test metrics across the 75 scenes have been saved under [`results/phase4/evaluation/`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation):
- **PSNR Distribution:** [`results/phase4/evaluation/test_psnr_boxplot.png`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/test_psnr_boxplot.png)
  *(Highlights CS-TransUNet's superior median and compressed interquartile range).*
- **SSIM Distribution:** [`results/phase4/evaluation/test_ssim_boxplot.png`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/test_ssim_boxplot.png)
  *(Demonstrates that CS-TransUNet achieves $\ge 0.90$ SSIM on 88.00% (66/75) of held-out scenes with median 0.9412 and an empirical lower bound of 0.8406 under dense, opaque cloud occlusion).*
- **$L_1$ Error Distribution:** [`results/phase4/evaluation/test_l1_boxplot.png`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/test_l1_boxplot.png)
  *(Demonstrates significant reduction in both mean error and peak error tails).*

---

## 8. Artifact Directory Map

All generated evaluation artifacts are stored and available for audit:
- **Per-Image Metrics Table:** [`results/phase4/evaluation/per_image_metrics.csv`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/per_image_metrics.csv)
- **Model Comparison Table:** [`results/phase4/evaluation/model_comparison.csv`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/model_comparison.csv)
- **Structured JSON Summary:** [`results/phase4/evaluation/evaluation_summary.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/evaluation_summary.json)
- **High-Resolution Figures:** [`results/phase4/evaluation/visualizations/`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/visualizations)

---

## 9. Scientific Caveats & Limitations

1. **Fixed Evaluation Split:** Evaluation was conducted on the fixed 75-scene held-out partition established in Seed 42. While results are statistically significant, validation across independent multi-temporal or geographic cross-validation splits will provide further generalizability.
2. **Synthetic Cloud Pairing:** The paired dataset consists of synthetic cloud-augmented Sentinel RGB scenes. Performance on genuine Level-2A multi-band Sentinel-2 imagery (as processed in Phase 3.2) will be integrated in downstream agricultural workflows.
3. **CSEB Unsupervised Self-Conditioned Severity Representation:** The Cloud Severity Estimator Branch was trained without supervised category targets or auxiliary classification loss. Its 3-class output logits/probabilities are uncalibrated and ungrounded; it functions strictly as an unsupervised self-conditioned severity representation (192-dim continuous embedding) for AdaLN transformer modulation.

---

## 10. Final Scientific Conclusion & Status Determination

The experimental evidence confirms:
1. CS-TransUNet achieves higher PSNR, higher SSIM, and lower $L_1$ reconstruction error than both baseline architectures.
2. The performance gains are statistically significant ($p < 0.0001$ across all tests).
3. The model achieves these improvements while using **$77.3\%$ fewer parameters** and running **$29.4\%$ faster** than Attention U-Net.

**FINAL STATUS:**

# **`CS-TRANSUNET OUTPERFORMS BASELINES`**

Phase 4.3 is complete. Diffusion model development (Phase 5) has **not** been started automatically.
Awaiting your review and explicit instructions before proceeding to the next phase of the GenVision research framework.
