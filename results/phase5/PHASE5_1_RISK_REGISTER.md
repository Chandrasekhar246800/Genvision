# GenVision Phase 5.1: Scientific & Operational Risk Register

**Project:** GenVision — A Cloud Severity-Aware Transformer-Diffusion Framework for Cloud-Free Satellite Image Reconstruction and Intelligent Agricultural Analysis  
**Repository Root:** `C:\Users\mcsr8\Desktop\cloud-removal-poc`  
**Date:** 16 November 2026  
**Document:** Risk Register & Mitigation Strategy (Phase 5.1)  
**Status:** **`DESIGN ONLY — NOT FOR IMPLEMENTATION`**  

---

## 1. Executive Summary

Integrating generative diffusion models into satellite remote sensing workflows carries distinct mathematical, algorithmic, and computational risks. Unlike natural image synthesis where aesthetic plausibility suffices, agricultural remote sensing demands strict radiometric accuracy, precise geometric preservation of parcel boundaries, and reproducible pixel values for downstream spectral index calculation (NDVI, NDWI, NDMI).

This register identifies the **ten primary scientific and computational risks** confronting Phase 5, analyzes their severity and likelihood, and establishes formal mitigation protocols.

---

## 2. Comprehensive Risk Assessment Matrix

| Risk ID | Risk Description | Severity | Likelihood | Impact Area | Primary Mitigation Strategy |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **R-01** | Small-Dataset Overfitting ($N=350$) | **HIGH** | Medium | Generalization | Frozen backbone + ultra-compact U-Net (1.85M) + data augmentation. |
| **R-02** | Structural & Agricultural Hallucination | **CRITICAL** | Medium | Scientific Validity | Residual manifold bounding + deterministic DDIM ($\eta=0$). |
| **R-03** | Complex/Multimodal Residual Manifold | **MEDIUM** | Medium | Convergence | Static residual scaling ($s_r=4.0$) + $L_1$ noise loss. |
| **R-04** | Conditioning Signal Failure | **MEDIUM** | Low | Architecture | Direct 9-channel spatial concatenation + AdaGN feature modulation. |
| **R-05** | Excessive CPU Training Duration | **HIGH** | Low | Feasibility | Offline pre-caching of CS-TransUNet outputs ($\approx 35\text{ min}$ for 50 epochs). |
| **R-06** | Unusable CPU Inference Latency | **HIGH** | Low | Practicality | Few-step DDIM ($S=5..10$) + zero spatial cross-attention layers. |
| **R-07** | Perception-Distortion Trade-Off (PSNR Drop)| **HIGH** | Medium | Benchmark Metrics | Bounded residual addition ($\hat{x}_0 + \hat{r}_0$) guaranteeing safety floor. |
| **R-08** | CSEB Severity Embedding Inefficacy | **LOW** | Medium | Novelty Claim | Controlled ablation (Exp C vs Exp D) + reporting actual delta without bias. |
| **R-09** | Synthetic-to-Real Cloud Domain Gap | **HIGH** | High | Real Deployment | Clear limitation documentation + validation against Phase 3.2 real Sentinel-2 data. |
| **R-10** | Test-Set Contamination / Data Leakage | **CRITICAL** | Low | Reproducibility | Strict quarantine of held-out test split (Seed 42) until Phase 5.3 benchmark. |

---

## 3. Detailed Risk Analysis & Action Protocols

### Risk R-01: Overfitting on Small Paired Training Dataset ($N = 350$)
- **Failure Mode:** Diffusion models are typically trained on tens of thousands of samples. A high-capacity denoising network could easily memorize the 350 training scenes, achieving near-zero training loss while failing to generalize to unseen cloud patterns.
- **Root Cause:** Excessive model capacity relative to training data volume.
- **Formal Mitigations:**
  1. **Strictly Frozen CS-TransUNet:** The 7.13M parameter transformer is permanently frozen. Gradients are confined exclusively to the diffusion denoiser.
  2. **Ultra-Compact Denoiser ($1.85\text{M}$ params):** The denoising U-Net is constrained to 4 shallow stages (`32, 64, 128, 256` channels), keeping capacity strictly proportioned to $N=350$.
  3. **Data Augmentation:** Random horizontal and vertical flips, 90-degree rotations, and slight brightness jitter applied on the fly during training.
  4. **Target Variance Reduction:** Training on residual error $r_0$ significantly restricts the target parameter space compared to raw image space.

---

### Risk R-02: Structural & Agricultural Parcel Hallucination
- **Failure Mode:** Generative diffusion models synthesize realistic-looking field lines, irrigation pivots, or crop rows that do not actually exist beneath dense clouds. In agricultural analytics, hallucinating non-existent crop patterns corrupts acreage estimation and index modeling.
- **Root Cause:** High stochastic variance during reverse sampling ($\sigma_t > 0$) combined with unconstrained image-space generation.
- **Formal Mitigations:**
  1. **Residual Manifold Formulation:** The diffusion model cannot generate arbitrary geometry; it only predicts the additive perturbation $\hat{r}_0$ on top of CS-TransUNet's deterministic prediction $\hat{x}_0$.
  2. **Deterministic DDIM Reverse Trajectory ($\eta = 0$):** Eliminates random Gaussian noise injection during sampling, ensuring fully reproducible, deterministic trajectories.
  3. **Strict Range Clamping:** $\hat{x}_{\text{final}} = \text{clamp}(\hat{x}_0 + \hat{r}_0, 0.0, 1.0)$ prevents non-physical pixel intensities.
  4. **Direct Spatial Guidance:** Passing the cloudy image $y$ directly as input channels locks the spatial boundaries to genuine ground features visible in non-cloudy pixels.

---

### Risk R-03: Complex or Multimodal Residual Manifold
- **Failure Mode:** In areas with heavy cumulus clouds, multiple ground truth patterns could plausibly correspond to a single saturated white patch. The residual error distribution becomes multimodal, causing $L_2$ diffusion models to produce blurry compromises or mode-averaging.
- **Root Cause:** Standard Gaussian diffusion assumes unimodal forward perturbation.
- **Formal Mitigations:**
  1. **Static Residual Scaling ($s_r = 4.0$):** Rescales the empirical residual distribution ($\sigma \approx 0.045$) to unit variance $[-1.5, +1.5]$, matching the standard Gaussian diffusion prior.
  2. **$L_1$ Noise Prediction Loss:** Optimizing $\mathcal{L}_{\text{diff}} = \mathbb{E}[\|\epsilon - \epsilon_\theta\|_1]$ rather than $L_2$ penalizes median rather than mean error, preserving sharp spatial transitions.
  3. **Global Atmospheric Conditioning:** Injecting $e_{\text{cseb}}$ via AdaGN provides global optical context to disambiguate local residual patterns.

---

### Risk R-04: Conditioning Signal Failure & Spatial Disconnection
- **Failure Mode:** The denoising network ignores the conditioning signals (cloudy image $y$, initial reconstruction $\hat{x}_0$, and severity vector $e_{\text{cseb}}$), generating unguided residuals.
- **Root Cause:** Complex conditioning mechanisms (such as high-dimensional cross-attention) failing to learn alignment on a small dataset.
- **Formal Mitigations:**
  1. **Direct Spatial Concatenation:** Spatial conditioning is supplied via direct 9-channel concatenation at the input layer ($[\tilde{r}_t, \hat{x}_0, y]$). Convolutional kernels cannot bypass or ignore input channels, forcing immediate spatial interaction at Stage 1.
  2. **AdaGN Feature Injection:** Global conditioning $e_{\text{cseb}}$ directly scales and shifts the normalization parameters at every single stage.

---

### Risk R-05: Excessive CPU Training Duration
- **Failure Mode:** Training diffusion models on CPU typically requires multiple days, stalling development and preventing rigorous parameter tuning.
- **Root Cause:** Re-running the 7.13M parameter CS-TransUNet forward pass at every diffusion training step across hundreds of epochs.
- **Formal Mitigations:**
  1. **Offline Pre-Caching:** Run CS-TransUNet **once** over the 350 training images before training starts ($\approx 45\text{ seconds}$). Store $\hat{x}_0$, $e_{\text{cseb}}$, and $\tilde{r}_0$ in RAM/SSD.
  2. **Zero CS-TransUNet Passes During Training:** The training loop loads pre-computed tensors directly, reducing step time to $\approx 173\text{ ms}$.
  3. **Measured Epoch Duration:** Training executes at **$\approx 35\text{ seconds}$ per epoch** on CPU, allowing a full 50-epoch run to complete in $\approx 30 - 35\text{ minutes}$.

---

### Risk R-06: Unusable CPU Inference Latency
- **Failure Mode:** End-to-end inference takes tens of seconds per scene, making the model unsuitable for interactive UI dashboards or batch API serving.
- **Root Cause:** High reverse step count ($T \ge 50$) and expensive multi-head cross-attention layers.
- **Formal Mitigations:**
  1. **Exclusion of Spatial Cross-Attention:** The denoising U-Net uses pure convolutional residual blocks, reducing single-step CPU forward latency from $\approx 350\text{ ms}$ to $\mathbf{38\text{ ms}}$.
  2. **Few-Step DDIM Sampling ($S = 5 \text{ to } 10$ steps):** Total reverse diffusion latency is bounded to **$190\text{ ms} - 380\text{ ms}$**.
  3. **Total Latency Budget:** $130\text{ ms}$ (CS-TransUNet) $+ 380\text{ ms}$ (DDIM $S=10$) $+ 20\text{ ms}$ (clamp/IO) $\approx \mathbf{530\text{ ms}}$ per image.

---

### Risk R-07: The Perception-Distortion Trade-Off (PSNR Degradation)
- **Failure Mode:** It is well-documented in vision literature (Blau & Michaeli, 2018) that diffusion models optimize perceptual quality (texture realism, sharpness) at the expense of mean squared error (PSNR). A diffusion model might synthesize crisp field boundaries that slightly misalign with pixel-exact ground truth, resulting in lower PSNR despite looking superior to human evaluators.
- **Root Cause:** Generative stochasticity drifting away from the pixel-wise conditional mean.
- **Formal Mitigations:**
  1. **Residual Bounds & Skip Addition:** The final reconstruction is $\hat{x}_{\text{final}} = \hat{x}_0 + \hat{r}_0$. Because $\hat{r}_0$ is small and zero-centered, large PSNR drops are mathematically bounded.
  2. **Deterministic Sampling ($\eta = 0$):** Using deterministic DDIM eliminates stochastic drift.
  3. **Balanced Reporting:** We will track both distortion metrics (PSNR, SSIM, $L_1$) and high-frequency sharpness metrics, evaluating the trade-off transparently.

---

### Risk R-08: CSEB Severity Embedding Providing Negligible Benefit
- **Failure Mode:** Experiments show that adding the 192-dimensional continuous CSEB embedding yields no statistically significant improvement over an unconditioned residual diffusion model.
- **Root Cause:** Spatial conditioning $[r_t, \hat{x}_0, y]$ may already contain sufficient atmospheric context, rendering the global embedding redundant.
- **Formal Mitigations:**
  1. **Ablation Studies (Exp C vs Exp D):** We design explicit controlled experiments specifically to test this hypothesis.
  2. **Stratified Evaluation:** We evaluate whether CSEB provides benefits specifically on the top 20% most severely cloudy scenes, where global context matters most.
  3. **Honest Scientific Reporting:** If the benefit is statistically insignificant ($p \ge 0.05$), we will document this transparently as a scientific negative finding and retain the simpler unconditioned architecture.

---

### Risk R-09: Domain Gap Between Synthetic and Genuine Satellite Clouds
- **Failure Mode:** A model trained on synthetic clouds fails to remove genuine atmospheric aerosols, cirrus veils, or shadow projections on real Sentinel-2 Level-2A imagery.
- **Root Cause:** Synthetic cloud masks often have simplified edge blur and lack realistic multi-spectral water vapor absorption dynamics.
- **Formal Mitigations:**
  1. **Architectural Separation:** The diffusion model operates as a modular add-on; downstream Level-2A agricultural pipelines (established in Phase 3.2 for the Pulivendula ROI) remain independent.
  2. **Qualitative Stress-Testing:** We will evaluate the trained model on real Level-2A cloudy tiles from the Pulivendula dataset to document real-world transferability.
  3. **Explicit Documentation:** Document synthetic cloud pairing as a formal research limitation in all reports.

---

### Risk R-10: Dataset Leakage and Test-Set Contamination
- **Failure Mode:** Held-out test scenes are inadvertently included in training pre-caching or hyperparameter selection, invalidating benchmark credibility.
- **Root Cause:** Loose data loading paths or unified evaluation loops.
- **Formal Mitigations:**
  1. **Strict Evaluation Freeze:** The 75 test scenes in [`results/data_split.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/data_split.json) (Seed 42) remain permanently quarantined.
  2. **Code-Level Assertions:** The pre-caching script will strictly assert:
     $$\text{cache\_indices} \cap \text{test\_indices} = \emptyset$$
  3. **Single Benchmark Evaluation:** The test set will only be accessed during the final Phase 5.3 benchmark after all models and sampling parameters are permanently locked.
