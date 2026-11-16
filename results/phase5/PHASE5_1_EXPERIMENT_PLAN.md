# GenVision Phase 5.1: Controlled Experimentation & Benchmarking Plan

**Project:** GenVision — A Cloud Severity-Aware Transformer-Diffusion Framework for Cloud-Free Satellite Image Reconstruction and Intelligent Agricultural Analysis  
**Repository Root:** `C:\Users\mcsr8\Desktop\cloud-removal-poc`  
**Date:** 16 November 2026  
**Document:** Scientific Experiment Protocol (Phase 5.1)  
**Status:** **`DESIGN ONLY — NOT FOR IMPLEMENTATION`**  

---

## 1. Executive Summary & Experimental Philosophy

To scientifically validate that adding a conditional diffusion module provides measurable, statistically significant value beyond the Phase 4 CS-TransUNet baseline, we establish a controlled experimental protocol.

### Fundamental Scientific Tenets
1. **The Held-Out Test Set Remains Sealed:**  
   The 75 held-out test scenes in [`results/data_split.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/data_split.json) (Seed 42) are under **strict evaluation freeze**. All architecture tuning, loss function selection, learning rate calibration, and ablation decisions must rely exclusively on the **350 training scenes** and **75 validation scenes**. The test set will only be accessed once during the final benchmark in Phase 5.3.
2. **Identical Preprocessing & Evaluation Harness:**  
   All experiments will strictly use the identical preprocessing pipeline, tensor normalization $[0.0, 1.0]$, clamping, and evaluation metric implementations validated in Phase 4.3.2.
3. **Statistical Significance Testing:**  
   Every performance gain must be accompanied by two-sided paired Student's t-tests, Wilcoxon signed-rank tests, 95% confidence intervals, and Bonferroni corrections for multiple comparisons.
4. **Distinction vs Perception Tracking:**  
   Because diffusion models are designed to generate high-frequency textures that can improve perceptual sharpness while occasionally altering pixel-level MSE, both standard distortion metrics (PSNR, SSIM, $L_1$) and high-frequency edge metrics will be tracked.

---

## 2. Core Experimental Suite (Experiments A Through F)

The experimental matrix consists of six structured experiments designed to isolate each architectural mechanism:

```
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                    GENVISION EXPERIMENTAL MATRIX                        |
+───────────────────────────┬───────────────────────────────────┬─────────────────────────+
| Experiment ID             | Core Architectural Configuration  | Primary Objective       |
+───────────────────────────┼───────────────────────────────────┼─────────────────────────+
| Experiment A (Baseline)   | Frozen CS-TransUNet Alone         | Performance Control     |
| Experiment B (Proposed)   | CS-TransUNet + Residual Diffusion | Full Framework Benchmark|
| Experiment C (Ablation 1) | Residual Diffusion w/o CSEB       | Isolate CSEB Benefit    |
| Experiment D (Ablation 2) | Residual Diffusion WITH CSEB      | Quantify Severity Prior |
| Experiment E (Ablation 3) | Sampling Step Sweep (S=1..50)     | Latency-Quality Curve   |
| Experiment F (Ablation 4) | Residual-Space vs Image-Space     | Prove Residual Manifold |
+───────────────────────────┴───────────────────────────────────┴─────────────────────────+
```

---

### Experiment A: Frozen CS-TransUNet Baseline (Control)

- **Research Question:** What is the exact benchmark performance of the deterministic CS-TransUNet model on the validation split under the frozen Phase 4 checkpoint?
- **Hypothesis:** CS-TransUNet serves as the primary lower-bound benchmark ($H_0$). The diffusion framework must demonstrate statistically significant improvement over this baseline.
- **Independent Variable:** None (Evaluation of the frozen checkpoint [`models/transformer_controlled_best.pth`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/transformer_controlled_best.pth)).
- **Controlled Variables:** 
  - Validation partition (75 validation scenes, Seed 42).
  - Image size $256 \times 256$, normalized $[0.0, 1.0]$.
  - Evaluation device: CPU.
- **Evaluation Metrics:** PSNR (dB), SSIM, $L_1$ Error, Inference Latency (ms).
- **Expected Computational Cost:** $\approx 10\text{ seconds}$ total evaluation time on CPU.

---

### Experiment B: CS-TransUNet with Full Conditional Residual Diffusion (Proposed Framework)

- **Research Question:** Does adding a lightweight conditional residual diffusion model ($1.85\text{M}$ params) statistically improve reconstruction fidelity, edge sharpness, and cloud inpainting over the frozen baseline?
- **Hypothesis:** Modeling the residual manifold with spatial cloudy conditioning $[r_t, \hat{x}_0, y]$ and AdaGN continuous severity conditioning $e_{\text{cseb}}$ will yield statistically significant gains in PSNR ($+0.5$ to $+1.5\text{ dB}$), SSIM ($+0.01$ to $+0.02$), and $L_1$ error reduction on the validation split.
- **Independent Variable:** Inclusion of the trained residual diffusion module ($S = 10$ DDIM steps).
- **Controlled Variables:**
  - Frozen CS-TransUNet weights.
  - Spatial input channels: 9.
  - Variance schedule: Linear $\beta \in [10^{-4}, 0.02], T = 1000$.
  - Conditioning: Combined $[ \phi(t), e_{\text{cseb}} ]$.
  - Training hyperparams: 50 epochs, AdamW ($\text{lr} = 2 \times 10^{-4}$), batch size 4.
- **Evaluation Metrics:** PSNR, SSIM, $L_1$ Error, Latency, Paired t-test ($p < 0.01$), Wilcoxon signed-rank ($p < 0.01$), 95% CI.
- **Expected Computational Cost:**
  - One-time pre-caching: $\approx 45\text{ seconds}$.
  - Training (50 epochs): $\approx 35\text{ minutes}$.
  - Validation evaluation (75 scenes): $\approx 40\text{ seconds}$.

---

### Experiment C: Residual Diffusion Ablation Without CSEB Conditioning

- **Research Question:** How much of the diffusion refinement performance is driven purely by the spatial conditioning $[r_t, \hat{x}_0, y]$ in the absence of the global cloud severity embedding?
- **Hypothesis:** Without the CSEB continuous embedding $e_{\text{cseb}}$, the diffusion model will struggle to adjust its denoising intensity between thin haze regions and opaque cloud cores, leading to lower reconstruction gains under severe occlusion.
- **Independent Variable:** Global conditioning vector restricted to timestep embedding alone: $\mathbf{v}_{\text{cond}} = \phi(t) \in \mathbb{R}^{128}$ (CSEB embedding omitted: $e_{\text{cseb}} = \mathbf{0}$).
- **Controlled Variables:**
  - Identical 1.85M U-Net architecture.
  - Identical training schedule, optimizer, seed (42), and dataset partition.
  - Identical 9-channel spatial input.
- **Evaluation Metrics:** $\Delta\text{PSNR}$, $\Delta\text{SSIM}$, $\Delta L_1$ compared to Experiment B, stratified by cloud thickness difficulty.
- **Expected Computational Cost:** $\approx 35\text{ minutes}$ training on CPU.

---

### Experiment D: Residual Diffusion With Continuous CSEB Conditioning

- **Research Question:** Does injecting the 192-dimensional unsupervised continuous severity representation from CSEB via AdaGN provide measurable, statistically significant guidance over the unconditioned baseline (Experiment C)?
- **Hypothesis:** Conditioning on $e_{\text{cseb}}$ allows intermediate feature maps to adapt their gain and bias according to global atmospheric optical depth, producing significantly sharper boundary recovery in cloudy areas ($p < 0.05$ vs Exp C).
- **Independent Variable:** Full conditioning vector $\mathbf{v}_{\text{cond}} = [ \phi(t), e_{\text{cseb}} ] \in \mathbb{R}^{320}$.
- **Controlled Variables:**
  - Identical architecture, seeds, learning rate, and sampling protocol to Experiment C.
  - Strict restriction: No uncalibrated classification logits used.
- **Evaluation Metrics:** Paired difference $(\text{Exp D} - \text{Exp C})$, Paired t-test, Wilcoxon test, Stratified metrics on top 20% most severe cloud scenes.
- **Expected Computational Cost:** Evaluated directly as part of Experiment B.

---

### Experiment E: Ablation of Reverse Sampling Steps (Efficiency vs Quality Pareto Frontier)

- **Research Question:** What is the optimal number of DDIM sampling steps ($S$) that balances reconstruction fidelity against CPU inference latency?
- **Hypothesis:** Residual-space diffusion converges rapidly; performance will plateau around $S = 5 \text{ to } 10$ steps, with $S > 10$ offering negligible marginal gain while linearly increasing CPU runtime.
- **Independent Variable:** DDIM reverse sampling steps:
  $$S \in \{ 1, \, 3, \, 5, \, 10, \, 15, \, 20, \, 50 \}$$
- **Controlled Variables:**
  - Trained model weights from Experiment B.
  - Identical 75 validation scenes.
  - Deterministic DDIM sampling ($\eta = 0$).
- **Evaluation Metrics:**
  - Curve plotting: PSNR vs Latency (ms).
  - Curve plotting: SSIM vs Latency (ms).
  - Pareto-optimal step identification.
- **Expected Computational Cost:** $\approx 3\text{ minutes}$ total on CPU across all 7 sweeps.

---

### Experiment F: Residual-Space Versus Image-Space Diffusion (The Manifold Ablation)

- **Research Question:** Does diffusing over the residual manifold ($r_0 = x_0 - \hat{x}_0$) provide demonstrable mathematical and practical advantages over diffusing directly over clean image space ($x_0$)?
- **Hypothesis:** Direct image-space conditional diffusion will suffer from slower convergence, higher spatial variance, and a lower minimum PSNR floor when trained on 350 images, whereas residual diffusion will maintain stability and preserve the CS-TransUNet floor.
- **Independent Variable:** Target diffusion representation:
  1. *Condition 1 (Residual Space):* $r_t = \sqrt{\bar{\alpha}_t} (s_r(x_0 - \hat{x}_0)) + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
  2. *Condition 2 (Image Space):* $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$, conditioned on $[y, \hat{x}_0]$.
- **Controlled Variables:**
  - Identical 1.85M parameter denoising backbone architecture.
  - Identical 50 epochs, learning rate, batch size, and seed.
  - Identical $S = 10$ DDIM steps.
- **Evaluation Metrics:** Validation loss convergence rate, final PSNR/SSIM, worst-case minimum PSNR, presence of spatial hallucination artifacts.
- **Expected Computational Cost:** $\approx 35\text{ minutes}$ training for Condition 2.

---

## 3. Comprehensive Summary Protocol Matrix

| Experiment | Configuration Tested | Input Channels | Conditioning Signals | Target Output | Primary Metric of Interest |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Exp A** | Frozen CS-TransUNet | 3 | None | $\hat{x}_0$ | Reference Baseline Floor |
| **Exp B** | ResDiff-TransUNet (Full) | 9 | $[ \phi(t), e_{\text{cseb}} ]$ | $\hat{r}_0$ | Overall Framework Gain ($p < 0.01$) |
| **Exp C** | ResDiff w/o CSEB | 9 | $\phi(t)$ only | $\hat{r}_0$ | CSEB Ablation ($\Delta\text{Metric}$) |
| **Exp D** | ResDiff with CSEB | 9 | $[ \phi(t), e_{\text{cseb}} ]$ | $\hat{r}_0$ | Severity Guidance Significance |
| **Exp E** | Sampling Step Sweep | 9 | $[ \phi(t), e_{\text{cseb}} ]$ | $\hat{r}_0$ ($S \in [1..50]$) | Pareto Frontier (Latency vs PSNR) |
| **Exp F** | Image-Space vs Residual | 6 vs 9 | Identical | $x_0$ vs $r_0$ | Manifold Stability & Floor Preservation |

---

## 4. Fair Comparison & Integrity Safeguards

To prevent confirmation bias or spurious claims of diffusion superiority:
1. **Identical Metric Code:** All metrics must be computed with the identical `compute_metrics()` implementation from [`evaluate_phase4_3.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/evaluate_phase4_3.py), which enforces `channel_axis=0`, `data_range=1.0`, and exact clamping to $[0.0, 1.0]$.
2. **Fixed Random Seeds:** Training and evaluation will enforce `torch.manual_seed(42)` and `np.random.seed(42)`.
3. **No Metric Cherry-Picking:** All metrics (PSNR, SSIM, $L_1$) must be reported in full. If diffusion improves perceptual texture but slightly reduces PSNR due to subtle stochastic noise, this must be explicitly documented and analyzed under the perception-distortion trade-off.
4. **Permanent Test Split Quarantine:** Under no circumstances will the 75 held-out test scenes be loaded or evaluated during Experiments A through F.
