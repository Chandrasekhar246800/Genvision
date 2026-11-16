# GenVision Phase 4.3.2: Scientific Reporting Corrections & Evaluation Freeze Report

**Project:** GenVision — A Cloud Severity-Aware Transformer-Diffusion Framework for Cloud-Free Satellite Image Reconstruction and Intelligent Agricultural Analysis  
**Repository Root:** `C:\Users\mcsr8\Desktop\cloud-removal-poc`  
**Date:** 16 November 2026  
**Auditor:** Senior Computer Vision Research Engineer & Scientific Reproducibility Auditor  
**Milestone:** Phase 4.3.2 (Corrections-Only Audit & Evaluation Freeze)  
**Status:** **`COMPLETE`**  

---

## 1. Executive Summary & Audit Mandate

In accordance with Phase 4.3.2 requirements, a strict, corrections-only scientific audit of the Phase 4.3 Held-Out Evaluation Report ([`PHASE4_3_HELD_OUT_EVALUATION_REPORT.md`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/PHASE4_3_HELD_OUT_EVALUATION_REPORT.md)) and the Phase 4.3.1 Reproducibility Audit Report ([`PHASE4_3_1_REPRODUCIBILITY_AUDIT.md`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/PHASE4_3_1_REPRODUCIBILITY_AUDIT.md)) was executed.

### Core Safeguards Maintained
- **Zero Checkpoint Alteration:** All model weight binaries ([`unet_controlled_best.pth`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/unet_controlled_best.pth), [`attention_controlled_best.pth`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/attention_controlled_best.pth), [`transformer_controlled_best.pth`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/transformer_controlled_best.pth)) remain bit-for-bit identical with their verified hashes and creation timestamps.
- **Zero Dataset Split Modification:** The 350 train / 75 validation / 75 test partition defined in [`results/data_split.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/data_split.json) remains strictly frozen.
- **Zero Metric Tampering:** The empirical per-image evaluation results in [`results/phase4/evaluation/per_image_metrics.csv`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/per_image_metrics.csv) and [`model_comparison.csv`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/model_comparison.csv) remain completely untouched and frozen.
- **Zero Retraining or Test-Tuning:** No model was retrained, fine-tuned, or adapted against test split performance.
- **Phase 5 Deferred:** Diffusion framework design and training remain uninitiated.

---

## 2. Itemized Scientific Issues & Corrections

### Issue 1: SSIM Window Implementation & Weighting Kernel Specification

#### 1. Context & Code-Level Inspection
The evaluation script [`evaluate_phase4_3.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/evaluate_phase4_3.py) invokes structural similarity via:
```python
ssim = float(structural_similarity(target_np, pred_np, channel_axis=0, data_range=1.0))
```
Inspection of the installed library environment (`scikit-image` version `0.26.0`, in [`venv/Lib/site-packages/skimage/metrics/_structural_similarity.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/venv/Lib/site-packages/skimage/metrics/_structural_similarity.py)) reveals the following exact runtime parameters:

| Parameter | Passed Value | Default in skimage 0.26.0 | Active Execution Behavior |
| :--- | :---: | :---: | :--- |
| `channel_axis` | `0` | `None` | Evaluates across channels at axis 0 corresponding to PyTorch `(C, H, W)` shape. |
| `data_range` | `1.0` | `None` | Normalized dynamic range of floating-point imagery $[0.0, 1.0]$. |
| `win_size` | `None` (unspecified) | `None` $\rightarrow$ `7` | Under `gaussian_weights=False`, defaults to an odd integer window of size **$7 \times 7$**. |
| `gaussian_weights` | `False` (unspecified) | `False` | **Uniform box filter** via `scipy.ndimage.uniform_filter`. A Gaussian kernel is **not** used. |
| `use_sample_covariance` | `True` (unspecified) | `True` | Uses sample covariance normalizer $\frac{NP}{NP - 1} = \frac{49}{48} \approx 1.020833$, not population covariance ($1.0$). |
| $K_1, K_2$ | `0.01, 0.03` | `0.01, 0.03` | Standard algorithm stability constants. |
| $C_1, C_2$ | Calculated | $(K_i \cdot \text{data\_range})^2$ | $C_1 = (0.01 \times 1.0)^2 = \mathbf{1.0 \times 10^{-4}}$, $C_2 = (0.03 \times 1.0)^2 = \mathbf{9.0 \times 10^{-4}}$. |

#### 2. Original Statement (Phase 4.3 Report, Section 3)
> *"Computed via `skimage.metrics.structural_similarity` with `channel_axis=0`, Gaussian window size $7 \times 7$, and `data_range=1.0`."*

#### 3. Corrected Statement
> *"Computed via `skimage.metrics.structural_similarity` with `channel_axis=0`, uniform box window of size $7 \times 7$ (`win_size=7`, `gaussian_weights=False`, `use_sample_covariance=True`), algorithm constants $K_1 = 0.01, K_2 = 0.03$, stabilization constants $C_1 = 10^{-4}, C_2 = 9 \times 10^{-4}$, and `data_range=1.0`."*

#### 4. Source Code Evidence
- File: [`evaluate_phase4_3.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/evaluate_phase4_3.py#L141)
- Library Source: `skimage.metrics._structural_similarity:structural_similarity` lines 112–136:
  ```python
  if win_size is None:
      if gaussian_weights:
          ...
      else:
          win_size = 7  # backwards compatibility
  ...
  if gaussian_weights:
      filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}
  else:
      filter_args = {'size': win_size}  # ndimage.uniform_filter
  ```

#### 5. Impact on Existing Results
**Zero numerical impact.** The original evaluation and the independent verification both used this exact function call. The metrics reported in [`per_image_metrics.csv`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/per_image_metrics.csv) were computed using this exact implementation. Only the narrative description of the weighting kernel has been corrected.

---

### Issue 2: Scientific Terminology for Reproducibility Verification

#### 1. Context & Scope Clarification
In scientific computing and remote sensing benchmarks, it is essential to distinguish between the full experimental run and a sampled verification audit:
- The **original held-out evaluation** evaluated all **75 paired test scenes** from [`results/data_split.json`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/data_split.json).
- The **independent reproducibility audit** conducted in Phase 4.3.1 executed fresh forward passes on a stratified sample of **5 representative test scenes** (`187.png`, `257.png`, `5.png`, `111.png`, `393.png`) selected across the PSNR difficulty range.
- The comparison between the freshly computed metrics and the pre-existing CSV records demonstrated numerical agreement within floating-point tolerance ($< 10^{-6}$).

#### 2. Original Statement (Phase 4.3.1 Report, Executive Summary)
> *"1. Reconstruction & Benchmark Reproducibility: 100% Bit-Exact Reproducibility. An independent verification pass executing on 5 representative test scenes across all three models confirmed that per-image PSNR, SSIM, and $L_1$ reconstruction errors in results/phase4/evaluation/per_image_metrics.csv match the live model outputs with absolute difference $< 10^{-6}$ (floating-point epsilon)."*

#### 3. Corrected Statement
> *"1. Reconstruction & Benchmark Reproducibility: Verified numerical reproducibility within floating-point tolerance ($< 10^{-6}$). While the full benchmark was executed on all 75 held-out scenes, an independent recalculation was conducted on a stratified subset of five representative test scenes across all three models. In every sampled case, newly generated inference outputs produced PSNR, SSIM, and $L_1$ metrics matching the recorded CSV values within an absolute threshold of $10^{-6}$."*

#### 4. Scientific Justification
Universal claims of "100% bit-exact reproducibility" across a 75-scene dataset cannot be stated when the recalculation sampled five scenes, nor should floating-point agreement within $10^{-6}$ be labeled as "bit-exact" (which strictly implies identical IEEE 754 bit representations). The corrected wording is scientifically precise and accurate.

---

### Issue 3: Error-Map Normalization & Cross-Figure Scaling Semantics

#### 1. Context & Code-Level Inspection
Visual heatmaps generated in [`results/phase4/evaluation/visualizations/`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/visualizations) depict pixel-wise mean absolute error ($L_1$) between reconstructed RGB outputs and ground-truth images.

In [`evaluate_phase4_3.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/evaluate_phase4_3.py#L225-L244):
```python
# Absolute error maps per model
err_u = torch.abs(preds_t["unet"] - clean_t).mean(dim=0).cpu().numpy()
err_a = torch.abs(preds_t["attention_unet"] - clean_t).mean(dim=0).cpu().numpy()
err_cs = torch.abs(preds_t["cs_transunet"] - clean_t).mean(dim=0).cpu().numpy()

# Fixed error scale for fair comparison within the panel
vmax = max(float(err_u.max()), float(err_a.max()), float(err_cs.max()), 0.25)
vmax = min(vmax, 0.5)

im1 = axes[1, 1].imshow(err_u, cmap="inferno", vmin=0.0, vmax=vmax)
im2 = axes[1, 2].imshow(err_a, cmap="inferno", vmin=0.0, vmax=vmax)
im3 = axes[1, 3].imshow(err_cs, cmap="inferno", vmin=0.0, vmax=vmax)
```

#### 2. Normalization Formulation & Scope
- **Formula:**
  $$v_{\max} = \min\left(\max\left(\max(\text{err}_U, \text{err}_A, \text{err}_{CS}), 0.25\right), 0.50\right)$$
- **Scale Bounds:** Clamped with a floor of $0.25$ (to prevent microscopic residual noise in thin-haze scenes from being artificially saturated) and a ceiling of $0.50$ (to prevent extreme single-pixel outliers from washing out mid-range error structure).
- **Intra-Figure vs Inter-Figure Scope:**
  - **Within each figure:** All three models (U-Net, Attention U-Net, CS-TransUNet) share the **exact same color scale** ($v_{\min} = 0.0, v_{\max} = v_{\max}$) and reference a single shared colorbar.
  - **Across different figures:** Each scene dynamically calculates its own $v_{\max} \in [0.25, 0.50]$ based on the local error dynamics of that test image.
  - **Clarification:** The visualizations do **not** share a single invariant global color scale across different figures. Cross-scene comparisons must reference the colorbar rendered in each individual panel.

#### 3. Corrected Narrative
> *"In all visual comparison panels, the three competing models strictly share an identical intra-figure color scale ($v_{\min} = 0.0$ to $v_{\max} = \min(\max(\max(\text{err}), 0.25), 0.50)$) with a shared colorbar. The scale adapts dynamically per scene to accommodate varying cloud opacities, meaning $v_{\max}$ is consistent within any given figure but varies between figures."*

---

### Issue 4: CSEB Architectural Role & Classification Target Clarification

#### 1. Context & Code-Level Inspection
The Cloud Severity Estimation Branch (CSEB) is defined in [`models/transformer/severity_estimator.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/transformer/severity_estimator.py) and integrated into [`models/transformer/cs_transunet.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/transformer/cs_transunet.py).

In [`train_transformer.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/train_transformer.py#L347-L442):
```python
criterion = nn.L1Loss()
...
reconstructed, sev_logits = model(cloudy, return_dict=False) # or sev_embedding
loss = criterion(reconstructed, clean)
loss.backward()
```

#### 2. Architectural Verification Findings
1. **Absence of Auxiliary Classification Loss:** `criterion` is strictly `nn.L1Loss()` between reconstructed and clean RGB imagery. The classification logits (`severity_logits`) and softmax probabilities (`severity_probabilities`) generated by `CSEB.classifier` are **never** passed into any loss function (e.g., `nn.CrossEntropyLoss()`).
2. **Absence of Ground-Truth Severity Labels:** The dataset ([`dataset/cloudy/`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/dataset/cloudy/), [`dataset/clean/`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/dataset/clean/)) does not contain thin/moderate/thick category annotations.
3. **Absence of Contrastive / Self-Supervised Objectives:** No contrastive loss (e.g. InfoNCE) or masked prediction objective was applied.
4. **Actual Function:** The branch produces a $192\text{-dimensional}$ continuous feature vector (`severity_embedding`) via adaptive pooling and MLP layers, which modulates the Stage 3 and Stage 4 transformer blocks via Adaptive Layer Normalization (AdaLN). Gradients flow exclusively from the end-to-end image reconstruction loss.

#### 3. Corrected Terminology & Scientific Boundaries
- **Approved Terminology:** **"Unsupervised self-conditioned severity representation."**
- **Disallowed Claims:** No claims of "verified thin/moderate/thick cloud classification accuracy", "cloud classification performance", or "self-supervised classification" may be made. The 3-class logits/probabilities output by the model are uncalibrated and ungrounded.

#### 4. Source Code Evidence
- [`models/transformer/severity_estimator.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/transformer/severity_estimator.py#L23-L68)
- [`models/transformer/cs_transunet.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/transformer/cs_transunet.py#L69-L74)
- [`train_transformer.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/train_transformer.py#L347, #L440)

---

### Issue 5: SSIM Performance Distribution & Narrative Precision

#### 1. Original Statement (Phase 4.3 Report, Section 7)
> *"SSIM Distribution: results/phase4/evaluation/test_ssim_boxplot.png (Demonstrates that CS-TransUNet consistently operates above 0.90 SSIM with minimal outliers)."*

#### 2. Empirical Verification
Analysis of all 75 held-out scenes from [`results/phase4/evaluation/per_image_metrics.csv`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/per_image_metrics.csv):
- **Total Test Scenes:** 75
- **Mean:** $0.935942 \pm 0.030900$
- **Median:** $0.941215$
- **25th Percentile ($Q_1$):** $0.919918$
- **75th Percentile ($Q_3$):** $0.959183$
- **Maximum:** $0.986629$
- **Empirical Minimum:** **$0.840625$**
- **Scenes with SSIM $\ge 0.90$:** **66 out of 75 (88.00%)**
- **Scenes with SSIM $< 0.90$:** **9 out of 75 (12.00%)**

#### 3. Corrected Statement (Implemented in Phase 4.3 Report)
> *"SSIM Distribution: [`results/phase4/evaluation/test_ssim_boxplot.png`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/evaluation/test_ssim_boxplot.png) (Demonstrates that CS-TransUNet achieves $\ge 0.90$ SSIM on 88.00% (66/75) of held-out scenes with median 0.9412 and an empirical lower bound of 0.8406 under dense, opaque cloud occlusion)."*

---

## 3. Master Summary of Document Modifications

Every documentation update across the repository is formally tracked below:

| File | Target Section | Original Text Summary | Corrected Text Summary | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| [`results/phase4/PHASE4_3_HELD_OUT_EVALUATION_REPORT.md`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/PHASE4_3_HELD_OUT_EVALUATION_REPORT.md) | Section 3 (Metric Definitions, Line 86) | Cited "Gaussian window size $7 \times 7$". | Corrected to "uniform box window size $7 \times 7$ (`win_size=7`, `gaussian_weights=False`, `use_sample_covariance=True`, $C_1 = 10^{-4}, C_2 = 9 \times 10^{-4}$)". | Accurately describes skimage 0.26.0 default uniform filter execution. |
| [`results/phase4/PHASE4_3_HELD_OUT_EVALUATION_REPORT.md`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/PHASE4_3_HELD_OUT_EVALUATION_REPORT.md) | Section 7 (Boxplots, Line 180) | Cited "consistently operates above 0.90 SSIM with minimal outliers". | Corrected to cite 88.00% (66/75) $\ge 0.90$ SSIM with empirical minimum of $0.8406$. | Aligns report narrative with empirical per-image CSV distribution. |
| [`results/phase4/PHASE4_3_HELD_OUT_EVALUATION_REPORT.md`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/PHASE4_3_HELD_OUT_EVALUATION_REPORT.md) | Section 9 (Caveats, Line 200) | Colloquial description of CSEB self-conditioning. | Formalized as "CSEB Unsupervised Self-Conditioned Severity Representation" noting uncalibrated logits. | Prevents misinterpretation of CSEB as a supervised classifier. |
| [`results/phase4/PHASE4_3_1_REPRODUCIBILITY_AUDIT.md`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/PHASE4_3_1_REPRODUCIBILITY_AUDIT.md) | Entire Document | Original audit findings from Phase 4.3.1. | **Preserved intact as historical audit record.** | Ensures transparency and maintains audit trail integrity. |
| [`results/phase4/PHASE4_3_2_SCIENTIFIC_CORRECTIONS.md`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/results/phase4/PHASE4_3_2_SCIENTIFIC_CORRECTIONS.md) | New Document | N/A | **Comprehensive corrections & evaluation freeze specification.** | Fulfills Phase 4.3.2 milestone requirements. |

---

## 4. Evaluation Freeze Status & Verification

All evaluation infrastructure, checkpoints, and benchmark datasets are now officially placed under **Permanent Evaluation Freeze**:

```
================================================================================
GENVISION EVALUATION FREEZE CERTIFICATE
================================================================================
Timestamp:               2026-11-16T15:46:00+05:30
Partition File:          results/data_split.json (Frozen: Seed 42, 350/75/75)
U-Net Checkpoint:        models/unet_controlled_best.pth (Frozen: 30,803,417 bytes)
Attn U-Net Checkpoint:   models/attention_controlled_best.pth (Frozen: 125,682,635 bytes)
CS-TransUNet Checkpoint: models/transformer_controlled_best.pth (Frozen: 85,688,669 bytes, Epoch 24)
Per-Image Metrics CSV:   results/phase4/evaluation/per_image_metrics.csv (Frozen: 75 scenes x 3 models)
Model Comparison CSV:    results/phase4/evaluation/model_comparison.csv (Frozen: 75 pairwise diffs)
Evaluation Summary JSON: results/phase4/evaluation/evaluation_summary.json (Frozen)
================================================================================
```

### Verification Checklist
- [x] Checkpoint SHA/size integrity verified.
- [x] Test split never exposed to training or hyperparameter tuning.
- [x] Metric CSV files preserved without modification.
- [x] All 75 held-out scenes evaluated under identical preprocessing.
- [x] Statistical tests (t-test, Wilcoxon, Bonferroni) verified.
- [x] Reporting text corrected and aligned with underlying implementations.

---

## 5. Remaining Scientific Limitations

While the Phase 4.3 benchmark confirms that CS-TransUNet significantly outperforms U-Net and Attention U-Net baselines, the following limitations remain documented for subsequent research:

1. **Synthetic Cloud Augmentation:**  
   The held-out benchmark pairs consist of synthetic cloud augmentations overlaid on clean Sentinel RGB scenes. Downstream validation on real-world Level-2A multi-temporal Sentinel-2 imagery (as established in Phase 3.2 for the Pulivendula ROI) will be required to assess real-world generalization.
2. **Unsupervised Severity Latent Guidance:**  
   The Cloud Severity Estimation Branch operates without ground-truth severity supervision. While this provides useful data-driven modulation through AdaLN, multi-task training with explicit SCL (Scene Classification Layer) cloud probabilities could further improve conditioning fidelity.
3. **Single Seed Split:**  
   Results are reported on a single fixed 70/15/15 partition (Seed 42). While sample size ($n=75$) is sufficient for high statistical power ($p \ll 0.001$), multi-fold cross-validation across diverse geographic biomes remains an opportunity for broader validation.

---

## 6. Final Status Determination

The Phase 4.3.2 corrections-only audit is complete. All reporting inaccuracies regarding SSIM kernel weighting, reproducibility scope, error-map scaling, and CSEB terminology have been resolved without altering underlying numerical data or model checkpoints.

# **FINAL STATUS: `COMPLETE`**
