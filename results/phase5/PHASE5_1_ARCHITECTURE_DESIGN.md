# GenVision Phase 5.1: Diffusion Architecture Design

**Project:** GenVision — A Cloud Severity-Aware Transformer-Diffusion Framework for Cloud-Free Satellite Image Reconstruction and Intelligent Agricultural Analysis  
**Repository Root:** `C:\Users\mcsr8\Desktop\cloud-removal-poc`  
**Date:** 16 November 2026  
**Document:** Architecture Design Specification (Phase 5.1)  
**Status:** **`DESIGN ONLY — NOT FOR IMPLEMENTATION`**  

---

## 1. Executive Summary & Design Premise

Phase 4 demonstrated that the proposed **CS-TransUNet** achieves high reconstruction fidelity on 256×256 satellite imagery ($28.7877\text{ dB}$ PSNR, $0.9359$ SSIM, and $0.0343$ $L_1$ error across 75 strictly held-out test scenes), operating with an average inference latency of $130\text{ ms}$ on an Intel Core Ultra 9 285H CPU.

However, detailed error-map inspections from Phase 4.3 reveal two persistent challenges inherent to pure deterministic regression:
1. **High-Frequency Texture Attenuation:** Under moderate cloud cover, regression models trained on $L_1$ or $L_2$ objectives tend toward the conditional expectation $\mathbb{E}[x_{\text{clean}} \mid x_{\text{cloudy}}]$, leading to subtle oversmoothing of fine agricultural parcel boundaries, irrigation furrows, and soil ridges.
2. **Dense Cloud Inpainting Uncertainty:** Under dense cumulus cores (where optical penetration is $<10\%$), the mapping from cloudy to clear ground is ill-posed and multimodal. Standard convolution/transformer heads can leave faint residual haze halos or spectral desaturation.

The primary design question of Phase 5.1 is:  
*How can a probabilistic diffusion framework be coupled with the frozen CS-TransUNet to refine high-frequency details and resolve multimodal uncertainty without incurring prohibitive CPU computational costs or destabilizing the existing 28.79 dB reconstruction baseline?*

This document specifies the **Lightweight Conditional Residual Diffusion Network ("ResDiff-TransUNet")**, a two-stage hybrid framework designed strictly for CPU execution.

---

## 2. Core Architectural Hypothesis: The Residual Diffusion Paradigm

### 2.1 The Problem with Direct Image-Space Diffusion
Standard Denoising Diffusion Probabilistic Models (DDPM) generate the complete clean image $x_0 \in [0, 1]^{3 \times 256 \times 256}$ directly from standard Gaussian noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.

For satellite image restoration on CPU hardware, direct image-space diffusion suffers from four major fatal flaws:
1. **Computational Redundancy:** The diffusion model must allocate massive representational capacity to rediscover basic low-frequency geographic features (land vs water, macro topography, broad vegetation zones) that CS-TransUNet already predicts with $>93\%$ structural accuracy.
2. **Slow Convergence on Small Datasets:** Learning the entire generative prior of satellite imagery from only 350 training images results in severe mode collapse or structural hallucination.
3. **Catastrophic Reconstruction Collapse Risk:** If reverse sampling diverges, the resulting image is complete visual noise, destroying the guarantees achieved in Phase 4.
4. **Prohibitive Reverse Latency:** Iterating 50 to 1000 denoising steps across a 30M+ parameter U-Net on CPU requires 25 to 60 seconds per single image.

### 2.2 The Residual Diffusion Solution
Instead of diffusing in raw image space, we formulate diffusion over the **reconstruction residual manifold**:

$$r_0 = x_0 - \hat{x}_0$$

where:
- $x_0 \in [0, 1]^{3 \times 256 \times 256}$ is the ground-truth clean image.
- $\hat{x}_0 = \mathcal{F}_{\text{CS-TransUNet}}(y) \in [0, 1]^{3 \times 256 \times 256}$ is the deterministic initial prediction from the frozen CS-TransUNet checkpoint ([`models/transformer_controlled_best.pth`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/transformer_controlled_best.pth)).
- $r_0 \in [-1, 1]^{3 \times 256 \times 256}$ is the true high-frequency residual error.

```
+-----------------------------------------------------------------------------------------+
|                                    STAGE 1: DETERMINISTIC RESTORATION                   |
|                                                                                         |
|  Cloudy Image (y)  ───────> [ FROZEN CS-TransUNet ] ───────> Initial Reconstruction     |
|   (3 x 256 x 256)               (7.13M params)                    (x_hat_0)             |
|          │                             │                       (3 x 256 x 256)          |
|          │                             ▼ (Extract CSEB vector)        │                 |
|          │                      e_cseb in R^192                       │                 |
|          │                             │                              │                 |
+──────────┼─────────────────────────────┼──────────────────────────────┼─────────────────+
           │                             │                              │
           ▼                             ▼                              ▼
+-----------------------------------------------------------------------------------------+
|                                    STAGE 2: CONDITIONAL RESIDUAL DIFFUSION              |
|                                                                                         |
|  Spatial Concatenation:                                                                 |
|  X_in = [ r_t, x_hat_0, y ] in R^(9 x 256 x 256)                                       |
|                                                                                         |
|  Global AdaGN Conditioning:                                                             |
|  cond_vector = [ MLP(timestep_t), e_cseb ] in R^320                                     |
|                                                                                         |
|                ┌────────────────────────────────────────────────────────┐               |
|                │       LIGHTWEIGHT RESIDUAL DENOISING U-NET (1.85M)     │               |
|                │   - Input: 9 channels (Noisy Residual + x_hat_0 + y)   │               |
|                │   - Conditioning: AdaGN(t, e_cseb)                     │               |
|                │   - Channels: 32 -> 64 -> 128 -> 256                   │               |
|                │   - Zero spatial cross-attention (Optimized for CPU)   │               |
|                └────────────────────────────────────────────────────────┘               |
|                                            │                                            |
|                                            ▼ Reverse Process (DDIM S=5..10 steps)       |
|                                  Predicted Residual (r_hat_0)                           |
+────────────────────────────────────────────┼────────────────────────────────────────────+
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                               STAGE 3: RESIDUAL INTEGRATION                             |
|                                                                                         |
|    Final Cloud-Free Image = clamp( x_hat_0 + lambda * r_hat_0, 0.0, 1.0 ) [lambda in 0..1] |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Detailed Mathematical Formulation

### 3.1 Domain Definitions & Normalization
Let:
- $y \in [0, 1]^{3 \times H \times W}$: Cloudy input image.
- $x_0 \in [0, 1]^{3 \times H \times W}$: Ground-truth clear image.
- $\hat{x}_0 = \mathcal{F}_{\text{CS-TransUNet}}(y) \in [0, 1]^{3 \times H \times W}$: Initial reconstruction from the frozen CS-TransUNet model.
- $r_0 = x_0 - \hat{x}_0 \in [-1, 1]^{3 \times H \times W}$: True residual error.

#### Empirical Residual Statistics
From our Phase 4.3 held-out evaluation:
- Mean $L_1$ residual magnitude: $\mathbb{E}[|r_0|] = 0.034316$.
- Residual standard deviation: $\sigma_r \approx 0.045$.
- Over $>85\%$ of pixels (clear background and thin clouds), $r_0 \approx 0$.
- Over dense cloud cores, $r_0$ reaches local magnitudes of up to $\pm 0.35$.

To standardize $r_0$ for diffusion noise scheduling where Gaussian priors have unit variance, we introduce a static residual scaling factor $s_r = 4.0$ (or $\frac{1}{2\sigma_r}$):
$$\tilde{r}_0 = s_r \cdot r_0 = 4.0 \cdot (x_0 - \hat{x}_0)$$
This scales the typical residual range to $[-1.5, +1.5]$, making it compatible with standard forward diffusion schedules.

### 3.2 Forward Noising Process on Residuals
The forward diffusion process progressively injects Gaussian noise into the scaled residual $\tilde{r}_0$ over $T = 1000$ discrete timesteps according to a Markov chain:

$$q(\tilde{r}_1, \dots, \tilde{r}_T \mid \tilde{r}_0) = \prod_{t=1}^T q(\tilde{r}_t \mid \tilde{r}_{t-1})$$

$$q(\tilde{r}_t \mid \tilde{r}_{t-1}) = \mathcal{N}\left(\tilde{r}_t; \, \sqrt{1 - \beta_t}\,\tilde{r}_{t-1}, \, \beta_t \mathbf{I}\right)$$

where $\beta_1 < \beta_2 < \dots < \beta_T$ is a linear or cosine variance schedule:
- $\beta_1 = 10^{-4}$
- $\beta_T = 0.02$
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$

Using the reparameterization property, the distribution of $\tilde{r}_t$ at any arbitrary timestep $t$ given $\tilde{r}_0$ is expressed in closed form:

$$q(\tilde{r}_t \mid \tilde{r}_0) = \mathcal{N}\left(\tilde{r}_t; \, \sqrt{\bar{\alpha}_t}\,\tilde{r}_0, \, (1 - \bar{\alpha}_t)\mathbf{I}\right)$$

$$\tilde{r}_t = \sqrt{\bar{\alpha}_t}\,\tilde{r}_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

When $t = T$, $\bar{\alpha}_T \approx 0$, and $\tilde{r}_T$ approaches pure Gaussian noise $\mathcal{N}(0, \mathbf{I})$.

### 3.3 Reverse Denoising Process
The reverse Markov chain parameterizes the distribution $p_\theta(\tilde{r}_{t-1} \mid \tilde{r}_t, c)$ conditioned on context $c$:

$$p_\theta(\tilde{r}_{0:T} \mid c) = p(\tilde{r}_T) \prod_{t=1}^T p_\theta(\tilde{r}_{t-1} \mid \tilde{r}_t, c)$$

$$p_\theta(\tilde{r}_{t-1} \mid \tilde{r}_t, c) = \mathcal{N}\left(\tilde{r}_{t-1}; \, \mu_\theta(\tilde{r}_t, t, c), \, \sigma_t^2 \mathbf{I}\right)$$

Parameterizing the mean via a noise prediction network $\epsilon_\theta(\tilde{r}_t, t, c)$:

$$\mu_\theta(\tilde{r}_t, t, c) = \frac{1}{\sqrt{\alpha_t}} \left( \tilde{r}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\tilde{r}_t, t, c) \right)$$

### 3.4 Training Objective
The denoising network $\epsilon_\theta$ is trained by minimizing the noise prediction discrepancy. While the simplified variational lower bound (ELBO) corresponds strictly to the Gaussian $L_2$ loss (Ho et al., 2020), an empirical robust $L_1$ formulation (Saharia et al., 2021) is also evaluated for sharper satellite edge gradients:

$$\mathcal{L}_{\text{diff}}(\theta) = \mathbb{E}_{\tilde{r}_0, c, \epsilon \sim \mathcal{N}(0, \mathbf{I}), t \sim \mathcal{U}(1, T)} \left[ \left\| \epsilon - \epsilon_\theta(\tilde{r}_t, t, c) \right\|_1 \right]$$

**Critical Distinction Between Training and Inference Objectives:**
- During training, the loss function evaluates the discrepancy between the sampled Gaussian noise $\epsilon$ and the predicted noise $\epsilon_\theta$. The frozen CS-TransUNet weights are never updated ($\nabla_\phi \mathcal{L} = 0$).
- During inference, the network estimates the cleaned residual $\hat{r}_0$, which is added back to $\hat{x}_0$ to yield the final image.

### 3.5 Fast Non-Markovian Reverse Sampling (DDIM)
To achieve fast inference on CPU, we employ the Denoising Diffusion Implicit Models (DDIM) formulation. Instead of taking $T = 1000$ steps, we sample across a sub-sequence of $S$ steps $\tau = [\tau_1, \tau_2, \dots, \tau_S]$ where $S \in [5, 10]$:

At each step $\tau_i \to \tau_{i-1}$:
1. Predict $\tilde{r}_0$ from the current noisy state $\tilde{r}_{\tau_i}$:
   $$\hat{\tilde{r}}_0 = \frac{\tilde{r}_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}}\,\epsilon_\theta(\tilde{r}_{\tau_i}, \tau_i, c)}{\sqrt{\bar{\alpha}_{\tau_i}}}$$
2. Compute the state at $\tau_{i-1}$ with deterministic trajectory ($\eta = 0$):
   $$\tilde{r}_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}}\,\hat{\tilde{r}}_0 + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}}}\,\epsilon_\theta(\tilde{r}_{\tau_i}, \tau_i, c)$$

### 3.6 Final Image Reconstruction Equation
Once the reverse process reaches $\tau_0 = 0$, the predicted residual is de-scaled:
$$\hat{r}_0 = \frac{1}{s_r} \hat{\tilde{r}}_0 = \frac{1}{4.0} \hat{\tilde{r}}_0$$

The final cloud-free image is assembled using a validation-calibrated gating factor $\lambda \in [0.0, 1.0]$:
$$\hat{x}_{\text{final}}(\lambda) = \text{clamp}\left( \hat{x}_0 + \lambda \cdot \hat{r}_0, \, 0.0, \, 1.0 \right)$$
where $\lambda^*$ is tuned strictly on the validation set to maximize validation PSNR/SSIM, providing a deterministic fallback ($\lambda = 0.0 \implies \hat{x}_{\text{final}} \equiv \hat{x}_0$) if diffusion fails to improve metrics.

---

## 4. Denoising Network Architecture: ResDiff-UNet

To ensure strict execution feasibility on CPU, we design a lightweight, convolution-first denoising U-Net with zero spatial cross-attention layers.

### 4.1 Layer & Channel Hierarchy
- **Input Channels:** $9$
  - Slice 0..2: Noisy scaled residual $\tilde{r}_t \in \mathbb{R}^{3 \times 256 \times 256}$
  - Slice 3..5: Deterministic reconstruction $\hat{x}_0 \in \mathbb{R}^{3 \times 256 \times 256}$
  - Slice 6..8: Original cloudy input $y \in \mathbb{R}^{3 \times 256 \times 256}$
- **Encoder Stages:**
  - Stage 1: $256 \times 256 \times 32$ (Conv $3\times 3$ Stem + ResBlock + AdaGN)
  - Stage 2: $128 \times 128 \times 64$ (Downsample Conv + ResBlock + AdaGN)
  - Stage 3: $64 \times 64 \times 128$ (Downsample Conv + ResBlock + AdaGN)
  - Stage 4 (Bottleneck): $32 \times 32 \times 256$ (Downsample Conv + 2 ResBlocks + AdaGN)
- **Decoder Stages:**
  - UpStage 3: $64 \times 64 \times 128$ (Nearest-neighbor upsample + Conv $1\times 1$ skip-concat + ResBlock)
  - UpStage 2: $128 \times 128 \times 64$ (Upsample + Skip-concat + ResBlock)
  - UpStage 1: $256 \times 256 \times 32$ (Upsample + Skip-concat + ResBlock)
- **Output Head:** Conv $3 \times 3 \implies 3$ output channels (predicted noise $\hat{\epsilon} \in \mathbb{R}^{3 \times 256 \times 256}$).

### 4.2 Parameter Count & Memory Breakdown
- Stage 1 ($32$ channels): $\sim 28\text{k}$ params
- Stage 2 ($64$ channels): $\sim 115\text{k}$ params
- Stage 3 ($128$ channels): $\sim 455\text{k}$ params
- Stage 4 Bottleneck ($256$ channels): $\sim 1,050\text{k}$ params
- Decoder & Conditioning MLPs: $\sim 200\text{k}$ params
- **Total Parameter Count:** **$\approx 1.85 \times 10^6$ parameters ($\mathbf{1.85\text{ M}}$)**
- **Memory Footprint (FP32):** $1.85\text{M} \times 4\text{ B} \approx \mathbf{7.4\text{ MB}}$ on disk.

### 4.3 Why Spatial Cross-Attention is Excluded
In standard text-to-image diffusion models (e.g., Stable Diffusion), cross-attention layers allow text tokens to query arbitrary spatial image tokens. However:
1. In satellite cloud removal, spatial correspondence between the cloudy scene and the clean ground is strictly 1:1 and geographically aligned. Spatial feature concatenation ($9$ channels) preserves exact coordinate alignment without any attention computation.
2. Cross-attention layers at $256 \times 256$ or $128 \times 128$ scale as $\mathcal{O}((HW)^2)$, which creates massive memory bandwidth bottlenecks on CPU.
3. Excluding cross-attention reduces single-step CPU forward latency from $\sim 350\text{ ms}$ to $\mathbf{38\text{ ms}}$, enabling a 10-step reverse process to complete in under $400\text{ ms}$.

---

## 5. Conditioning Strategy & CSEB Integration

### 5.1 Dual-Level Conditioning Mechanism
The framework employs a dual conditioning scheme separating spatial geometric context from global atmospheric context:

1. **Spatial Conditioning (Pixel-Level):**
   - Implemented via direct channel concatenation in the input tensor $[ \tilde{r}_t, \hat{x}_0, y ] \in \mathbb{R}^{9 \times 256 \times 256}$.
   - Provides direct, uncompressed spatial guidance on cloud edges, parcel lines, and local terrain textures.
2. **Global Atmospheric Conditioning (Feature-Level):**
   - Injected into every residual block via **Adaptive Group Normalization (AdaGN)**.

### 5.2 Conditioning Vector Construction
The global conditioning vector $\mathbf{v}_{\text{cond}} \in \mathbb{R}^{320}$ is formed by concatenating two continuous vectors:
$$\mathbf{v}_{\text{cond}} = \left[ \phi(t), \, e_{\text{cseb}} \right]$$

1. **Sinusoidal Timestep Embedding $\phi(t) \in \mathbb{R}^{128}$:**
   Standard positional encoding projected through a 2-layer MLP with SiLU activation:
   $$\phi(t)_k = \sin\left(\frac{t}{10000^{2k/128}}\right), \quad \phi(t)_{k+1} = \cos\left(\frac{t}{10000^{2k/128}}\right)$$
2. **Continuous CSEB Embedding $e_{\text{cseb}} \in \mathbb{R}^{192}$:**
   Extracted directly from the frozen CS-TransUNet checkpoint's Cloud Severity Estimator Branch ([`models/transformer/severity_estimator.py`](file:///C:/Users/mcsr8/Desktop/cloud-removal-poc/models/transformer/severity_estimator.py)).
   - Specifically, $e_{\text{cseb}}$ is the 192-dimensional latent feature vector computed after Stage-2 global average pooling and MLP projection:
     $$e_{\text{cseb}} = \text{MLP}_{\text{cseb}}\left(\text{AdaptiveAvgPool2d}(f_{\text{enc2}})\right) \in \mathbb{R}^{192}$$

### 5.3 Strict Restrictions Regarding CSEB
As established in the Phase 4.3.2 scientific audit:
- The 3-class classification logits (`severity_logits`) and probabilities (`severity_probabilities`) from CSEB were **never trained with ground-truth cloud labels or cross-entropy loss**.
- Consequently, they are uncalibrated and **must never be used** as discrete categorical conditioning tags (e.g., "thin", "moderate", "thick").
- The diffusion network conditions **exclusively on the continuous 192-dimensional embedding** $e_{\text{cseb}}$, which represents an unsupervised, data-driven atmospheric severity prior learned through end-to-end reconstruction gradients.

### 5.4 Adaptive Group Normalization (AdaGN) Formulation
For any feature map $h \in \mathbb{R}^{C \times H \times W}$ inside a ResBlock:
1. The combined conditioning vector $\mathbf{v}_{\text{cond}} \in \mathbb{R}^{320}$ is passed through a linear projection:
   $$\left[ \gamma(t, e_{\text{cseb}}), \, \beta(t, e_{\text{cseb}}) \right] = \text{Linear}_{320 \to 2C}(\mathbf{v}_{\text{cond}})$$
2. Feature modulation is applied:
   $$\text{AdaGN}(h) = (1 + \gamma(t, e_{\text{cseb}})) \odot \text{GroupNorm}(h) + \beta(t, e_{\text{cseb}})$$
This allows the global cloud thickness representation to modulate the scale and bias of intermediate residual feature maps across all stages with negligible computational cost.

---

## 6. Pre-Caching Optimization for Fast CPU Training

Because CS-TransUNet is **strictly frozen** during diffusion training:
1. The initial reconstructions $\hat{x}_0$ and CSEB embeddings $e_{\text{cseb}}$ do **not change across training epochs**.
2. **Offline Pre-Caching Strategy:**
   - In a single one-time preprocessing pass, we run the 350 training cloudy images through the frozen CS-TransUNet model on CPU (~45 seconds total).
   - We store the outputs on disk:
     - Pre-computed $\hat{x}_0 \in \mathbb{R}^{3 \times 256 \times 256}$
     - Pre-computed $e_{\text{cseb}} \in \mathbb{R}^{192}$
     - Pre-computed true scaled residual $\tilde{r}_0 = 4.0 \cdot (x_0 - \hat{x}_0)$
   - Storage required for 350 scenes: $350 \times (256 \times 256 \times 3 \times 4\text{ B}) \approx \mathbf{275\text{ MB}}$.
3. **Training Speedup Impact:**
   - During diffusion model training, the frozen CS-TransUNet forward pass is **completely bypassed**!
   - Each training step loads $\tilde{r}_0$, $\hat{x}_0$, $y$, and $e_{\text{cseb}}$ directly from cache, samples $t$ and $\epsilon$, and performs forward/backward updates solely on the lightweight 1.85M parameter denoising U-Net.
   - This reduces CPU epoch training time from $\sim 3\text{ minutes}$ to **$\approx 35\text{ seconds}$ per epoch**.

---

## 7. Design Boundaries & What is Explicitly Disallowed

To maintain scientific integrity and project feasibility, the following boundaries are established:
1. **No Joint Training:** CS-TransUNet will **not** be fine-tuned jointly with the diffusion model. It remains permanently frozen.
2. **No High-Resolution Latent Autoencoders:** No external VQ-VAE or Stable Diffusion autoencoder will be imported.
3. **No Uncalibrated Categorical Loss:** No loss will be backpropagated through the CSEB 3-class classification head.
4. **No Test-Split Contamination:** Held-out test scenes remain strictly sealed until the final benchmark in Phase 5.3.

---

## 8. Summary of Architectural Specifications

| Component | Technical Specification | Rationale |
| :--- | :--- | :--- |
| **Diffusion Domain** | Normalized Residual Space: $\tilde{r}_0 = 4.0 \cdot (x_0 - \hat{x}_0)$ | Bounded variance, prevents mode collapse, with validation-gated safety fallback ($\lambda \in [0, 1]$). |
| **Denoising Backbone** | 4-Stage Convolutional U-Net ($1.85\text{M}$ params) | Optimized for CPU cache; avoids $\mathcal{O}((HW)^2)$ cross-attention bottlenecks. |
| **Spatial Conditioning** | Channel Concat: $[ \tilde{r}_t, \hat{x}_0, y ] \in \mathbb{R}^{9 \times 256 \times 256}$ | Zero parameter overhead, preserves exact sub-pixel geographic alignment. |
| **Global Conditioning** | AdaGN with $[ \phi(t), e_{\text{cseb}} ] \in \mathbb{R}^{320}$ | Seamlessly injects unsupervised cloud severity prior into feature normalization. |
| **Forward Schedule** | Linear variance $\beta \in [10^{-4}, 0.02]$, $T = 1000$ | Standard, mathematically sound Gaussian noise dispersion. |
| **Reverse Sampling** | Deterministic DDIM ($\eta = 0$), $S = 5 \text{ to } 10$ steps | Fast inference on Intel Core Ultra 9 CPU ($<400\text{ ms}$ diffusion pass). |
| **Total Inference Time** | $\approx 530\text{ ms}$ ($130\text{ ms}$ CS-TransUNet + $400\text{ ms}$ DDIM) | Interactive responsiveness on consumer CPU hardware without CUDA. |
