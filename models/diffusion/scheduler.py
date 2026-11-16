"""
Gaussian Diffusion Noise Scheduler for GenVision ResDiff-TransUNet
==================================================================

Implements the forward Gaussian diffusion process and variance scheduling
as specified in Phase 5.1 and Phase 5.1.1.

Schedule Configuration:
- Type: Linear beta schedule
- beta_start: 0.0001 (1e-4)
- beta_end: 0.02
- num_timesteps: 1000
- alpha_t = 1.0 - beta_t
- alphas_cumprod = product_{s=0}^t alpha_s

References:
- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
- Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn


class GaussianDiffusionScheduler(nn.Module):
    """
    Device-aware linear variance scheduler for discrete-time Gaussian diffusion.
    Registers precomputed constants as non-trainable persistent buffers.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_timesteps = int(num_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        # 1. Linear beta schedule (T = 1000)
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        # 2. Precomputed tensors for forward noising: q(x_t | x_0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(torch.clamp(1.0 - alphas_cumprod, min=1e-20))
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        # 3. Precomputed tensors for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        )
        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # Register as buffers (saved with model, moves with .to(device), no gradients)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("log_one_minus_alphas_cumprod", log_one_minus_alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    @staticmethod
    def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Extract values from 1D buffer `arr` at 1D indices `timesteps` and reshape
        for broadcasting with shape `broadcast_shape` (e.g. [B, C, H, W]).
        """
        res = arr.to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process q(x_t | x_0).
        Diffuse the data x_start for given timesteps t.
        
        Args:
            x_start: Target tensor [B, C, H, W] (e.g. scaled residual r_tilde_0)
            t: Discrete timesteps tensor [B] with integer values in [0, num_timesteps - 1]
            noise: Optional Gaussian noise tensor [B, C, H, W]. If None, sampled from N(0, I).
            
        Returns:
            Tuple of (x_t, noise) where x_t = sqrt(alpha_bar_t) * x_start + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar = self._extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        x_t = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate x_0 from x_t and predicted noise epsilon:
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        """
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        """
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
