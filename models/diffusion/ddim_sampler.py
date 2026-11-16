"""
Deterministic DDIM Sampler for GenVision ResDiff-TransUNet
==========================================================

Implements Denoising Diffusion Implicit Models (DDIM; Song et al., ICLR 2021)
reverse sampling for fast, deterministic residual inference on CPU.

Mathematical Formulation:
- Forward schedule: T = 1000 timesteps (indexed 0 to 999).
- Sub-sequence selection: S steps tau = [tau_{S-1}, ..., tau_0] spaced across [0, T-1].
- Starting timestep: tau_{S-1} = 999 (ensures maximum forward noise alignment alpha_bar ~ 0.006).
- Deterministic trajectory: eta = 0.0 eliminates stochastic noise injection.
- Dynamic clipping: clip_denoised=True with clip_range=4.0 prevents division-by-alpha_bar explosion.
- Terminal transition: Reaches t_prev = -1 -> returns predicted clean residual r_hat_0.
- Final De-scaling: r_hat_0 = 0.25 * r_tilde_hat_0.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from .scheduler import GaussianDiffusionScheduler


class DDIMSampler:
    """
    Deterministic DDIM reverse sampler for ResDiffUNet.
    """

    def __init__(
        self,
        scheduler: GaussianDiffusionScheduler,
        model: nn.Module,
    ):
        self.scheduler = scheduler
        self.model = model

    @staticmethod
    def compute_timesteps(num_steps: int, total_timesteps: int = 1000) -> List[int]:
        """
        Generate deterministic descending timestep sub-sequence ending at 0.
        
        For num_steps = 10, total_timesteps = 1000:
        Ascending grid: [0, 111, 222, 333, 444, 555, 666, 777, 888, 999]
        Descending reverse sequence: [999, 888, 777, 666, 555, 444, 333, 222, 111, 0]
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if num_steps > total_timesteps:
            raise ValueError(f"num_steps ({num_steps}) exceeds total_timesteps ({total_timesteps})")

        # Linearly spaced integers across [0, total_timesteps - 1]
        steps = np.linspace(0, total_timesteps - 1, num_steps, dtype=int)
        # Reverse to get descending order [999 -> 0]
        return list(reversed(steps.tolist()))

    def sample(
        self,
        cloudy_images: torch.Tensor,
        transformer_predictions: torch.Tensor,
        cseb_embeddings: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        eta: float = 0.0,
        clip_denoised: bool = True,
        clip_range: float = 4.0,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Execute deterministic DDIM reverse sampling on residual space.
        
        Args:
            cloudy_images: [B, 3, H, W] in [0, 1]
            transformer_predictions: [B, 3, H, W] in [0, 1]
            cseb_embeddings: Optional [B, 192]
            num_steps: Number of reverse diffusion steps (default: 10)
            eta: Stochasticity parameter (default: 0.0 for deterministic DDIM)
            clip_denoised: If True, clamps intermediate r_tilde_0 to [-clip_range, +clip_range]
            clip_range: Bound for clipping scaled residual (default: 4.0, corresponding to [-1, 1] unscaled)
            generator: Optional torch.Generator for reproducible initial noise
            initial_noise: Optional explicit initial noise tensor [B, 3, H, W]
            return_trajectory: If True, returns intermediate residual states
            
        Returns:
            predicted_residual: De-scaled clean residual [B, 3, H, W] in [-1.0, 1.0]
            (optional) trajectory: List of intermediate residual tensors
        """
        b, _, h, w = cloudy_images.shape
        device = cloudy_images.device

        # 1. Initialize noisy residual r_tilde_T ~ N(0, I)
        if initial_noise is not None:
            r_t = initial_noise.to(device=device, dtype=torch.float32)
        else:
            r_t = torch.randn((b, 3, h, w), generator=generator, device=device, dtype=torch.float32)

        # 2. Compute descending timestep sequence
        timesteps = self.compute_timesteps(num_steps, total_timesteps=self.scheduler.num_timesteps)

        trajectory = [r_t] if return_trajectory else []

        # 3. Reverse diffusion loop
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

        with torch.inference_mode():
            for i, t in enumerate(timesteps):
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                t_tensor = torch.full((b,), t, device=device, dtype=torch.long)

                # Assemble 9-channel spatial input: [r_tilde_t, x_hat_0, y]
                x_in = torch.cat([r_t, transformer_predictions, cloudy_images], dim=1)

                # Predict noise epsilon_theta
                eps_pred = self.model(x_in, t_tensor, cseb_embeddings)

                # Current cumulative alpha
                alpha_bar_t = alphas_cumprod[t]
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

                # Predict clean scaled residual: r_tilde_0_pred
                r_tilde_0_pred = (r_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

                if clip_denoised:
                    r_tilde_0_pred = torch.clamp(r_tilde_0_pred, -clip_range, clip_range)

                if t_prev < 0:
                    # Final step reached (t_prev = -1) -> clean state
                    r_t = r_tilde_0_pred
                else:
                    # Intermediate reverse step
                    alpha_bar_t_prev = alphas_cumprod[t_prev]
                    sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
                    sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1.0 - alpha_bar_t_prev)

                    if eta > 0.0:
                        sigma = eta * torch.sqrt(
                            (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
                            * (1.0 - alpha_bar_t / alpha_bar_t_prev)
                        )
                        noise = torch.randn_like(r_t)
                        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_t_prev - sigma**2, min=0.0)) * eps_pred
                        r_t = sqrt_alpha_bar_t_prev * r_tilde_0_pred + dir_xt + sigma * noise
                    else:
                        # Deterministic ODE step (eta = 0.0)
                        dir_xt = sqrt_one_minus_alpha_bar_t_prev * eps_pred
                        r_t = sqrt_alpha_bar_t_prev * r_tilde_0_pred + dir_xt

                if return_trajectory:
                    trajectory.append(r_t)

        # 4. De-scale predicted residual by 0.25 (s_r = 4.0)
        predicted_residual = 0.25 * r_t

        if return_trajectory:
            return predicted_residual, trajectory
        return predicted_residual
