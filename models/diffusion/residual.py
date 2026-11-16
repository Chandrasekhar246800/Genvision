"""
Residual Processing & Frozen CS-TransUNet Conditioning
======================================================

Implements the residual transformation manifold and frozen transformer conditioning
for the GenVision ResDiff-TransUNet architecture.

Mathematical Definitions:
- True Residual: r_0 = x_clean - x_hat_0 in [-1.0, 1.0]
- Scaled Residual: r_tilde_0 = s_r * r_0 = 4.0 * (x_clean - x_hat_0)
- De-scaled Residual: r_hat_0 = (1.0 / s_r) * r_tilde_hat_0 = 0.25 * r_tilde_hat_0

IMPORTANT NOTE ON RESIDUAL SCALING:
The scaling factor s_r = 4.0 was determined in Phase 5.1.1 based on empirical residual
distributions across 350 training scenes (sigma_train = 0.043528). Scaling by 4.0 expands
the empirical standard deviation to ~0.174 while maintaining 99.9989% of pixels within
[-1.5, +1.5]. This value is an initial experimental setting, not a proven universal optimum.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

# Import the CS-TransUNet architecture
from models.transformer.cs_transunet import CSTransUNet


class ResidualProcessor:
    """
    Handles residual computation, scaling, inverse scaling, and diagnostic profiling.
    """

    DEFAULT_SCALE_FACTOR: float = 4.0
    DEFAULT_INV_SCALE_FACTOR: float = 0.25

    def __init__(self, s_r: float = DEFAULT_SCALE_FACTOR):
        self.s_r = float(s_r)
        self.inv_s_r = 1.0 / self.s_r

    @staticmethod
    def compute_residual(
        clean_target: torch.Tensor,
        transformer_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute true reconstruction residual: r_0 = x_clean - x_hat_0.
        Both inputs are expected in range [0.0, 1.0].
        Output residual is bounded in [-1.0, 1.0].
        """
        return clean_target - transformer_prediction

    def scale_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Scale residual by s_r (default 4.0) to match diffusion noise schedule range:
        r_tilde = s_r * r
        """
        return self.s_r * residual

    def unscale_residual(self, scaled_residual: torch.Tensor) -> torch.Tensor:
        """
        Invert residual scaling: r = (1 / s_r) * r_tilde = 0.25 * r_tilde.
        """
        return self.inv_s_r * scaled_residual

    @staticmethod
    def prepare_spatial_conditioning(
        noisy_scaled_residual: torch.Tensor,
        transformer_prediction: torch.Tensor,
        cloudy_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assemble the 9-channel spatial conditioning input tensor:
        X_in = [noisy_scaled_residual, transformer_prediction, cloudy_input] in R^(B x 9 x H x W).
        """
        return torch.cat([noisy_scaled_residual, transformer_prediction, cloudy_input], dim=1)

    @staticmethod
    def diagnostics(
        residual: torch.Tensor,
        scaled_residual: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute empirical distribution diagnostics for residual tensors.
        """
        r_flat = residual.detach().cpu().numpy().flatten()
        stats = {
            "mean": float(np.mean(r_flat)),
            "std": float(np.std(r_flat)),
            "mae": float(np.mean(np.abs(r_flat))),
            "min": float(np.min(r_flat)),
            "max": float(np.max(r_flat)),
        }
        if scaled_residual is not None:
            sr_flat = scaled_residual.detach().cpu().numpy().flatten()
            stats["scaled_mean"] = float(np.mean(sr_flat))
            stats["scaled_std"] = float(np.std(sr_flat))
            stats["scaled_min"] = float(np.min(sr_flat))
            stats["scaled_max"] = float(np.max(sr_flat))
            stats["scaled_out_of_bounds_pct"] = float(
                np.mean((sr_flat < -1.5) | (sr_flat > 1.5)) * 100.0
            )
        return stats


class FrozenTransformerConditioner(nn.Module):
    """
    Wrapper for the frozen CS-TransUNet model.
    Guarantees:
    - requires_grad is False for every parameter.
    - Model is permanently in eval() mode.
    - Forward pass runs under torch.no_grad() with detached tensors.
    - Returns initial reconstruction (clamped to [0, 1]) and continuous CSEB embedding (192-dim).
    """

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None):
        super().__init__()
        self.model = CSTransUNet()

        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
            
            # Load weights safely
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)

        # Permanently freeze all parameters
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(
        self, cloudy_images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract frozen conditioning signals from cloudy images.
        
        Args:
            cloudy_images: Tensor [B, 3, 256, 256] in range [0.0, 1.0].
            
        Returns:
            Tuple of:
              - x_hat_0: Initial reconstruction [B, 3, 256, 256] clamped in [0.0, 1.0].
              - e_cseb: Continuous cloud severity embedding [B, 192].
        """
        with torch.no_grad():
            out_dict = self.model(cloudy_images, return_dict=True)
            x_hat_0 = out_dict["reconstructed"].detach().clamp(0.0, 1.0).clone()
            e_cseb = out_dict["severity_embedding"].detach().clone()
            
        return x_hat_0, e_cseb
