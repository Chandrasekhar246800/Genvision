"""
GenVision ResDiff-TransUNet Module
==================================

Components:
- GaussianDiffusionScheduler: Linear beta schedule and forward noising.
- ResidualProcessor: Residual calculation, scaling (s_r=4.0), and unscaling.
- FrozenTransformerConditioner: Safe wrapper for frozen CS-TransUNet backbone.
- ResDiffUNet: 4-stage lightweight denoising U-Net with AdaGN conditioning.
- DDIMSampler: Fast deterministic reverse sampling (eta=0.0).
- ResidualGating: Validation-calibrated residual integration (lambda in [0.0, 1.0]).
"""

from .scheduler import GaussianDiffusionScheduler
from .residual import ResidualProcessor, FrozenTransformerConditioner
from .resdiff_unet import ResDiffUNet, AdaGNResBlock, ConditioningProjection, SinusoidalTimestepEmbedding
from .ddim_sampler import DDIMSampler
from .residual_gating import ResidualGating

__all__ = [
    "GaussianDiffusionScheduler",
    "ResidualProcessor",
    "FrozenTransformerConditioner",
    "ResDiffUNet",
    "AdaGNResBlock",
    "ConditioningProjection",
    "SinusoidalTimestepEmbedding",
    "DDIMSampler",
    "ResidualGating",
]
