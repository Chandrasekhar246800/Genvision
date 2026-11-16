"""
ResDiff-UNet: Lightweight Denoising Network for Residual Diffusion
==================================================================

Optimized for consumer CPU execution on 256 x 256 satellite imagery.
Architecture Highlights:
- Spatial Input: 9 channels [noisy_scaled_residual, transformer_prediction, cloudy_image]
- Target Output: 3 channels (predicted Gaussian noise epsilon_theta)
- Channel Hierarchy: (32, 64, 128, 256)
- Global Conditioning: 320 dimensions [128-dim timestep + 192-dim CSEB embedding]
- Conditioning Injection: Adaptive Group Normalization (AdaGN) in every residual block
- Zero Spatial Cross-Attention: Eliminates O((HW)^2) memory bandwidth bottlenecks on CPU.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Computes sinusoidal positional embeddings for diffusion timesteps t.
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 1D tensor of timesteps of shape [B].
        Returns:
            Positional embedding tensor of shape [B, dim].
        """
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=device) / (half_dim - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1, 0, 0))
        return embedding


class ConditioningProjection(nn.Module):
    """
    Projects and combines the 128-dim timestep embedding and 192-dim continuous
    CSEB severity embedding into a unified 320-dim global conditioning representation.
    """

    def __init__(self, time_dim: int = 128, cseb_dim: int = 192):
        super().__init__()
        self.time_dim = time_dim
        self.cseb_dim = cseb_dim
        self.total_dim = time_dim + cseb_dim  # 320

        self.time_embed = SinusoidalTimestepEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cseb_mlp = nn.Sequential(
            nn.Linear(cseb_dim, cseb_dim),
            nn.SiLU(),
            nn.Linear(cseb_dim, cseb_dim),
        )

    def forward(self, t: torch.Tensor, e_cseb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            t: Timesteps tensor [B]
            e_cseb: Continuous CSEB embedding [B, 192]. If None, zero-filled.
        Returns:
            Unified conditioning vector [B, 320].
        """
        b = t.shape[0]
        device = t.device

        t_emb = self.time_mlp(self.time_embed(t))

        if e_cseb is None:
            cseb_emb = torch.zeros((b, self.cseb_dim), device=device, dtype=torch.float32)
        else:
            cseb_emb = self.cseb_mlp(e_cseb.to(device).float())

        return torch.cat([t_emb, cseb_emb], dim=1)  # [B, 320]


class AdaGNResBlock(nn.Module):
    """
    Residual block with Adaptive Group Normalization (AdaGN).
    Modulates intermediate feature channels using scale and shift parameters
    derived from the 320-dim global conditioning vector.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 320,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Adjust num_groups if channels < 8
        groups = min(num_groups, in_channels)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        groups_out = min(num_groups, out_channels)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        # AdaGN linear projection: cond_dim -> 2 * in_channels (scale and shift)
        self.cond_proj = nn.Linear(cond_dim, 2 * in_channels)

        # Residual shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, in_channels, H, W]
            cond: Global conditioning vector [B, cond_dim]
        Returns:
            Refined feature tensor [B, out_channels, H, W]
        """
        # Compute AdaGN modulation
        scale_shift = self.cond_proj(cond)  # [B, 2 * in_channels]
        scale, shift = scale_shift.chunk(2, dim=1)  # [B, in_channels] each
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        # First normalized block with AdaGN modulation
        h = self.norm1(x)
        h = (1.0 + scale) * h + shift
        h = self.act(h)
        h = self.conv1(h)

        # Second block
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class ResDiffUNet(nn.Module):
    """
    Lightweight ResDiff-UNet Denoising Backbone.
    
    Structure:
    - Input: 9 channels [noisy_scaled_residual, transformer_prediction, cloudy_image]
    - Encoder: Stage 1 (32), Stage 2 (64), Stage 3 (128)
    - Bottleneck: Stage 4 (256)
    - Decoder: UpStage 3 (128), UpStage 2 (64), UpStage 1 (32)
    - Output: 3 channels (predicted noise)
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 3,
        stage_channels: Tuple[int, int, int, int] = (32, 64, 128, 256),
        time_dim: int = 128,
        cseb_dim: int = 192,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stage_channels = stage_channels
        c1, c2, c3, c4 = stage_channels
        cond_dim = time_dim + cseb_dim  # 320

        # 1. Conditioning Projection Network
        self.cond_net = ConditioningProjection(time_dim=time_dim, cseb_dim=cseb_dim)

        # 2. Input Stem (9 -> 32 channels at 256 x 256)
        self.stem = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        # 3. Encoder Hierarchy
        # Stage 1: 256 x 256 x 32
        self.enc1 = AdaGNResBlock(c1, c1, cond_dim=cond_dim)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)  # 128 x 128

        # Stage 2: 128 x 128 x 64
        self.enc2 = AdaGNResBlock(c2, c2, cond_dim=cond_dim)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)  # 64 x 64

        # Stage 3: 64 x 64 x 128
        self.enc3 = AdaGNResBlock(c3, c3, cond_dim=cond_dim)
        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)  # 32 x 32

        # 4. Bottleneck Stage 4: 32 x 32 x 256
        self.bot1 = AdaGNResBlock(c4, c4, cond_dim=cond_dim)
        self.bot2 = AdaGNResBlock(c4, c4, cond_dim=cond_dim)

        # 5. Decoder Hierarchy
        # UpStage 3: 32 x 32 -> 64 x 64
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2, bias=False)
        self.fuse3 = nn.Conv2d(c3 * 2, c3, kernel_size=1, bias=False)
        self.dec3 = AdaGNResBlock(c3, c3, cond_dim=cond_dim)

        # UpStage 2: 64 x 64 -> 128 x 128
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=False)
        self.fuse2 = nn.Conv2d(c2 * 2, c2, kernel_size=1, bias=False)
        self.dec2 = AdaGNResBlock(c2, c2, cond_dim=cond_dim)

        # UpStage 1: 128 x 128 -> 256 x 256
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=False)
        self.fuse1 = nn.Conv2d(c1 * 2, c1, kernel_size=1, bias=False)
        self.dec1 = AdaGNResBlock(c1, c1, cond_dim=cond_dim)

        # 6. Reconstruction Head
        self.out_norm = nn.GroupNorm(8, c1)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x_in: torch.Tensor,
        t: torch.Tensor,
        e_cseb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the denoising U-Net.
        
        Args:
            x_in: Spatial conditioning tensor [B, 9, H, W]
                  Slice 0..3: Noisy scaled residual r_tilde_t
                  Slice 3..6: Initial prediction x_hat_0
                  Slice 6..9: Cloudy input y
            t: Diffusion timesteps tensor [B] with integers in [0, 999]
            e_cseb: Optional continuous CSEB embedding [B, 192]
            
        Returns:
            Predicted noise tensor epsilon_pred of shape [B, 3, H, W].
        """
        # 1. Compute global conditioning vector [B, 320]
        cond = self.cond_net(t, e_cseb)

        # 2. Input Stem
        h1 = self.stem(x_in)  # [B, 32, 256, 256]

        # 3. Encoder Pass
        e1 = self.enc1(h1, cond)         # [B, 32, 256, 256]
        h2 = self.down1(e1)              # [B, 64, 128, 128]

        e2 = self.enc2(h2, cond)         # [B, 64, 128, 128]
        h3 = self.down2(e2)              # [B, 128, 64, 64]

        e3 = self.enc3(h3, cond)         # [B, 128, 64, 64]
        h4 = self.down3(e3)              # [B, 256, 32, 32]

        # 4. Bottleneck Pass
        b = self.bot1(h4, cond)          # [B, 256, 32, 32]
        b = self.bot2(b, cond)           # [B, 256, 32, 32]

        # 5. Decoder Pass with Skip Connections
        d3 = self.up3(b)                                # [B, 128, 64, 64]
        d3 = self.fuse3(torch.cat([d3, e3], dim=1))     # [B, 128, 64, 64]
        d3 = self.dec3(d3, cond)                        # [B, 128, 64, 64]

        d2 = self.up2(d3)                               # [B, 64, 128, 128]
        d2 = self.fuse2(torch.cat([d2, e2], dim=1))     # [B, 64, 128, 128]
        d2 = self.dec2(d2, cond)                        # [B, 64, 128, 128]

        d1 = self.up1(d2)                               # [B, 32, 256, 256]
        d1 = self.fuse1(torch.cat([d1, e1], dim=1))     # [B, 32, 256, 256]
        d1 = self.dec1(d1, cond)                        # [B, 32, 256, 256]

        # 6. Head
        out = self.out_norm(d1)
        out = self.out_act(out)
        noise_pred = self.out_conv(out)  # [B, 3, 256, 256]

        return noise_pred
