"""
GenVision CS-TransUNet (Cloud Severity-Aware Hybrid Transformer)
Asymmetric hybrid encoder-decoder network combining local convolutional spatial inductive bias
with global multi-dconv transposed channel attention (MDTA), gated-dconv feedforward layers (GDFN),
and an auxiliary Cloud Severity Estimation Branch (CSEB).
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from .blocks import (
    DoubleResidualBlock,
    LayerNorm2d,
    SeverityModulatedTransformerBlock,
)
from .severity_estimator import CloudSeverityEstimator


class CSTransUNet(nn.Module):
    """
    Cloud Severity-Aware Hybrid Transformer (CS-TransUNet) for satellite image cloud removal.

    Architecture Progression:
      Input:       B x 3   x 256 x 256
      Stage 1:     B x 48  x 256 x 256  (Conv Stem + Double Residual Block)
      Stage 2:     B x 96  x 128 x 128  (Stride-2 Conv + Double Residual Block)
      CSEB Branch: B x 192              (Global Average Pool + MLP from Stage 2)
      Stage 3:     B x 192 x 64  x 64   (Stride-2 Conv + AdaLN-MDTA + GDFN Transformer Blocks)
      Stage 4:     B x 384 x 32  x 32   (Stride-2 Conv + AdaLN-MDTA + GDFN Bottleneck Blocks)
      Decoder D3:  B x 192 x 64  x 64   (ConvTranspose + Cat with Stage 3 + Residual Block)
      Decoder D2:  B x 96  x 128 x 128  (ConvTranspose + Cat with Stage 2 + Residual Block)
      Decoder D1:  B x 48  x 256 x 256  (ConvTranspose + Cat with Stage 1 + Residual Block)
      Head:        B x 3   x 256 x 256  (Final Conv3x3 Residual ΔX + Input Clamping)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        stage_channels: Tuple[int, int, int, int] = (48, 96, 192, 384),
        transformer_depths: Tuple[int, int] = (2, 2),
        num_heads: Tuple[int, int] = (6, 12),
        severity_embedding_dim: int = 192,
        num_severity_classes: int = 3,
        ffn_expansion_factor: float = 2.0,
    ):
        super().__init__()

        c1, c2, c3, c4 = stage_channels
        self.stage_channels = stage_channels
        self.severity_embedding_dim = severity_embedding_dim

        # ---------------------------------------------------------------------
        # 1. Convolutional Encoder: Stem & Stage 1 (256 x 256)
        # ---------------------------------------------------------------------
        self.stem = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False)
        self.stage1_block = DoubleResidualBlock(c1, c1)

        # ---------------------------------------------------------------------
        # 2. Convolutional Encoder: Stage 2 (128 x 128)
        # ---------------------------------------------------------------------
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage2_block = DoubleResidualBlock(c2, c2)

        # ---------------------------------------------------------------------
        # 3. Cloud Severity Estimation Branch (CSEB)
        # ---------------------------------------------------------------------
        self.cseb = CloudSeverityEstimator(
            in_channels=c2,
            embedding_dim=severity_embedding_dim,
            num_classes=num_severity_classes,
        )

        # ---------------------------------------------------------------------
        # 4. Transformer Encoder: Stage 3 (64 x 64)
        # ---------------------------------------------------------------------
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage3_blocks = nn.ModuleList(
            [
                SeverityModulatedTransformerBlock(
                    dim=c3,
                    num_heads=num_heads[0],
                    severity_dim=severity_embedding_dim,
                    ffn_expansion_factor=ffn_expansion_factor,
                )
                for _ in range(transformer_depths[0])
            ]
        )

        # ---------------------------------------------------------------------
        # 5. Transformer Bottleneck: Stage 4 (32 x 32)
        # ---------------------------------------------------------------------
        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage4_blocks = nn.ModuleList(
            [
                SeverityModulatedTransformerBlock(
                    dim=c4,
                    num_heads=num_heads[1],
                    severity_dim=severity_embedding_dim,
                    ffn_expansion_factor=ffn_expansion_factor,
                )
                for _ in range(transformer_depths[1])
            ]
        )

        # ---------------------------------------------------------------------
        # 6. Progressive Decoder (Stages 4 -> 3 -> 2 -> 1)
        # ---------------------------------------------------------------------
        # Decoder 3: 32 x 32 -> 64 x 64
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2, bias=False)
        self.fuse3 = nn.Conv2d(c3 * 2, c3, kernel_size=1, bias=False)
        self.dec3_block = DoubleResidualBlock(c3, c3)

        # Decoder 2: 64 x 64 -> 128 x 128
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=False)
        self.fuse2 = nn.Conv2d(c2 * 2, c2, kernel_size=1, bias=False)
        self.dec2_block = DoubleResidualBlock(c2, c2)

        # Decoder 1: 128 x 128 -> 256 x 256
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=False)
        self.fuse1 = nn.Conv2d(c1 * 2, c1, kernel_size=1, bias=False)
        self.dec1_block = DoubleResidualBlock(c1, c1)

        # ---------------------------------------------------------------------
        # 7. Residual Reconstruction Head
        # ---------------------------------------------------------------------
        self.final_head = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for CS-TransUNet.
        Args:
            x: Input cloudy image tensor of shape (B, 3, 256, 256) in [0, 1].
            return_dict: If True, returns dictionary with reconstructed image,
                         severity outputs, and multiscale feature maps.
                         If False, returns only the reconstructed tensor.
            return_features: Whether to return intermediate features.
        Returns:
            Dict containing:
              - 'reconstructed': (B, 3, 256, 256) clamped to [0, 1]
              - 'residual': (B, 3, 256, 256) raw residual ΔX
              - 'severity_logits': (B, 3) raw 3-class logits
              - 'severity_probabilities': (B, 3) softmax probabilities (sum=1)
              - 'severity_embedding': (B, 192) global severity embedding
              - 'features' / 'multiscale_features': dict of encoder feature maps (stage1..stage4)
        """
        input_image = x

        # ---------------------------------------------------------------------
        # Encoder Stage 1 (256 x 256)
        # ---------------------------------------------------------------------
        f_enc1 = self.stage1_block(self.stem(x))

        # ---------------------------------------------------------------------
        # Encoder Stage 2 (128 x 128)
        # ---------------------------------------------------------------------
        f_enc2 = self.stage2_block(self.down1(f_enc1))

        # ---------------------------------------------------------------------
        # Cloud Severity Estimation Branch (CSEB)
        # ---------------------------------------------------------------------
        sev_embedding, sev_logits, sev_probs = self.cseb(f_enc2)

        # ---------------------------------------------------------------------
        # Encoder Stage 3: Transformer (64 x 64)
        # ---------------------------------------------------------------------
        f_stage3 = self.down2(f_enc2)
        for block in self.stage3_blocks:
            f_stage3 = block(f_stage3, severity_embedding=sev_embedding)
        f_enc3 = f_stage3

        # ---------------------------------------------------------------------
        # Encoder Stage 4: Transformer Bottleneck (32 x 32)
        # ---------------------------------------------------------------------
        f_stage4 = self.down3(f_enc3)
        for block in self.stage4_blocks:
            f_stage4 = block(f_stage4, severity_embedding=sev_embedding)
        f_enc4 = f_stage4

        # ---------------------------------------------------------------------
        # Progressive Decoder
        # ---------------------------------------------------------------------
        # Stage 4 -> 3
        d3 = self.up3(f_enc4)
        d3 = torch.cat([d3, f_enc3], dim=1)
        d3 = self.dec3_block(self.fuse3(d3))

        # Stage 3 -> 2
        d2 = self.up2(d3)
        d2 = torch.cat([d2, f_enc2], dim=1)
        d2 = self.dec2_block(self.fuse2(d2))

        # Stage 2 -> 1
        d1 = self.up1(d2)
        d1 = torch.cat([d1, f_enc1], dim=1)
        d1 = self.dec1_block(self.fuse1(d1))

        # ---------------------------------------------------------------------
        # Residual Reconstruction
        # ---------------------------------------------------------------------
        residual = self.final_head(d1)
        reconstructed = torch.clamp(input_image + residual, 0.0, 1.0)

        if not return_dict:
            return reconstructed

        feats = {
            "stage1": f_enc1,
            "stage2": f_enc2,
            "stage3": f_enc3,
            "stage4": f_enc4,
        }

        return {
            "reconstructed": reconstructed,
            "residual": residual,
            "severity_logits": sev_logits,
            "severity_probabilities": sev_probs,
            "severity_embedding": sev_embedding,
            "multiscale_features": feats,
            "features": feats,
        }
