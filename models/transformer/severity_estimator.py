"""
GenVision Cloud Severity Estimation Branch (CSEB)
Extracts global atmospheric cloud thickness representations and produces severity embeddings
along with three-class probabilistic predictions (thin, medium, thick).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CloudSeverityEstimator(nn.Module):
    """
    Cloud Severity Estimation Branch (CSEB).
    Computes global atmospheric representation from Stage-2 convolutional features.
    Produces:
      1. severity_embedding: 192-dimensional vector used for AdaLN Transformer modulation.
      2. severity_logits: Unnormalized 3-class logits (thin=0, medium=1, thick=2).
      3. severity_probabilities: Softmax probability distribution over the 3 classes.
    """

    def __init__(self, in_channels: int = 96, embedding_dim: int = 192, num_classes: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Multi-Layer Perceptron embedding network
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Auxiliary 3-class classification head (thin, medium, thick)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the cloud severity branch.
        Args:
            x: Stage-2 feature map tensor of shape (B, in_channels, H, W)
        Returns:
            severity_embedding: (B, embedding_dim) - 192-dim embedding for AdaLN
            severity_logits: (B, num_classes) - raw logits for 3 classes
            severity_probabilities: (B, num_classes) - normalized softmax probabilities (sum=1)
        """
        b = x.shape[0]

        # Spatial average pooling: (B, in_channels, H, W) -> (B, in_channels)
        pooled = self.pool(x).view(b, -1)

        # Severity embedding projection
        severity_embedding = self.mlp(pooled)  # (B, 192)

        # 3-class classification head
        severity_logits = self.classifier(severity_embedding)  # (B, 3)
        severity_probabilities = F.softmax(severity_logits, dim=-1)  # (B, 3)

        return severity_embedding, severity_logits, severity_probabilities
