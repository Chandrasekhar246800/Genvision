"""
GenVision CS-TransUNet Architectural Blocks
Implements LayerNorm2d, DoubleResidualBlock, Multi-Dconv Head Transposed Attention (MDTA),
Gated-Dconv Feed-Forward Network (GDFN), and Severity-Modulated Transformer Blocks.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization over the channel dimension for (B, C, H, W) tensors.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance along channel axis
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x_norm + self.bias[:, None, None]


class DoubleResidualBlock(nn.Module):
    """
    Two 3x3 convolutional layers with residual shortcut and GELU activations.
    Maintains spatial inductive biases in shallow encoder/decoder stages.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = LayerNorm2d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = LayerNorm2d(out_channels)
        self.act2 = nn.GELU()

        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                LayerNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.act2(out + res)
        return out


class MultiDconvHeadTransposedAttention(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (MDTA).
    Calculates cross-covariance attention across channel dimensions (C x C) rather than
    spatial pixels (HW x HW). Achieves linear spatial complexity O(C^2 * HW), providing
    global receptive fields across cloud occlusions with sub-millisecond CPU execution.
    """

    def __init__(self, dim: int, num_heads: int = 8, bias: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Feature dimension {dim} must be divisible by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Learnable temperature parameter per head
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Combined 1x1 conv for Q, K, V
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        # Depthwise 3x3 conv encoding localized spatial context
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )

        # Output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to (B, heads, head_dim, HW)
        q = q.reshape(b, self.num_heads, self.head_dim, h * w)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w)

        # Normalize Q and K along the spatial axis for bounded dot-products
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Transposed attention map: (head_dim, HW) @ (HW, head_dim) -> (head_dim, head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        # Attention weighted aggregation: (head_dim, head_dim) @ (head_dim, HW) -> (head_dim, HW)
        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)

        out = self.project_out(out)
        return out


class GatedDconvFeedForward(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN).
    Enriches features with depthwise spatial context and non-linear gating.
    """

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.0, bias: bool = False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.project_in(x)
        x_conv = self.dwconv(x_proj)
        x1, x2 = x_conv.chunk(2, dim=1)
        # Gated activation
        x_gated = F.gelu(x1) * x2
        out = self.project_out(x_gated)
        return out


class SeverityModulatedTransformerBlock(nn.Module):
    """
    Cloud Severity-Aware Transformer Block.
    Integrates MDTA attention and GDFN feed-forward layers with Adaptive Layer Normalization
    (AdaLN) conditioned on the global cloud severity embedding.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        severity_dim: Optional[int] = 192,
        ffn_expansion_factor: float = 2.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.severity_dim = severity_dim

        self.norm1 = LayerNorm2d(dim)
        self.attn = MultiDconvHeadTransposedAttention(dim, num_heads=num_heads, bias=bias)

        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedDconvFeedForward(dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        # AdaLN modulation projection if severity conditioning is active
        if severity_dim is not None:
            self.modulation = nn.Sequential(
                nn.Linear(severity_dim, dim * 4),
            )
            # Initialize modulation projection weights to 0 for neutral starting point
            nn.init.zeros_(self.modulation[0].weight)
            nn.init.zeros_(self.modulation[0].bias)
        else:
            self.modulation = None

    def forward(
        self,
        x: torch.Tensor,
        severity_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape

        if self.modulation is not None and severity_embedding is not None:
            # Generate scale (gamma) and shift (beta) parameters from severity embedding
            mod = self.modulation(severity_embedding)  # (B, dim * 4)
            gamma1, beta1, gamma2, beta2 = mod.chunk(4, dim=-1)

            gamma1 = gamma1[:, :, None, None]
            beta1 = beta1[:, :, None, None]
            gamma2 = gamma2[:, :, None, None]
            beta2 = beta2[:, :, None, None]

            # Attention with AdaLN
            norm1 = self.norm1(x) * (1.0 + gamma1) + beta1
            x = x + self.attn(norm1)

            # FFN with AdaLN
            norm2 = self.norm2(x) * (1.0 + gamma2) + beta2
            x = x + self.ffn(norm2)
        else:
            # Standard LayerNorm fallback
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))

        return x
