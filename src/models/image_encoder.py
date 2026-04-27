"""
Image encoder: EfficientNet-B0 fine-tuned on malware images -> R^256 embedding.
Implements Eq. (5) from the MAXFUSE paper.

v_I = W_I * GAP(EfficientNet(I)) + b_I,  v_I in R^256
"""

import torch
import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    """
    EfficientNet-B0 backbone with a learned projection head.
    Input: (B, 1, 224, 224) grayscale image
    Output: (B, 256) embedding vector
    """

    def __init__(
        self,
        out_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()

        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
            in_chans=1
        )
        in_features = self.backbone.num_features

        # Projection head: R^1280 -> R^256 (Eq. 5)
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 224, 224) float32 tensor, values in [0,1]

        Returns:
            (B, 256) embedding
        """
        feat = self.backbone(x)
        return self.proj(feat)
