"""
Numeric encoder: 3-layer MLP operating on 80-dim NCA features -> R^128 embedding.
Implements Eq. (6) from the MAXFUSE paper.

v_N = ReLU(W3 * ReLU(W2 * ReLU(W1*x + b1) + b2) + b3),  v_N in R^128

NOTE: Dropout is intentionally left active at inference time
      to enable MC-Dropout uncertainty estimation (N2).
"""

import torch
import torch.nn as nn


class NumericEncoder(nn.Module):
    """
    3-layer MLP for numeric PE features.
    Input: (B, 80) NCA-reduced feature vector
    Output: (B, 128) embedding vector
    """

    def __init__(
        self,
        in_dim: int = 80,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        self.dropout_rate = dropout

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 80) float32 tensor

        Returns:
            (B, 128) embedding
        """
        x = self.drop1(self.relu(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu(self.bn2(self.fc2(x))))
        x = self.drop3(self.relu(self.fc3(x)))
        return x

    def forward_with_dropout_forced(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dropout ALWAYS active (for MC-Dropout sampling).
        """
        training_state = self.training
        self.train()
        out = self.forward(x)
        if not training_state:
            self.eval()
        return out
