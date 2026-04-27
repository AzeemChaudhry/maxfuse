"""
N2: Confidence-Calibrated Dynamic Fusion via MC-Dropout.
Implements Equations (11)-(14) from the MAXFUSE paper.

MC-Dropout variance (Eqs. 11-12):
    sigma^2_I = (1/T) sum_t (y_hat^(t)_I - y_bar_I)^2
    sigma^2_N = (1/T) sum_t (y_hat^(t)_N - y_bar_N)^2

Dynamic weights (Eq. 13):
    w_I = (1/sigma^2_I) / (1/sigma^2_I + 1/sigma^2_N)
    w_N = 1 - w_I

Fused representation (Eq. 14):
    z = w_I * v_hat_I + w_N * v_hat_N

Key behaviour on packed malware:
    Image looks like noise -> sigma^2_I >> sigma^2_N -> w_I -> 0 -> numeric branch dominates.
"""

import torch
import torch.nn as nn
from typing import Callable


def mc_dropout_passes(
    forward_fn: Callable,
    x: torch.Tensor,
    T: int = 10
) -> tuple:
    """
    Run T stochastic forward passes with dropout forced active.

    Args:
        forward_fn: A callable that maps x -> embedding (with dropout active)
        x: Input tensor
        T: Number of MC passes

    Returns:
        (mean, variance) — both shape matching the output of forward_fn
    """
    preds = []
    for _ in range(T):
        with torch.no_grad():
            pred = forward_fn(x)
        preds.append(pred)

    preds = torch.stack(preds, dim=0)
    mean  = preds.mean(dim=0)
    var   = preds.var(dim=0, unbiased=True)
    return mean, var


def dynamic_weights(
    var_i: torch.Tensor,
    var_n: torch.Tensor,
    eps: float = 1e-8
) -> tuple:
    """
    Compute inverse-variance fusion weights (Eq. 13).

    Args:
        var_i: Image branch variance (B, dim) or (B,)
        var_n: Numeric branch variance (B, dim) or (B,)
        eps: Numerical stability term

    Returns:
        (w_i, w_n) — weights that sum to 1, shape (B, 1)
    """
    if var_i.dim() > 1:
        sigma2_i = var_i.mean(dim=-1, keepdim=True) + eps
        sigma2_n = var_n.mean(dim=-1, keepdim=True) + eps
    else:
        sigma2_i = var_i.unsqueeze(-1) + eps
        sigma2_n = var_n.unsqueeze(-1) + eps

    inv_i = 1.0 / sigma2_i
    inv_n = 1.0 / sigma2_n
    total = inv_i + inv_n

    w_i = inv_i / total
    w_n = inv_n / total

    return w_i, w_n


def fuse(
    v_hat_i: torch.Tensor,
    v_hat_n: torch.Tensor,
    w_i: torch.Tensor,
    w_n: torch.Tensor
) -> torch.Tensor:
    """
    Weighted fusion (Eq. 14):  z = w_I * v_hat_I + w_N * v_hat_N

    Args:
        v_hat_i: (B, shared_dim)
        v_hat_n: (B, shared_dim)
        w_i: (B, 1)
        w_n: (B, 1)

    Returns:
        z: (B, shared_dim) fused representation
    """
    return w_i * v_hat_i + w_n * v_hat_n


class DynamicFusionModule(nn.Module):
    """
    Wrapper module that holds the MC-Dropout fusion logic.
    """

    def __init__(self, T: int = 10, eps: float = 1e-8):
        super().__init__()
        self.T = T
        self.eps = eps

    def forward(
        self,
        v_hat_i: torch.Tensor,
        v_hat_n: torch.Tensor,
        fn_i: Callable,
        fn_n: Callable,
        x_i: torch.Tensor,
        x_n: torch.Tensor
    ) -> torch.Tensor:
        """
        Full N2 forward pass.

        Args:
            v_hat_i, v_hat_n: Cross-attended embeddings from N1 (used as base)
            fn_i: Callable for image branch (with dropout active)
            fn_n: Callable for numeric branch (with dropout active)
            x_i: Raw image input for MC passes
            x_n: Raw numeric input for MC passes

        Returns:
            z: (B, shared_dim) fused embedding
        """
        _, var_i = mc_dropout_passes(fn_i, x_i, T=self.T)
        _, var_n = mc_dropout_passes(fn_n, x_n, T=self.T)
        w_i, w_n = dynamic_weights(var_i, var_n, eps=self.eps)
        return fuse(v_hat_i, v_hat_n, w_i, w_n)
