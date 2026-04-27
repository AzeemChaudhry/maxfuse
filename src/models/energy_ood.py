"""
N3: Energy-Score OOD Rejection.
Implements Equations (15)-(17) from the MAXFUSE paper.

Energy score (Eq. 15):
    E(z; T) = -T * log sum_k exp(f_k(z) / T)

Rejection decision (Eq. 16):
    y_hat = arg max_k f_k(z)   if E(z) <= tau
    y_hat = UNKNOWN            if E(z) > tau

tau is set at FPR95 operating point on the validation set:
    95% of in-distribution samples should have E(z) <= tau.

Energy margin training loss (Eq. 17):
    (See energy_margin_loss.py — this module only handles inference-time scoring)

Reference: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
           https://github.com/wetliu/energy_ood
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


UNKNOWN_LABEL = -1


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute energy score for each sample (Eq. 15).

    Lower energy  = confident, fits a known family (in-distribution).
    Higher energy = logits spread flat = likely OOD / zero-day.

    Args:
        logits: Classifier output (B, num_classes)
        temperature: Temperature T (default 1.0)

    Returns:
        energy: (B,) float tensor — more negative = more in-distribution
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def find_threshold_fpr95(
    energy_id: torch.Tensor,
    fpr_target: float = 0.05
) -> float:
    """
    Find tau such that FPR = (1 - fpr_target) of in-distribution samples are accepted.
    tau = 95th percentile of in-distribution energies (FPR95 operating point).

    Args:
        energy_id: Energy scores for ALL in-distribution validation samples (N,)
        fpr_target: Fraction of in-dist to reject (default 0.05 for FPR95)

    Returns:
        tau: float threshold
    """
    tau = float(torch.quantile(energy_id, 1.0 - fpr_target))
    accepted_pct = (energy_id <= tau).float().mean().item()
    print(f"[N3] tau = {tau:.4f}  |  In-dist acceptance rate: {accepted_pct*100:.1f}%")
    return tau


def reject_or_classify(
    logits: torch.Tensor,
    tau: float,
    temperature: float = 1.0
) -> tuple:
    """
    Apply energy threshold to decide between known classification and UNKNOWN.
    Implements Eq. (16).

    Args:
        logits: (B, num_classes)
        tau: Energy threshold (from find_threshold_fpr95)
        temperature: Temperature for energy score

    Returns:
        preds: (B,) int tensor — predicted class index, or UNKNOWN_LABEL (-1)
        energies: (B,) float tensor — energy scores
    """
    energies = energy_score(logits, temperature)
    preds    = logits.argmax(dim=-1).clone()
    ood_mask = energies > tau
    preds[ood_mask] = UNKNOWN_LABEL
    return preds, energies


def compute_ood_auroc(
    energy_id: np.ndarray,
    energy_ood: np.ndarray
) -> float:
    """
    Compute AUROC for OOD detection using energy scores.

    Args:
        energy_id: Energy scores for in-distribution test samples
        energy_ood: Energy scores for OOD (held-out family) test samples

    Returns:
        AUROC score (higher = better OOD separation)
    """
    labels = np.concatenate([
        np.zeros(len(energy_id)),
        np.ones(len(energy_ood))
    ])
    scores = np.concatenate([energy_id, energy_ood])
    auroc = roc_auc_score(labels, scores)
    return auroc


def compute_fpr95(
    energy_id: np.ndarray,
    energy_ood: np.ndarray
) -> float:
    """
    FPR at 95% TPR (fraction of OOD samples incorrectly accepted when 95% TPR).
    """
    threshold = np.percentile(energy_id, 95)
    fpr = (energy_ood <= threshold).mean()
    return float(fpr)
