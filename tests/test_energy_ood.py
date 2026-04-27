import torch
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.energy_ood import (
    energy_score, find_threshold_fpr95, reject_or_classify,
    compute_ood_auroc, compute_fpr95, UNKNOWN_LABEL
)


def test_energy_score_shape():
    logits = torch.randn(8, 25)
    E = energy_score(logits)
    assert E.shape == (8,)


def test_energy_score_dtype():
    logits = torch.randn(4, 25)
    E = energy_score(logits)
    assert E.dtype == torch.float32


def test_energy_id_lower_than_ood():
    """In-dist logits (peaked) should have lower energy than OOD (flat)."""
    id_logits  = torch.zeros(10, 25)
    id_logits[:, 0] = 10.0

    ood_logits = torch.zeros(10, 25)

    E_id  = energy_score(id_logits)
    E_ood = energy_score(ood_logits)
    assert E_id.mean() < E_ood.mean(), "In-dist should have lower energy than OOD"


def test_reject_or_classify_all_unknown():
    logits = torch.randn(6, 25)
    tau = -1000.0  # extremely low threshold -> all samples are OOD
    preds, energies = reject_or_classify(logits, tau)
    assert (preds == UNKNOWN_LABEL).all()


def test_reject_or_classify_none_unknown():
    logits = torch.randn(6, 25)
    tau = 1000.0  # extremely high threshold -> all samples accepted
    preds, energies = reject_or_classify(logits, tau)
    assert (preds != UNKNOWN_LABEL).all()


def test_reject_or_classify_shape():
    logits = torch.randn(6, 25)
    tau = 0.0
    preds, energies = reject_or_classify(logits, tau)
    assert preds.shape == (6,)
    assert energies.shape == (6,)


def test_auroc_separable():
    """Perfect separation should give AUROC = 1."""
    E_id  = np.array([-20.0] * 100)
    E_ood = np.array([-5.0]  * 100)
    auroc = compute_ood_auroc(E_id, E_ood)
    assert auroc > 0.99


def test_auroc_random():
    """Random scores should give AUROC around 0.5."""
    rng = np.random.default_rng(42)
    E_id  = rng.normal(-10, 1, 200)
    E_ood = rng.normal(-10, 1, 200)
    auroc = compute_ood_auroc(E_id, E_ood)
    assert 0.3 < auroc < 0.7


def test_fpr95_perfect():
    """OOD samples with higher energy than all in-dist -> FPR95 = 0."""
    E_id  = np.array([-20.0] * 100)
    E_ood = np.array([-5.0]  * 100)
    fpr = compute_fpr95(E_id, E_ood)
    assert fpr == 0.0


def test_find_threshold_fpr95():
    energies = torch.linspace(-30, -5, 100)
    tau = find_threshold_fpr95(energies, fpr_target=0.05)
    # 95% of samples should be <= tau
    accepted = (energies <= tau).float().mean().item()
    assert accepted >= 0.94
