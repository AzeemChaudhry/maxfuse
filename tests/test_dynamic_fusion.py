import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.dynamic_fusion import dynamic_weights, fuse, mc_dropout_passes


def test_weights_sum_to_one():
    B = 4
    var_i = torch.rand(B, 128)
    var_n = torch.rand(B, 128)
    w_i, w_n = dynamic_weights(var_i, var_n)
    total = (w_i + w_n).squeeze(-1)
    assert torch.allclose(total, torch.ones(B), atol=1e-5)


def test_high_variance_gets_low_weight():
    B = 2
    var_i = torch.full((B, 128), 100.0)
    var_n = torch.full((B, 128), 0.01)
    w_i, w_n = dynamic_weights(var_i, var_n)
    assert (w_i < w_n).all(), "High variance branch should get lower weight"


def test_equal_variance_gives_half_half():
    B = 3
    var_i = torch.ones(B, 128)
    var_n = torch.ones(B, 128)
    w_i, w_n = dynamic_weights(var_i, var_n)
    assert torch.allclose(w_i, torch.full((B, 1), 0.5), atol=1e-5)
    assert torch.allclose(w_n, torch.full((B, 1), 0.5), atol=1e-5)


def test_fuse_output_shape():
    B, D = 4, 128
    v_i = torch.randn(B, D)
    v_n = torch.randn(B, D)
    w_i = torch.full((B, 1), 0.6)
    w_n = torch.full((B, 1), 0.4)
    z = fuse(v_i, v_n, w_i, w_n)
    assert z.shape == (B, D)


def test_fuse_weighted_correctness():
    B, D = 2, 4
    v_i = torch.ones(B, D) * 2.0
    v_n = torch.ones(B, D) * 4.0
    w_i = torch.full((B, 1), 0.5)
    w_n = torch.full((B, 1), 0.5)
    z = fuse(v_i, v_n, w_i, w_n)
    expected = torch.ones(B, D) * 3.0
    assert torch.allclose(z, expected, atol=1e-5)


def test_mc_dropout_passes():
    def noisy_fn(x):
        return x + torch.randn_like(x) * 0.1

    x = torch.randn(4, 128)
    mean, var = mc_dropout_passes(noisy_fn, x, T=20)
    assert mean.shape == x.shape
    assert var.shape == x.shape
    assert (var >= 0).all()
