import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.cross_modal_attention import CrossModalAttention


@pytest.fixture
def attn():
    return CrossModalAttention(img_dim=256, num_dim=128, shared_dim=128, num_heads=4)


def test_output_shapes(attn):
    B = 4
    v_i = torch.randn(B, 256)
    v_n = torch.randn(B, 128)
    out_i, out_n = attn(v_i, v_n)
    assert out_i.shape == (B, 128), f"Expected (4,128), got {out_i.shape}"
    assert out_n.shape == (B, 128), f"Expected (4,128), got {out_n.shape}"


def test_gradient_flow(attn):
    B = 2
    v_i = torch.randn(B, 256, requires_grad=True)
    v_n = torch.randn(B, 128, requires_grad=True)
    out_i, out_n = attn(v_i, v_n)
    loss = (out_i.sum() + out_n.sum())
    loss.backward()
    assert v_i.grad is not None
    assert v_n.grad is not None


def test_batch_size_1(attn):
    v_i = torch.randn(1, 256)
    v_n = torch.randn(1, 128)
    out_i, out_n = attn(v_i, v_n)
    assert out_i.shape == (1, 128)
    assert out_n.shape == (1, 128)


def test_output_not_nan(attn):
    v_i = torch.randn(8, 256)
    v_n = torch.randn(8, 128)
    out_i, out_n = attn(v_i, v_n)
    assert not torch.isnan(out_i).any()
    assert not torch.isnan(out_n).any()


def test_different_heads():
    attn2 = CrossModalAttention(img_dim=256, num_dim=128, shared_dim=128, num_heads=8)
    v_i = torch.randn(4, 256)
    v_n = torch.randn(4, 128)
    out_i, out_n = attn2(v_i, v_n)
    assert out_i.shape == (4, 128)
