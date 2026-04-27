import numpy as np
import pytest
import sys
import tempfile
import os
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.pe_extractor import extract_pe_features, _entropy, _load_pixels


def make_png(array: np.ndarray) -> str:
    """Save a numpy array as a temp PNG and return its path."""
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    Image.fromarray(array.astype(np.uint8), mode='L').save(path)
    return path


# ---- _entropy helper ----

def test_entropy_uniform():
    data = np.tile(np.arange(256, dtype=np.float32), 4)
    H = _entropy(data)
    assert abs(H - 8.0) < 0.05, f"Expected ~8.0, got {H}"


def test_entropy_constant():
    data = np.zeros(256, dtype=np.float32)
    H = _entropy(data)
    assert H < 0.1, f"Expected ~0.0, got {H}"


def test_entropy_empty():
    H = _entropy(np.array([], dtype=np.float32))
    assert H >= 0.0


# ---- _load_pixels ----

def test_load_pixels_png():
    path = make_png(np.random.randint(0, 256, (64, 64), dtype=np.uint8))
    try:
        px = _load_pixels(path)
        assert px.shape == (224, 224)
        assert px.dtype == np.float32
        assert px.min() >= 0.0 and px.max() <= 255.0
    finally:
        os.unlink(path)


# ---- extract_pe_features ----

def test_extract_features_shape():
    path = make_png(np.random.randint(0, 256, (224, 224), dtype=np.uint8))
    try:
        feats = extract_pe_features(path)
        assert feats is not None
        assert feats.shape == (508,)
    finally:
        os.unlink(path)


def test_extract_features_dtype():
    path = make_png(np.zeros((128, 128), dtype=np.uint8))
    try:
        feats = extract_pe_features(path)
        assert feats is not None
        assert feats.dtype == np.float32
    finally:
        os.unlink(path)


def test_extract_features_no_nan():
    path = make_png(np.random.randint(0, 256, (224, 224), dtype=np.uint8))
    try:
        feats = extract_pe_features(path)
        assert feats is not None
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isinf(feats))
    finally:
        os.unlink(path)


def test_extract_features_nonexistent():
    result = extract_pe_features('/nonexistent/path/file.png')
    assert result is None


def test_extract_features_all_zeros():
    path = make_png(np.zeros((224, 224), dtype=np.uint8))
    try:
        feats = extract_pe_features(path)
        assert feats is not None
        assert feats.shape == (508,)
    finally:
        os.unlink(path)


def test_extract_features_all_max():
    path = make_png(np.full((224, 224), 255, dtype=np.uint8))
    try:
        feats = extract_pe_features(path)
        assert feats is not None
        assert feats.shape == (508,)
    finally:
        os.unlink(path)


def test_features_differ_across_samples():
    path_a = make_png(np.full((224, 224), 50,  dtype=np.uint8))
    path_b = make_png(np.full((224, 224), 200, dtype=np.uint8))
    try:
        fa = extract_pe_features(path_a)
        fb = extract_pe_features(path_b)
        assert not np.allclose(fa, fb)
    finally:
        os.unlink(path_a)
        os.unlink(path_b)
