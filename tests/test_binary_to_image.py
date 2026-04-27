import numpy as np
import tempfile
import os
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.binary_to_image import binary_to_image


def test_output_shape():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as f:
        f.write(os.urandom(10000))
        path = f.name
    try:
        img = binary_to_image(path, size=224)
        assert img.shape == (224, 224), f"Expected (224,224), got {img.shape}"
        assert img.dtype == np.float32
        assert 0.0 <= img.min() and img.max() <= 1.0
    finally:
        os.unlink(path)


def test_small_file_padding():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b'\x00' * 100)
        path = f.name
    try:
        img = binary_to_image(path, size=224)
        assert img.shape == (224, 224)
    finally:
        os.unlink(path)


def test_custom_size():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        f.write(os.urandom(5000))
        path = f.name
    try:
        img = binary_to_image(path, size=64)
        assert img.shape == (64, 64)
    finally:
        os.unlink(path)


def test_normalisation_range():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        # Write bytes covering full range 0-255
        f.write(bytes(range(256)) * 40)
        path = f.name
    try:
        img = binary_to_image(path, size=224)
        assert img.min() >= 0.0
        assert img.max() <= 1.0
    finally:
        os.unlink(path)
