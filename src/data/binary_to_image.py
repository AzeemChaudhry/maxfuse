"""
Binary-to-image conversion for malware visualization.
Implements Equation (1) from the MAXFUSE paper.
Reference: Nataraj et al., "Malware Images: Visualization and Automatic Classification", VizSec 2011.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def binary_to_image(path: str, size: int = 224) -> np.ndarray:
    """
    Convert a PE binary file to a 224x224 grayscale image.

    Algorithm (Eq. 1):
      1. Read raw bytes b = [b1, b2, ..., bL], bi in {0,...,255}
      2. Reshape to 2D array of width=256: shape = (floor(L/256), 256)
      3. Resize to (size, size)
      4. Normalise pixel values to [0, 1]

    Args:
        path: Path to PE binary file
        size: Target image dimension (default 224 for EfficientNet)

    Returns:
        np.ndarray of shape (size, size), dtype float32, range [0, 1]
    """
    with open(path, 'rb') as f:
        bytez = np.frombuffer(f.read(), dtype=np.uint8)

    if len(bytez) < 256:
        bytez = np.pad(bytez, (0, 256 - len(bytez)), mode='constant')

    width = 256
    height = len(bytez) // width
    bytez = bytez[:height * width].reshape(height, width)

    img = Image.fromarray(bytez, mode='L')
    img = img.resize((size, size), Image.Resampling.BILINEAR)

    return np.array(img, dtype=np.float32) / 255.0


def convert_directory(binary_dir: str, output_dir: str, size: int = 224):
    """
    Batch convert a directory of PE binaries to images.
    Expects binary_dir to have sub-folders per family (same structure as Malimg).

    Args:
        binary_dir: Root directory with family sub-folders containing binaries
        output_dir: Root directory to save grayscale PNGs
        size: Target image size
    """
    binary_dir = Path(binary_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    families = [d for d in binary_dir.iterdir() if d.is_dir()]
    print(f"Found {len(families)} families.")

    for family_dir in tqdm(families, desc="Converting families"):
        out_family_dir = output_dir / family_dir.name
        out_family_dir.mkdir(exist_ok=True)

        for binary_path in family_dir.glob('*'):
            if binary_path.suffix.lower() in ('.exe', '.bin', ''):
                try:
                    img_array = binary_to_image(str(binary_path), size=size)
                    img_uint8 = (img_array * 255).astype(np.uint8)
                    out_path = out_family_dir / (binary_path.stem + '.png')
                    Image.fromarray(img_uint8, mode='L').save(out_path)
                except Exception as e:
                    print(f"  [WARN] Failed {binary_path.name}: {e}")

    print(f"Conversion complete. Images saved to {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert PE binaries to malware images")
    parser.add_argument('--binary_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--size', type=int, default=224)
    args = parser.parse_args()
    convert_directory(args.binary_dir, args.output_dir, args.size)
