"""
Image-based statistical feature extraction for the MAXFUSE numeric branch.
Implements Eq. (2): extracts a 508-dimensional feature vector from each
malware visualization PNG (grayscale image of PE binary bytes).

Since the Malimg dataset provides pre-rendered PNG visualizations rather than
raw PE files, we extract image statistics that capture the same structural
information: byte frequency distribution, entropy by spatial region, and
texture statistics. The grayscale image IS the binary — pixel intensities
directly represent byte values (0-255).

Feature layout (total = 508 dims):
  [  0:256] Pixel intensity histogram  — byte frequency distribution (256 bins)
  [256:288] Row-band entropy           — 32 horizontal bands, 7 rows each
  [288:320] Column-band entropy        — 32 vertical bands, 7 cols each
  [320:336] 4x4 block means            — 16 spatial region means
  [336:352] 4x4 block std-devs         — 16 spatial region stds
  [352:368] 4x4 block entropies        — 16 spatial region entropies
  [368:374] Global statistics          — mean, std, min, max, skew, kurtosis
  [374:438] 64-bin coarser histogram   — byte distribution summary
  [438:502] Row mean profile           — 64 horizontal band means
  [502:508] 6-strip entropy            — header + 5 section-proxy strips
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from scipy.stats import skew, kurtosis
from tqdm import tqdm


def _entropy(values: np.ndarray) -> float:
    """Shannon entropy in bits using a 256-bin histogram."""
    hist, _ = np.histogram(values, bins=256, range=(0.0, 256.0))
    p = (hist.astype(np.float64) + 1e-10)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


def _load_pixels(path: str) -> np.ndarray:
    """
    Load a file as a 224x224 float32 pixel array (values 0-255).
    Handles PNG images and raw PE binaries transparently.
    """
    p = Path(path)
    if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.gif'):
        img = Image.open(str(p)).convert('L').resize(
            (224, 224), Image.Resampling.BILINEAR)
    else:
        with open(str(p), 'rb') as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        if len(raw) < 256:
            raw = np.pad(raw, (0, 256 - len(raw)))
        w = 256
        h = len(raw) // w
        raw = raw[:h * w].reshape(h, w)
        img = Image.fromarray(raw, mode='L').resize(
            (224, 224), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.float32)   # (224, 224), range [0, 255]


def extract_pe_features(path: str) -> np.ndarray:
    """
    Extract a 508-dimensional feature vector from a malware sample.

    Accepts both PNG visualization files (Malimg dataset) and raw PE binaries.
    Maintains the original pe_extractor interface so downstream code is unchanged.

    Args:
        path: Path to PNG image or PE binary

    Returns:
        np.ndarray of shape (508,), dtype float32, or None on failure
    """
    try:
        px = _load_pixels(path)   # (224, 224), float32, range [0, 255]
        feats = []

        # [0:256] 256-bin intensity histogram
        hist256, _ = np.histogram(px, bins=256, range=(0.0, 256.0))
        feats.extend((hist256 / (px.size + 1e-8)).tolist())

        # [256:288] Row-band entropy — 32 bands x 7 rows = 224 rows
        for i in range(32):
            feats.append(_entropy(px[i * 7:(i + 1) * 7, :].ravel()))

        # [288:320] Column-band entropy — 32 bands x 7 cols = 224 cols
        for j in range(32):
            feats.append(_entropy(px[:, j * 7:(j + 1) * 7].ravel()))

        # [320:336 / 336:352 / 352:368] 4x4 block means / stds / entropies
        b_means, b_stds, b_ents = [], [], []
        for bi in range(4):
            for bj in range(4):
                blk = px[bi * 56:(bi + 1) * 56, bj * 56:(bj + 1) * 56]
                b_means.append(float(blk.mean()))
                b_stds.append(float(blk.std()))
                b_ents.append(_entropy(blk.ravel()))
        feats.extend(b_means)
        feats.extend(b_stds)
        feats.extend(b_ents)

        # [368:374] Global statistics
        flat = px.ravel()
        gstats = [
            float(flat.mean()), float(flat.std()),
            float(flat.min()),  float(flat.max()),
            float(skew(flat)),  float(kurtosis(flat)),
        ]
        feats.extend([0.0 if (np.isnan(v) or np.isinf(v)) else v
                      for v in gstats])

        # [374:438] 64-bin coarser histogram
        hist64, _ = np.histogram(px, bins=64, range=(0.0, 256.0))
        feats.extend((hist64 / (px.size + 1e-8)).tolist())

        # [438:502] Row mean profile — 64 horizontal bands
        row_bands = np.array_split(px, 64, axis=0)
        feats.extend([float(b.mean()) for b in row_bands])

        # [502:508] 6-strip entropy — top strip = header proxy, rest = sections
        strip_h = 224 // 6   # = 37
        for s in range(5):
            feats.append(_entropy(px[s * strip_h:(s + 1) * strip_h, :].ravel()))
        feats.append(_entropy(px[5 * strip_h:, :].ravel()))

        result = np.array(feats, dtype=np.float32)
        assert result.shape == (508,), f"Expected 508, got {result.shape[0]}"
        return result

    except Exception as e:
        print(f"[WARN] Feature extraction failed for {path}: {e}")
        return None


def extract_directory(image_dir: str, out_csv: str, labels_csv: str):
    """
    Extract features from all PNGs in a family-organised directory tree.

    Expected layout:
        image_dir/
            FamilyName/
                sample.png

    Output CSVs:
        out_csv    — columns: [filename, pe_0 ... pe_507]
        labels_csv — columns: [filename, label_name, label_id]

    'filename' is stored as 'FamilyName/basename.png' to guarantee
    uniqueness when different families share the same basename.
    """
    image_dir = Path(image_dir)
    families = sorted([d.name for d in image_dir.iterdir() if d.is_dir()])
    family_to_id = {f: i for i, f in enumerate(families)}

    print(f"Found {len(families)} families in {image_dir}")

    feat_col_names = [f'pe_{i}' for i in range(508)]
    rows_feat, rows_label = [], []
    failed = 0

    IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.pgm')
    for family in tqdm(families, desc='Extracting features'):
        fam_dir = image_dir / family
        imgs = sorted(p for ext in IMAGE_EXTS for p in fam_dir.glob(f'*{ext}'))
        for img_path in imgs:
            feats = extract_pe_features(str(img_path))
            if feats is None:
                failed += 1
                continue
            key = f"{family}/{img_path.name}"   # unique across families
            row_f = {'filename': key}
            row_f.update(dict(zip(feat_col_names, feats.tolist())))
            rows_feat.append(row_f)
            rows_label.append({
                'filename':   key,
                'label_name': family,
                'label_id':   family_to_id[family],
            })

    feat_df = pd.DataFrame(rows_feat)
    feat_df.to_csv(out_csv, index=False)

    label_df = pd.DataFrame(rows_label)
    label_df.to_csv(labels_csv, index=False)

    print(f"Saved {len(feat_df)} feature vectors to {out_csv}")
    print(f"Saved {len(label_df)} label entries to {labels_csv}")
    if failed:
        print(f"[WARN] {failed} files failed feature extraction")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract 508-dim image statistics from malware PNGs')
    parser.add_argument('--image_dir', required=True,
                        help='Root dir with FamilyName/ sub-folders of PNGs')
    parser.add_argument('--out',    required=True, help='Output features CSV')
    parser.add_argument('--labels', required=True, help='Output labels CSV')
    args = parser.parse_args()
    extract_directory(args.image_dir, args.out, args.labels)
