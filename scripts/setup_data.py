"""
Download the Malimg dataset from Kaggle and organise it into
data/processed/images/<FamilyName>/*.png

Usage:
    python scripts/setup_data.py
"""

import shutil
import argparse
from pathlib import Path

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.pgm', '.gif'}


def all_images(directory: Path):
    """Yield all image files in directory (any supported extension)."""
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            yield f


def find_dataset_root(base_path: Path) -> Path:
    """
    Walk the download tree and return the directory that is the direct
    parent of the family sub-folders (i.e. each child dir contains images).
    Supports any nesting depth and any image extension.
    """
    print(f"Scanning downloaded path: {base_path}")

    # Collect every directory that contains at least one image file
    family_dirs = [
        d for d in base_path.rglob('*')
        if d.is_dir() and any(True for _ in all_images(d))
    ]

    if not family_dirs:
        # Show what IS in the directory to help debug
        print("WARNING: No image files found. Contents of downloaded path:")
        for item in sorted(base_path.rglob('*'))[:40]:
            print(f"  {item}")
        raise RuntimeError(
            f"No image files found under {base_path}.\n"
            "Manually place the Malimg family folders in data/processed/images/ "
            "and re-run without the download step."
        )

    parents = set(d.parent for d in family_dirs)
    if len(parents) == 1:
        root = parents.pop()
    else:
        root = max(parents, key=lambda p: sum(
            1 for d in p.iterdir()
            if d.is_dir() and any(True for _ in all_images(d))
        ))

    n_families = sum(1 for d in root.iterdir()
                     if d.is_dir() and any(True for _ in all_images(d)))
    print(f"Dataset root: {root}  ({n_families} family folders detected)")
    return root


def copy_dataset(src_root: Path, dest: Path):
    """Copy all image files from family sub-folders into dest."""
    families = sorted(d for d in src_root.iterdir() if d.is_dir())
    print(f"Copying {len(families)} families to {dest} ...")

    total = 0
    for family_dir in families:
        imgs = list(all_images(family_dir))
        if not imgs:
            continue
        out_dir = dest / family_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            # Always save as .png so the rest of the pipeline is consistent
            dst = out_dir / (img.stem + '.png')
            if not dst.exists():
                if img.suffix.lower() == '.png':
                    shutil.copy2(img, dst)
                else:
                    # Convert non-PNG formats to PNG via Pillow
                    from PIL import Image
                    Image.open(img).convert('L').save(dst)
                total += 1

    families_copied = [d.name for d in dest.iterdir() if d.is_dir()]
    print(f"Done. {total} images copied across {len(families_copied)} families.")
    print(f"Families: {families_copied[:5]} {'...' if len(families_copied) > 5 else ''}")


def main(dest: str = 'data/processed/images'):
    import kagglehub

    print("Downloading Malimg dataset from Kaggle (cached after first run)...")
    path = kagglehub.dataset_download("ikrambenabd/malimg-original")
    print(f"Download path: {path}")

    src_root = find_dataset_root(Path(path))

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Check if already populated
    existing = list(dest_path.glob('*/*.png'))
    if existing:
        print(f"Destination already has {len(existing)} images, skipping copy.")
    else:
        copy_dataset(src_root, dest_path)

    total_imgs = len(list(dest_path.glob('*/*.png')))
    print(f"\nDataset ready: {total_imgs} images in {dest_path.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', default='data/processed/images')
    args = parser.parse_args()
    main(args.dest)
