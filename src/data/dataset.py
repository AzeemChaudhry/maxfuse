"""
PyTorch Dataset and DataLoader for MAXFUSE.
Loads (image, numeric_features, label) triples.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split


class MalwareDataset(Dataset):
    """
    Dataset that returns (image_tensor, numeric_tensor, label) for each sample.

    Expected directory structure:
        image_dir/
            FamilyName1/
                sample_001.png
                ...
            FamilyName2/
                ...

    Args:
        split_csv: CSV with columns [filename, label_name, label_id]
        image_dir: Root directory of 224x224 grayscale PNGs
        features_csv: CSV with columns [filename, nca_0, ..., nca_79]
        transform: Optional torchvision transform for images
    """

    def __init__(
        self,
        split_csv: str,
        image_dir: str,
        features_csv: str,
        transform=None
    ):
        self.split_df = pd.read_csv(split_csv)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # If split CSV already contains NCA columns (SMOTE-balanced output),
        # use them directly; otherwise fall back to a separate features_csv lookup.
        embedded = [c for c in self.split_df.columns if c.startswith('nca_')]
        self._embedded = bool(embedded)

        if self._embedded:
            self.feat_cols = embedded
        else:
            feat_df = pd.read_csv(features_csv)
            feat_cols = [c for c in feat_df.columns if c != 'filename']
            self.feat_lookup = feat_df.set_index('filename')[feat_cols]
            self.feat_cols = feat_cols

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        filename   = row['filename']
        label_name = row['label_name']
        label_id   = int(row['label_id'])

        # -- Image --
        # filename is 'FamilyName/basename.ext'; search for it regardless of extension
        img_path = self.image_dir / label_name / Path(filename).name
        if not img_path.exists():
            # Extension may differ from what was stored in the CSV — find the file
            for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.pgm'):
                candidate = self.image_dir / label_name / (Path(filename).stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)

        # -- Numeric features --
        if self._embedded:
            feats = row[self.feat_cols].values.astype(np.float32)
        else:
            try:
                # filename is 'FamilyName/basename.png' — matches the key in features CSV
                feats = self.feat_lookup.loc[filename, self.feat_cols].values.astype(np.float32)
            except KeyError:
                feats = np.zeros(len(self.feat_cols), dtype=np.float32)

        numeric = torch.tensor(feats, dtype=torch.float32)

        return img, numeric, label_id


def build_split_manifest(
    image_dir: str,
    split_dir: str,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42
):
    """
    Build train/val/test split CSV files from an image directory.
    Stratified by family label.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1"
    Path(split_dir).mkdir(parents=True, exist_ok=True)

    families = sorted([d.name for d in Path(image_dir).iterdir() if d.is_dir()])
    family_to_id = {f: i for i, f in enumerate(families)}

    IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.pgm')
    rows = []
    for family in families:
        family_dir = Path(image_dir) / family
        imgs = sorted(p for ext in IMAGE_EXTS for p in family_dir.glob(f'*{ext}'))
        for img_path in imgs:
            rows.append({
                'filename':   f"{family}/{img_path.name}",
                'label_name': family,
                'label_id':   family_to_id[family],
            })

    if not rows:
        raise RuntimeError(
            f"No images found in {image_dir}. "
            "Run 'python scripts/setup_data.py' first to populate the image directory."
        )

    df = pd.DataFrame(rows)
    labels = df['label_id'].values

    train_df, temp_df = train_test_split(
        df, test_size=(val + test), stratify=labels, random_state=seed)
    val_ratio_of_temp = val / (val + test)
    temp_labels = temp_df['label_id'].values
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_of_temp),
        stratify=temp_labels, random_state=seed)

    train_df.to_csv(Path(split_dir) / 'train.csv', index=False)
    val_df.to_csv(Path(split_dir) / 'val.csv', index=False)
    test_df.to_csv(Path(split_dir) / 'test.csv', index=False)
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


def get_dataloaders(config: dict) -> dict:
    """
    Build DataLoaders from config dict.

    Expected config keys:
        data.img_dir, data.train_csv (or train_balanced_csv), data.val_csv,
        data.test_csv, data.features_csv, training.batch_size, data.num_workers
    """
    import torchvision.transforms as T

    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
    ])
    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    data_cfg = config['data']
    train_split = data_cfg.get('train_balanced_csv', data_cfg['train_csv'])

    datasets = {
        'train': MalwareDataset(train_split, data_cfg['img_dir'],
                                data_cfg['features_csv'], transform=train_transform),
        'val':   MalwareDataset(data_cfg['val_csv'], data_cfg['img_dir'],
                                data_cfg['features_csv'], transform=eval_transform),
        'test':  MalwareDataset(data_cfg['test_csv'], data_cfg['img_dir'],
                                data_cfg['features_csv'], transform=eval_transform),
    }

    # num_workers=0 is fastest on Windows (avoids process-spawn overhead per epoch)
    num_workers = data_cfg.get('num_workers', 0)
    pin_memory  = (num_workers > 0)   # pin_memory only helps with background workers

    loaders = {
        split: DataLoader(
            ds,
            batch_size=config['training']['batch_size'],
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            drop_last=(split == 'train')
        )
        for split, ds in datasets.items()
    }
    return loaders
