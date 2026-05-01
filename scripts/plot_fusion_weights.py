"""
Visualise N2 MC-Dropout dynamic fusion weights across the test set.

For each test sample the model estimates per-branch uncertainty and assigns
inverse-variance weights (w_img, w_num).  This script plots their distribution,
showing which modality the model relies on more and how that varies per family.

Usage:
    python scripts/plot_fusion_weights.py \
        --config     configs/maxfuse_full.yaml \
        --checkpoint outputs/checkpoints/maxfuse_full/best.pt \
        --out_dir    outputs/results/
"""

import sys
import yaml
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.maxfuse import MAXFUSE
from data.dataset import get_dataloaders


def load_model(checkpoint_path: str, cfg: dict, device: str) -> MAXFUSE:
    m_cfg = cfg['model']
    model = MAXFUSE(
        num_classes=m_cfg['num_classes'],
        img_dim=m_cfg['img_dim'],
        num_dim=m_cfg['num_dim'],
        shared_dim=m_cfg['shared_dim'],
        mc_passes=m_cfg['mc_passes'],
        dropout=m_cfg['dropout'],
        use_n1=m_cfg.get('use_n1', True),
        use_n2=m_cfg.get('use_n2', True),
        use_n3=m_cfg.get('use_n3', True),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if 'tau' in ckpt:
        model.tau.fill_(ckpt['tau'])
    model.eval()
    return model


def collect_weights(model, loader, device: str):
    w_imgs, w_nums, labels = [], [], []
    for img, num_feats, lbl in tqdm(loader, desc='Collecting weights'):
        img, num_feats = img.to(device), num_feats.to(device)
        result = model.inference(img, num_feats)
        w_imgs.append(result['weights_i'].squeeze(-1).cpu().numpy())
        w_nums.append(result['weights_n'].squeeze(-1).cpu().numpy())
        labels.append(lbl.numpy())
    return (np.concatenate(w_imgs), np.concatenate(w_nums),
            np.concatenate(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out_dir',    default='outputs/results/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.checkpoint, cfg, device)

    if not cfg['model'].get('use_n2', True):
        print("N2 is disabled in this config — fusion weights are fixed at 0.5/0.5.")
        return

    loaders = get_dataloaders(cfg)
    w_i, w_n, labels = collect_weights(model, loaders['test'], device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Histogram of image branch weights
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(w_i, bins=50, alpha=0.7, color='steelblue',  label='w_image', density=True)
    ax.hist(w_n, bins=50, alpha=0.7, color='darkorange', label='w_numeric', density=True)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='equal weight')
    ax.set_xlabel('Fusion weight')
    ax.set_ylabel('Density')
    ax.set_title('N2 Dynamic Fusion Weight Distribution (Test Set)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(out_dir / 'fusion_weights_hist.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fusion_weights_hist.png'}")

    # 2. Per-family mean image weight (sorted)
    import pandas as pd
    import seaborn as sns

    test_df = pd.read_csv(cfg['data']['test_csv'])
    if 'label_name' in test_df.columns and 'label_id' in test_df.columns:
        fam_map = test_df.drop_duplicates('label_id').sort_values('label_id')
        family_names = fam_map['label_name'].tolist()
    else:
        family_names = [str(i) for i in range(cfg['model']['num_classes'])]

    df = pd.DataFrame({'label': labels, 'w_img': w_i, 'w_num': w_n})
    df['family'] = df['label'].map(lambda x: family_names[x] if x < len(family_names) else str(x))
    mean_wi = df.groupby('family')['w_img'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    mean_wi.plot(kind='bar', ax=ax, color='steelblue', alpha=0.85)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Mean image branch weight (w_img)')
    ax.set_title('Per-Family Mean Fusion Weight — Image Branch')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    fig.savefig(str(out_dir / 'fusion_weights_per_family.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'fusion_weights_per_family.png'}")

    # Summary
    print(f"\nMean w_img: {w_i.mean():.3f} +/- {w_i.std():.3f}")
    print(f"Mean w_num: {w_n.mean():.3f} +/- {w_n.std():.3f}")
    print(f"Families where image dominates (w_img > 0.6): "
          f"{(mean_wi > 0.6).sum()}/{len(mean_wi)}")


if __name__ == '__main__':
    main()
