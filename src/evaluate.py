"""
MAXFUSE evaluation: closed-set accuracy + open-set OOD AUROC.
Produces Table 3 and Table 4 from the paper.
"""

import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.maxfuse import MAXFUSE
from models.energy_ood import compute_ood_auroc, compute_fpr95, energy_score, UNKNOWN_LABEL
from data.dataset import get_dataloaders, MalwareDataset
from torch.utils.data import DataLoader


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


@torch.no_grad()
def evaluate_closed_set(model, loader, device: str, family_names: list = None) -> dict:
    """Evaluate accuracy, macro F1, and per-class AUC on the test set."""
    all_preds, all_labels, all_probs = [], [], []

    for img, num_feats, labels in tqdm(loader, desc='Evaluating'):
        img       = img.to(device)
        num_feats = num_feats.to(device)

        v_i, v_n         = model.encode(img, num_feats)
        v_hat_i, v_hat_n = model.attend(v_i, v_n)
        z = 0.5 * v_hat_i + 0.5 * v_hat_n
        logits = model.classifier(z)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.append(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.vstack(all_probs)

    acc   = accuracy_score(all_labels, all_preds)
    f1    = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    auc   = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    print(f"\n{'='*50}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Macro F1:  {f1:.4f}")
    print(f"  Macro AUC: {auc:.4f}")
    print(f"{'='*50}\n")

    return {'accuracy': acc, 'macro_f1': f1, 'macro_auc': auc,
            'predictions': all_preds, 'labels': all_labels, 'probs': all_probs}


@torch.no_grad()
def collect_energies_loader(model, loader, device: str) -> np.ndarray:
    """Collect energy scores from a DataLoader."""
    energies = []
    for img, num_feats, _ in tqdm(loader, desc='Collecting energies'):
        img       = img.to(device)
        num_feats = num_feats.to(device)
        v_i, v_n         = model.encode(img, num_feats)
        v_hat_i, v_hat_n = model.attend(v_i, v_n)
        z = 0.5 * v_hat_i + 0.5 * v_hat_n
        logits = model.classifier(z)
        E = energy_score(logits)
        energies.append(E.cpu().numpy())
    return np.concatenate(energies)


@torch.no_grad()
def collect_energies_noise(model, n_samples: int, batch_size: int, device: str) -> np.ndarray:
    """Collect energy scores for synthetic uniform-noise OOD samples."""
    energies = []
    collected = 0
    while collected < n_samples:
        bsz = min(batch_size, n_samples - collected)
        img_ood  = torch.rand(bsz, 1, 224, 224, device=device)
        feat_ood = torch.rand(bsz, 80,         device=device)
        v_i, v_n         = model.encode(img_ood, feat_ood)
        v_hat_i, v_hat_n = model.attend(v_i, v_n)
        z = 0.5 * v_hat_i + 0.5 * v_hat_n
        logits = model.classifier(z)
        E = energy_score(logits)
        energies.append(E.cpu().numpy())
        collected += bsz
    return np.concatenate(energies)


@torch.no_grad()
def evaluate_ood(model, id_loader, ood_loader, device: str) -> dict:
    """
    Evaluate N3 OOD detection (Table 4).
    id_loader: in-distribution test samples
    ood_loader: held-out families (zero-day samples)
    """
    E_id  = collect_energies_loader(model, id_loader, device)
    E_ood = collect_energies_loader(model, ood_loader, device)

    auroc = compute_ood_auroc(E_id, E_ood)
    fpr95 = compute_fpr95(E_id, E_ood)

    print(f"\n{'='*50}")
    print(f"  OOD AUROC: {auroc:.4f}")
    print(f"  FPR@95:    {fpr95:.4f}")
    print(f"{'='*50}\n")

    return {'ood_auroc': auroc, 'fpr95': fpr95,
            'energy_id': E_id, 'energy_ood': E_ood}


def plot_confusion_matrix(labels, preds, family_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=family_names or range(len(cm)))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap='Blues')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def plot_energy_histogram(E_id, E_ood, tau, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(E_id,  bins=50, alpha=0.6, label='In-distribution', color='steelblue',  density=True)
    ax.hist(E_ood, bins=50, alpha=0.6, label='OOD (zero-day)',  color='salmon',     density=True)
    ax.axvline(tau, color='black', linestyle='--', label=f'tau = {tau:.2f}')
    ax.set_xlabel('Energy E(z)')
    ax.set_ylabel('Density')
    ax.set_title('Energy Score Distribution: In-dist vs OOD')
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Energy histogram saved to {save_path}")


def run_evaluation(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(checkpoint_path, cfg, device)

    loaders = get_dataloaders(cfg)

    # Derive sorted family names from the test split CSV
    test_df = pd.read_csv(cfg['data']['test_csv'])
    if 'label_name' in test_df.columns and 'label_id' in test_df.columns:
        fam_map = test_df.drop_duplicates('label_id').sort_values('label_id')
        family_names = fam_map['label_name'].tolist()
    else:
        family_names = None

    out_dir = Path('outputs/results')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Closed-Set Evaluation ===")
    cs_results = evaluate_closed_set(model, loaders['test'], device, family_names=family_names)

    # Confusion matrix
    plot_confusion_matrix(
        cs_results['labels'], cs_results['predictions'], family_names,
        save_path=str(out_dir / (Path(checkpoint_path).parent.name + '_confusion.png'))
    )

    ood_cfg = cfg.get('ood', {})
    ood_families = ood_cfg.get('held_out_families', [])
    ood_results = None
    if ood_families:
        print("=== OOD Evaluation ===")
        ood_csv = cfg['data'].get('ood_test_csv', None)
        if ood_csv and Path(ood_csv).exists():
            import torchvision.transforms as T
            eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
            ood_ds = MalwareDataset(
                ood_csv, cfg['data']['img_dir'],
                cfg['data']['features_csv'], transform=eval_tf)
            ood_loader = DataLoader(ood_ds, batch_size=cfg['training']['batch_size'],
                                    shuffle=False, num_workers=4)
            ood_results = evaluate_ood(model, loaders['test'], ood_loader, device)

            tau = model.tau.item()
            plot_energy_histogram(
                ood_results['energy_id'], ood_results['energy_ood'], tau,
                save_path=str(out_dir / 'energy_histogram.png')
            )
        else:
            # Fall back to synthetic uniform-noise OOD (same distribution used during training)
            print("  No ood_test_csv found — using synthetic uniform noise as OOD baseline.")
            n_test = len(loaders['test'].dataset)
            bsz    = cfg['training']['batch_size']
            E_id  = collect_energies_loader(model, loaders['test'], device)
            E_ood = collect_energies_noise(model, n_test, bsz, device)

            auroc = compute_ood_auroc(E_id, E_ood)
            fpr95 = compute_fpr95(E_id, E_ood)
            print(f"\n{'='*50}")
            print(f"  OOD AUROC (vs noise): {auroc:.4f}")
            print(f"  FPR@95   (vs noise): {fpr95:.4f}")
            print(f"{'='*50}\n")
            tau = model.tau.item()
            plot_energy_histogram(
                E_id, E_ood, tau,
                save_path=str(out_dir / 'energy_histogram_noise.png')
            )
            ood_results = {'ood_auroc': auroc, 'fpr95': fpr95,
                           'energy_id': E_id, 'energy_ood': E_ood}

    results_file = out_dir / (Path(checkpoint_path).parent.name + '.json')
    summary = {
        'accuracy':  cs_results['accuracy'],
        'macro_f1':  cs_results['macro_f1'],
        'macro_auc': cs_results['macro_auc'],
    }
    if ood_results is not None:
        summary['ood_auroc'] = ood_results['ood_auroc']
        summary['fpr95']     = ood_results['fpr95']

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    run_evaluation(args.config, args.checkpoint)
