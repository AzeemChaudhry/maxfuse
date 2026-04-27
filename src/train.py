"""
MAXFUSE training loop.
Two-phase training:
  Phase 1 (epochs 1-20):  Freeze EfficientNet backbone — warm up heads
  Phase 2 (epochs 21-60): Unfreeze backbone — full fine-tuning

Logs to Weights & Biases.
"""

import os
import sys
import yaml
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from pathlib import Path

# Allow running from src/ directly
sys.path.insert(0, str(Path(__file__).parent))

from models.maxfuse import MAXFUSE
from losses.energy_margin_loss import EnergyMarginLoss
from data.dataset import get_dataloaders


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> MAXFUSE:
    m_cfg = cfg['model']
    return MAXFUSE(
        num_classes=m_cfg['num_classes'],
        img_dim=m_cfg['img_dim'],
        num_dim=m_cfg['num_dim'],
        shared_dim=m_cfg['shared_dim'],
        num_heads=m_cfg.get('num_heads', 4),
        mc_passes=m_cfg['mc_passes'],
        dropout=m_cfg['dropout'],
        use_n1=m_cfg.get('use_n1', True),
        use_n2=m_cfg.get('use_n2', True),
        use_n3=m_cfg.get('use_n3', True),
    )


def make_ood_batch(batch_size: int, device: str) -> tuple:
    """Generate a synthetic OOD batch using uniform noise."""
    img_ood  = torch.rand(batch_size, 1, 224, 224).to(device)
    feat_ood = torch.rand(batch_size, 80).to(device)
    return img_ood, feat_ood


def train_epoch(
    model:     MAXFUSE,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: EnergyMarginLoss,
    device:    str,
    ood_ratio: float = 0.5
) -> dict:
    model.train()
    total_loss = correct = total = 0

    for img, num_feats, labels in tqdm(loader, desc='  train', leave=False):
        img       = img.to(device)
        num_feats = num_feats.to(device)
        labels    = labels.to(device)

        ood_size = max(1, int(len(img) * ood_ratio))
        img_ood, feat_ood = make_ood_batch(ood_size, device)

        optimizer.zero_grad()

        logits_id  = model(img, num_feats)
        logits_ood = model(img_ood, feat_ood)

        loss, loss_dict = criterion(logits_id, labels, logits_ood)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict['loss_total'] * len(labels)
        preds   = logits_id.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += len(labels)

    return {
        'train_loss': total_loss / total,
        'train_acc':  correct / total,
    }


@torch.no_grad()
def validate(model: MAXFUSE, loader, criterion: EnergyMarginLoss, device: str) -> dict:
    model.eval()
    total_loss = correct = total = 0

    for img, num_feats, labels in tqdm(loader, desc='  val  ', leave=False):
        img       = img.to(device)
        num_feats = num_feats.to(device)
        labels    = labels.to(device)

        logits_id = model(img, num_feats)

        ood_size   = max(1, len(img) // 4)
        img_ood, feat_ood = make_ood_batch(ood_size, device)
        logits_ood = model(img_ood, feat_ood)

        loss, loss_dict = criterion(logits_id, labels, logits_ood)

        total_loss += loss_dict['loss_total'] * len(labels)
        preds   = logits_id.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += len(labels)

    return {
        'val_loss': total_loss / total,
        'val_acc':  correct / total,
    }


def run_training(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    wandb.init(
        project=cfg['logging']['wandb_project'],
        config=cfg,
        name=Path(config_path).stem
    )

    loaders = get_dataloaders(cfg)

    model = build_model(cfg).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    l_cfg = cfg['loss']
    criterion = EnergyMarginLoss(
        m_in=l_cfg['m_in'], m_out=l_cfg['m_out'], alpha=l_cfg['alpha']
    )

    t_cfg = cfg['training']
    epochs = t_cfg['epochs']

    # Phase 1: freeze backbone
    for param in model.image_encoder.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t_cfg['lr'],
        weight_decay=t_cfg['weight_decay']
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=t_cfg['warmup_epochs']
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=(epochs - t_cfg['warmup_epochs']), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[t_cfg['warmup_epochs']]
    )

    save_dir = Path(cfg['logging']['save_dir']) / Path(config_path).stem
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    phase2_started = False

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}  |  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Phase 2 switch at epoch 21
        if epoch == 21 and not phase2_started:
            print("  => Phase 2: unfreezing EfficientNet backbone")
            for param in model.image_encoder.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.image_encoder.backbone.parameters(), 'lr': 1e-5},
                {'params': [p for n, p in model.named_parameters()
                             if 'backbone' not in n], 'lr': t_cfg['lr']},
            ], weight_decay=t_cfg['weight_decay'])
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=(epochs - epoch + 1), eta_min=1e-7
            )
            scheduler = cosine_scheduler
            phase2_started = True

        train_metrics = train_epoch(model, loaders['train'], optimizer, criterion, device)
        val_metrics   = validate(model, loaders['val'], criterion, device)
        scheduler.step()

        log = {**train_metrics, **val_metrics, 'epoch': epoch,
               'lr': optimizer.param_groups[0]['lr']}
        wandb.log(log)
        print(f"  Train Acc: {train_metrics['train_acc']*100:.2f}%  |  "
              f"Val Acc: {val_metrics['val_acc']*100:.2f}%")

        if val_metrics['val_acc'] > best_val_acc:
            best_val_acc = val_metrics['val_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': cfg,
            }, save_dir / 'best.pt')
            print(f"  * Best model saved (val_acc={best_val_acc*100:.2f}%)")

    print("\nCalibrating OOD threshold on validation set...")
    checkpoint = torch.load(save_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    tau = model.calibrate_threshold(loaders['val'], device=device)
    checkpoint['tau'] = tau
    torch.save(checkpoint, save_dir / 'best.pt')
    print(f"Final tau = {tau:.4f} saved to checkpoint.")

    wandb.finish()
    print(f"\nTraining complete. Best val acc: {best_val_acc*100:.2f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run_training(args.config)
