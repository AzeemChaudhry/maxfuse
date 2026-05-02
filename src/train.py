"""
MAXFUSE training loop.
Two-phase training:
  Phase 1 (epochs 1-20):  Freeze EfficientNet backbone — warm up heads
  Phase 2 (epochs 21-60): Unfreeze backbone — full fine-tuning

Supports resuming from last.pt via --resume flag.
Logs to Weights & Biases.
"""

import os
import sys
import yaml
import wandb
import torch

# Reduce CUDA memory fragmentation (safe on all PyTorch 2.x builds)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from pathlib import Path

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
    img_ood  = torch.rand(batch_size, 1, 224, 224).to(device)
    feat_ood = torch.rand(batch_size, 80).to(device)
    return img_ood, feat_ood


def train_epoch(model, loader, optimizer, criterion, scaler, device, ood_ratio=0.5) -> dict:
    model.train()
    total_loss = correct = total = 0

    for img, num_feats, labels in tqdm(loader, desc='  train', leave=False):
        img       = img.to(device, non_blocking=True)
        num_feats = num_feats.to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True)

        ood_size = max(1, int(len(img) * ood_ratio))
        img_ood, feat_ood = make_ood_batch(ood_size, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
            logits_id  = model(img, num_feats)
            logits_ood = model(img_ood, feat_ood)
            loss, loss_dict = criterion(logits_id, labels, logits_ood)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_dict['loss_total'] * len(labels)
        correct    += (logits_id.argmax(dim=-1) == labels).sum().item()
        total      += len(labels)

    return {'train_loss': total_loss / total, 'train_acc': correct / total}


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = correct = total = 0

    for img, num_feats, labels in tqdm(loader, desc='  val  ', leave=False):
        img       = img.to(device, non_blocking=True)
        num_feats = num_feats.to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True)

        with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
            logits_id  = model(img, num_feats)
            ood_size   = max(1, len(img) // 4)
            img_ood, feat_ood = make_ood_batch(ood_size, device)
            logits_ood = model(img_ood, feat_ood)
            loss, loss_dict = criterion(logits_id, labels, logits_ood)

        total_loss += loss_dict['loss_total'] * len(labels)
        correct    += (logits_id.argmax(dim=-1) == labels).sum().item()
        total      += len(labels)

    return {'val_loss': total_loss / total, 'val_acc': correct / total}


def _build_phase1_optimizer(model, t_cfg):
    for p in model.image_encoder.backbone.parameters():
        p.requires_grad = False
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t_cfg['lr'], weight_decay=t_cfg['weight_decay']
    )


def _build_phase2_optimizer(model, t_cfg):
    for p in model.image_encoder.backbone.parameters():
        p.requires_grad = True
    return optim.AdamW([
        {'params': model.image_encoder.backbone.parameters(), 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
         'lr': t_cfg['lr']},
    ], weight_decay=t_cfg['weight_decay'])


def run_training(config_path: str, resume: bool = False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test CUDA compatibility (especially for P100 on Kaggle)
    if device == 'cuda':
        try:
            torch.zeros(1, device=device)
            print(f"✓ CUDA device ready: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            if 'no kernel image' in str(e) or 'not compatible' in str(e):
                print("[WARN] GPU CUDA incompatibility detected (likely P100 on Kaggle).")
                print("       PyTorch was built for sm_70+ but P100 is sm_60.")
                print("\n[FIX] On Kaggle, run this in a cell BEFORE training:")
                print("       !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                print("\n[ALT] Or fall back to CPU (slower but will work).")
                print("       Falling back to CPU now...")
                device = 'cpu'
            else:
                raise
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    t_cfg    = cfg['training']
    epochs   = t_cfg['epochs']
    save_dir = Path(cfg['logging']['save_dir']) / Path(config_path).stem
    save_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = save_dir / 'last.pt'

    loaders   = get_dataloaders(cfg)
    model     = build_model(cfg).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    criterion = EnergyMarginLoss(
        m_in=cfg['loss']['m_in'],
        m_out=cfg['loss']['m_out'],
        alpha=cfg['loss']['alpha']
    )
    scaler = GradScaler(enabled=(device == 'cuda'))

    # ── Resume state ─────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_acc  = 0.0
    phase2_started = False

    if resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch    = ckpt['epoch'] + 1
        best_val_acc   = ckpt.get('best_val_acc', 0.0)
        phase2_started = ckpt.get('phase2_started', False)
        print(f"Resuming from epoch {start_epoch}  "
              f"(best val_acc so far: {best_val_acc*100:.2f}%)")
    elif resume:
        print(f"[WARN] --resume set but no last.pt found at {last_ckpt}. Starting fresh.")

    # ── Build optimizer for the correct phase ─────────────────────────────────
    if phase2_started or start_epoch > 20:
        phase2_started = True
        torch.cuda.empty_cache()
        optimizer = _build_phase2_optimizer(model, t_cfg)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=(epochs - start_epoch + 1), eta_min=1e-7
        )
    else:
        optimizer = _build_phase1_optimizer(model, t_cfg)
        warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                             total_iters=t_cfg['warmup_epochs'])
        cosine    = CosineAnnealingLR(
            optimizer, T_max=(epochs - t_cfg['warmup_epochs']), eta_min=1e-6
        )
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                 milestones=[t_cfg['warmup_epochs']])
        # Step scheduler forward to match the epoch we're resuming from
        for _ in range(start_epoch - 1):
            scheduler.step()

    # Restore optimizer state if available and phase matches
    if resume and last_ckpt.exists() and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("  Optimizer state restored.")
        except Exception as e:
            print(f"  [WARN] Could not restore optimizer state ({e}). Using fresh optimizer.")

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_kwargs = dict(
        project=cfg['logging']['wandb_project'],
        config=cfg,
        name=Path(config_path).stem,
        resume='allow' if resume else None,
        settings=wandb.Settings(
            _disable_stats=True,
        )
    )
    if os.environ.get('KAGGLE_URL_BASE') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        wandb_kwargs['mode'] = 'disabled'

    wandb.init(**wandb_kwargs)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}  |  LR: {scheduler.get_last_lr()[0]:.2e}")

        if epoch == 21 and not phase2_started:
            print("  => Phase 2: unfreezing EfficientNet backbone")
            torch.cuda.empty_cache()
            optimizer = _build_phase2_optimizer(model, t_cfg)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=(epochs - epoch + 1), eta_min=1e-7
            )
            phase2_started = True

        train_metrics = train_epoch(model, loaders['train'], optimizer, criterion, scaler, device)
        val_metrics   = validate(model, loaders['val'], criterion, device)
        scheduler.step()

        log = {**train_metrics, **val_metrics, 'epoch': epoch,
               'lr': optimizer.param_groups[0]['lr']}
        wandb.log(log)
        print(f"  Train Acc: {train_metrics['train_acc']*100:.2f}%  |  "
              f"Val Acc: {val_metrics['val_acc']*100:.2f}%")

        # Save last.pt every epoch for resume support
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc':         best_val_acc,
            'phase2_started':       phase2_started,
            'config':               cfg,
        }, last_ckpt)

        if val_metrics['val_acc'] > best_val_acc:
            best_val_acc = val_metrics['val_acc']
            ckpt_data = {
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':              best_val_acc,
                'config':               cfg,
            }
            torch.save(ckpt_data, save_dir / 'best.pt')
            print(f"  * Best model saved (val_acc={best_val_acc*100:.2f}%)")

    # ── OOD threshold calibration ─────────────────────────────────────────────
    print("\nCalibrating OOD threshold on validation set...")
    best = torch.load(save_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    tau = model.calibrate_threshold(loaders['val'], device=device)
    best['tau'] = tau
    torch.save(best, save_dir / 'best.pt')
    print(f"Final tau = {tau:.4f} saved to checkpoint.")

    wandb.finish()
    print(f"\nTraining complete. Best val acc: {best_val_acc*100:.2f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last.pt if it exists')
    args = parser.parse_args()
    run_training(args.config, resume=args.resume)
