#!/usr/bin/env bash
# MAXFUSE — full pipeline: setup -> data -> tests -> train -> evaluate
# Run from the maxfuse/ directory:
#   bash run_all.sh
set -euo pipefail

echo ""
echo "========================================"
echo "  MAXFUSE — Full Pipeline"
echo "========================================"

# ── 1. Install dependencies ──────────────────────────────────────────────────
echo ""
echo "[1/7] Installing dependencies..."
pip install -r requirements.txt

# ── 2. Download + organise Malimg dataset ────────────────────────────────────
echo ""
echo "[2/7] Downloading Malimg dataset from Kaggle..."
python scripts/setup_data.py

# ── 3. Data preparation pipeline ─────────────────────────────────────────────
echo ""
echo "[3/7] Running data preparation pipeline..."
bash scripts/prepare_data.sh

# ── 4. Sanity check dataloaders ──────────────────────────────────────────────
echo ""
echo "[4/7] Verifying dataloaders..."
python -c "
import sys, yaml
sys.path.insert(0, 'src')
from data.dataset import get_dataloaders
cfg = yaml.safe_load(open('configs/base.yaml'))
loaders = get_dataloaders(cfg)
img, num, lbl = next(iter(loaders['train']))
print('  img :', img.shape)
print('  num :', num.shape)
print('  labels (first 4):', lbl[:4].tolist())
assert img.shape[1:] == (1, 224, 224), 'Wrong image shape'
assert num.shape[1]  == 80,            'Wrong feature dim'
print('  Dataloaders OK')
"

# ── 5. Unit tests ─────────────────────────────────────────────────────────────
echo ""
echo "[5/7] Running unit tests..."
pytest

# ── 6. Train ──────────────────────────────────────────────────────────────────
echo ""
echo "[6/7] Training MAXFUSE (full model)..."
python src/train.py --config configs/maxfuse_full.yaml

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
echo ""
echo "[7/7] Evaluating on test set..."
python src/evaluate.py \
  --config     configs/maxfuse_full.yaml \
  --checkpoint outputs/checkpoints/maxfuse_full/best.pt

echo ""
echo "========================================"
echo "  Done. Results in outputs/results/"
echo "========================================"
