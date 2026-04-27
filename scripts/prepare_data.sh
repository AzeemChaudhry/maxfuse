#!/usr/bin/env bash
# Full data preparation pipeline for the Malimg image-only workflow.
# Run this AFTER setup_data.py has placed images in data/processed/images/.
#
# Usage:
#   bash scripts/prepare_data.sh
set -euo pipefail

IMG_DIR="data/processed/images"
FEAT_RAW="data/processed/features/pe_features_raw.csv"
LABELS="data/processed/features/labels.csv"
FEAT_NCA="data/processed/features/pe_features_nca.csv"
SPLIT_DIR="data/splits"

mkdir -p data/processed/features "$SPLIT_DIR" outputs

# Verify images exist
if [ ! -d "$IMG_DIR" ] || [ -z "$(ls -A $IMG_DIR 2>/dev/null)" ]; then
  echo "ERROR: $IMG_DIR is empty. Run 'python scripts/setup_data.py' first."
  exit 1
fi

echo "=== Step 1: Extract 508-dim image statistics (numeric branch features) ==="
python src/data/pe_extractor.py \
  --image_dir "$IMG_DIR" \
  --out       "$FEAT_RAW" \
  --labels    "$LABELS"

echo ""
echo "=== Step 2: Build stratified train/val/test split manifests (70/15/15) ==="
python -c "
import sys; sys.path.insert(0, 'src')
from data.dataset import build_split_manifest
build_split_manifest('$IMG_DIR', '$SPLIT_DIR', train=0.70, val=0.15, test=0.15, seed=42)
"

echo ""
echo "=== Step 3: NCA dimensionality reduction (508 -> 80 dims) ==="
python src/data/nca_selector.py \
  --in_csv      "$FEAT_RAW" \
  --labels_csv  "$LABELS" \
  --train_csv   "$SPLIT_DIR/train.csv" \
  --out_csv     "$FEAT_NCA" \
  --model_path  "outputs/nca_model.pkl" \
  --n_components 80

echo ""
echo "=== Step 4: SMOTE class balancing on training split ==="
python src/data/smote_balancer.py \
  --features_csv "$FEAT_NCA" \
  --train_csv    "$SPLIT_DIR/train.csv" \
  --labels_csv   "$LABELS" \
  --out_csv      "$SPLIT_DIR/train_balanced.csv"

echo ""
echo "=== Data preparation complete ==="
echo "    Images   : $IMG_DIR"
echo "    Features : $FEAT_NCA"
echo "    Splits   : $SPLIT_DIR/{train,val,test}.csv"
echo "    Balanced : $SPLIT_DIR/train_balanced.csv"
echo "    NCA model: outputs/nca_model.pkl"
