#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/base.yaml"
  "configs/ablation_no_n1.yaml"
  "configs/ablation_no_n2.yaml"
  "configs/ablation_no_n3.yaml"
  "configs/maxfuse_full.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "========================================"
  echo " Training: $cfg"
  echo "========================================"
  python src/train.py --config "$cfg"

  ckpt="outputs/checkpoints/$(basename $cfg .yaml)/best.pt"
  echo "  Evaluating: $ckpt"
  python src/evaluate.py --config "$cfg" --checkpoint "$ckpt"
done

echo ""
echo "All ablation runs complete. Results in outputs/results/"
