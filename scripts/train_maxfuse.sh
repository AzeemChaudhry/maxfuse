#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/maxfuse_full.yaml}"
echo "Training with config: $CONFIG"
python src/train.py --config "$CONFIG"
