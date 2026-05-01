#!/usr/bin/env bash
# MAXFUSE end-to-end pipeline runner
#
# Usage:
#   bash run.sh                    # full pipeline
#   bash run.sh --from-step train  # start from a specific step
#   bash run.sh --ablation         # also run ablation configs
#   bash run.sh --skip-tests       # skip pytest
#
# Steps: install | download | features | splits | nca | smote | verify | tests | train | evaluate
#
# Fallbacks: each data stage is skipped if its output already exists.
# Timeouts:  uses GNU `timeout` if available, otherwise runs without timeout.

set -euo pipefail
cd "$(dirname "$0")"

# ── Timeouts (seconds) ────────────────────────────────────────────────────────
T_INSTALL=300
T_DOWNLOAD=1200
T_FEATURES=7200
T_SPLITS=60
T_NCA=3600
T_SMOTE=600
T_VERIFY=120
T_TESTS=600
T_TRAIN=172800
T_EVAL=3600

# ── Argument parsing ──────────────────────────────────────────────────────────
FROM_STEP="install"
SKIP_TESTS=0
ABLATION=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-step) FROM_STEP="$2"; shift 2 ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --ablation) ABLATION=1; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

ALL_STEPS=(install download features splits nca smote verify tests train evaluate)

# ── Helpers ───────────────────────────────────────────────────────────────────

# Wrap a command with GNU timeout if available; otherwise run directly.
run_timed() {
    local timeout_s="$1"; shift
    if command -v timeout &>/dev/null; then
        timeout "$timeout_s" "$@"
    else
        "$@"
    fi
}

step_header() {
    echo ""
    echo "========================================================"
    echo "  $1"
    echo "========================================================"
}

has_images() {
    local dir="data/processed/images"
    [[ -d "$dir" ]] && [[ -n "$(find "$dir" -name '*.png' -maxdepth 3 2>/dev/null | head -1)" ]]
}

# Returns 0 if this step should be skipped, 1 if it should run.
should_skip() {
    case "$1" in
        download)
            if has_images; then
                n=$(find data/processed/images -name '*.png' 2>/dev/null | wc -l)
                echo "[SKIP] download — ${n} images already in data/processed/images/"
                return 0
            fi ;;
        features)
            if [[ -f "data/processed/features/pe_features_raw.csv" ]]; then
                echo "[SKIP] features — pe_features_raw.csv already exists"
                return 0
            fi ;;
        splits)
            if [[ -f "data/splits/train.csv" ]]; then
                echo "[SKIP] splits — train.csv already exists"
                return 0
            fi ;;
        nca)
            if [[ -f "data/processed/features/pe_features_nca.csv" ]]; then
                echo "[SKIP] nca — pe_features_nca.csv already exists"
                return 0
            fi ;;
        smote)
            if [[ -f "data/splits/train_balanced.csv" ]]; then
                echo "[SKIP] smote — train_balanced.csv already exists"
                return 0
            fi ;;
    esac
    return 1
}

# Check whether a step is at or after the starting step.
step_index() {
    local target="$1"
    for i in "${!ALL_STEPS[@]}"; do
        [[ "${ALL_STEPS[$i]}" == "$target" ]] && echo "$i" && return
    done
    echo "-1"
}

START_IDX=$(step_index "$FROM_STEP")
if [[ "$START_IDX" == "-1" ]]; then
    echo "Unknown step: $FROM_STEP"
    echo "Available: ${ALL_STEPS[*]}"
    exit 1
fi

should_run() {
    local idx
    idx=$(step_index "$1")
    [[ "$idx" -ge "$START_IDX" ]]
}

# ── Steps ─────────────────────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "  MAXFUSE — Full Pipeline"
echo "  Starting from : $FROM_STEP"
echo "  Ablation      : $([ $ABLATION -eq 1 ] && echo yes || echo no)"
echo "========================================================"

# 1. Install
if should_run install; then
    step_header "[install] Installing Python dependencies"
    run_timed $T_INSTALL pip install -r requirements.txt
fi

# 2. Download dataset
if should_run download; then
    step_header "[download] Downloading Malimg dataset from Kaggle"
    if ! should_skip download; then
        run_timed $T_DOWNLOAD python scripts/setup_data.py
    fi
fi

# 3. Extract features
if should_run features; then
    step_header "[features] Extracting 508-dim image statistics"
    mkdir -p data/processed/features data/splits outputs
    if ! should_skip features; then
        run_timed $T_FEATURES python src/data/pe_extractor.py \
            --image_dir data/processed/images \
            --out       data/processed/features/pe_features_raw.csv \
            --labels    data/processed/features/labels.csv
    fi
fi

# 4. Build splits
if should_run splits; then
    step_header "[splits] Building train/val/test split manifests (70/15/15)"
    mkdir -p data/splits
    if ! should_skip splits; then
        run_timed $T_SPLITS python -c "
import sys; sys.path.insert(0,'src')
from data.dataset import build_split_manifest
build_split_manifest('data/processed/images','data/splits',train=0.70,val=0.15,test=0.15,seed=42)
"
    fi
fi

# 5. NCA reduction
if should_run nca; then
    step_header "[nca] Dimensionality reduction (508 -> 80 dims)"
    mkdir -p outputs
    if ! should_skip nca; then
        run_timed $T_NCA python src/data/nca_selector.py \
            --in_csv       data/processed/features/pe_features_raw.csv \
            --labels_csv   data/processed/features/labels.csv \
            --train_csv    data/splits/train.csv \
            --out_csv      data/processed/features/pe_features_nca.csv \
            --model_path   outputs/nca_model.pkl \
            --n_components 80
    fi
fi

# 6. SMOTE balancing
if should_run smote; then
    step_header "[smote] SMOTE class balancing on training split"
    if ! should_skip smote; then
        run_timed $T_SMOTE python src/data/smote_balancer.py \
            --features_csv data/processed/features/pe_features_nca.csv \
            --train_csv    data/splits/train.csv \
            --labels_csv   data/processed/features/labels.csv \
            --out_csv      data/splits/train_balanced.csv
    fi
fi

# 7. Dataloader sanity check
if should_run verify; then
    step_header "[verify] Dataloader sanity check"
    run_timed $T_VERIFY python -c "
import sys, yaml; sys.path.insert(0,'src')
from data.dataset import get_dataloaders
cfg = yaml.safe_load(open('configs/base.yaml'))
loaders = get_dataloaders(cfg)
img, num, lbl = next(iter(loaders['train']))
print('  img:', img.shape, '  num:', num.shape)
assert img.shape[1:] == (1, 224, 224), 'Wrong image shape'
assert num.shape[1]  == 80,            'Wrong feature dim'
print('  Dataloaders OK')
"
fi

# 8. Unit tests
if should_run tests; then
    if [[ $SKIP_TESTS -eq 1 ]]; then
        echo ""
        echo "[SKIP] tests — --skip-tests flag set"
    else
        step_header "[tests] Running unit tests (36 tests)"
        run_timed $T_TESTS python -m pytest -v --tb=short
    fi
fi

# 9. Train full model
if should_run train; then
    step_header "[train] Training MAXFUSE full model (60 epochs)"
    run_timed $T_TRAIN python src/train.py --config configs/maxfuse_full.yaml
fi

# 10. Evaluate
if should_run evaluate; then
    CKPT="outputs/checkpoints/maxfuse_full/best.pt"
    if [[ ! -f "$CKPT" ]]; then
        echo ""
        echo "[WARN] evaluate skipped — checkpoint not found: $CKPT"
    else
        step_header "[evaluate] Evaluating on test set"
        run_timed $T_EVAL python src/evaluate.py \
            --config     configs/maxfuse_full.yaml \
            --checkpoint "$CKPT"
    fi
fi

# Ablation study (optional)
if [[ $ABLATION -eq 1 ]]; then
    echo ""
    echo "========================================================"
    echo "  ABLATION STUDY"
    echo "========================================================"
    for cfg in configs/base.yaml configs/ablation_no_n1.yaml \
               configs/ablation_no_n2.yaml configs/ablation_no_n3.yaml; do
        stem=$(basename "$cfg" .yaml)
        ckpt="outputs/checkpoints/${stem}/best.pt"
        step_header "[train] $cfg"
        run_timed $T_TRAIN python src/train.py --config "$cfg"
        step_header "[evaluate] $ckpt"
        run_timed $T_EVAL python src/evaluate.py --config "$cfg" --checkpoint "$ckpt"
    done
fi

echo ""
echo "========================================================"
echo "  Pipeline complete. Results in outputs/results/"
echo "========================================================"
