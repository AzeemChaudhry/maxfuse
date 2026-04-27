# MAXFUSE: Multimodal Malware Classifier with Uncertainty-Aware Fusion and Open-Set Rejection

A PyTorch implementation of MAXFUSE, a dual-branch deep learning system for malware family classification. It fuses visual (grayscale binary visualisation via EfficientNet-B0) and statistical (508-dim image statistics via NCA-reduced MLP) representations of PE binaries, with three architectural novelties for robust real-world deployment.

---

## Architecture

```
PE Binary / PNG
      |
      +------ Image Branch --------> EfficientNet-B0 --> R^256
      |                                                       \
      +------ Numeric Branch ------> NCA (508->80)            +--> [N1] Cross-Modal
               + SMOTE               + MLP --> R^128          |         Attention
                                                             /
                                             R^128 (x2) ---+
                                                            |
                                             [N2] MC-Dropout Dynamic Fusion
                                                            |
                                                       Classifier
                                                      (25 families)
                                                            |
                                             [N3] Energy-Score OOD Rejection
                                                            |
                                              Family Label  OR  UNKNOWN
```

### Three Novelties

| ID | Name | Description |
|----|------|-------------|
| N1 | Cross-Modal Attention | Bidirectional `nn.MultiheadAttention` between image CNN features and numeric statistical features; learns which modality is more discriminative per sample |
| N2 | MC-Dropout Dynamic Fusion | T=10 stochastic forward passes estimate per-branch uncertainty; inverse-variance weights replace fixed 0.5/0.5 fusion |
| N3 | Energy-Score OOD Rejection | `E(z) = -T·logsumexp(z/T)`; threshold calibrated at FPR95 on validation set; unknown/zero-day samples returned as `UNKNOWN` |

---

## Dataset

**Malimg** — 9,458 PE malware samples across 25 families, distributed as 224x224 grayscale PNG visualisations (Nataraj et al., VizSec 2011).

Download via Kaggle (handled automatically by the pipeline):
```
https://www.kaggle.com/datasets/ikrambenabd/malimg-original
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/maxfuse.git
cd maxfuse
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.1+. For GPU training (strongly recommended):
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Quickstart

```bash
# Full pipeline: download data -> features -> train -> evaluate
python run_all.py
```

Or step by step:

```bash
# 1. Download and organise Malimg dataset
python scripts/setup_data.py

# 2. Extract features, build splits, NCA reduction, SMOTE balancing
bash scripts/prepare_data.sh        # Linux/Mac
# -- or on Windows --
python run_all.py                   # runs only data steps if training already done

# 3. Train (60 epochs, two-phase)
python src/train.py --config configs/maxfuse_full.yaml

# 4. Evaluate (accuracy, F1, AUC, OOD AUROC, FPR@95)
python src/evaluate.py \
  --config     configs/maxfuse_full.yaml \
  --checkpoint outputs/checkpoints/maxfuse_full/best.pt

# 5. Single-sample inference
python src/infer.py \
  --binary  path/to/sample.png \
  --config  configs/maxfuse_full.yaml \
  --checkpoint outputs/checkpoints/maxfuse_full/best.pt \
  --nca_model  outputs/nca_model.pkl
```

---

## Ablation Study

```bash
bash scripts/run_ablation.sh
```

Trains and evaluates five configurations: full model, no-N1, no-N2, no-N3, and base. Results saved to `outputs/results/`.

| Config | N1 | N2 | N3 |
|--------|----|----|-----|
| `maxfuse_full.yaml` | On | On | On |
| `ablation_no_n1.yaml` | Off | On | On |
| `ablation_no_n2.yaml` | On | Off | On |
| `ablation_no_n3.yaml` | On | On | Off |
| `base.yaml` | On | On | On |

---

## Project Structure

```
maxfuse/
├── configs/                    # YAML experiment configs
│   ├── base.yaml
│   ├── maxfuse_full.yaml
│   └── ablation_*.yaml
├── scripts/
│   ├── setup_data.py           # Kaggle download + organise
│   ├── prepare_data.sh         # Feature extraction pipeline
│   ├── train_maxfuse.sh
│   └── run_ablation.sh
├── src/
│   ├── data/
│   │   ├── pe_extractor.py     # 508-dim image statistics extractor
│   │   ├── nca_selector.py     # NCA 508->80 reduction (Eq. 3)
│   │   ├── smote_balancer.py   # SMOTE class balancing (Eq. 4)
│   │   ├── binary_to_image.py  # PE bytes -> grayscale image (Eq. 1)
│   │   └── dataset.py          # PyTorch Dataset + DataLoaders
│   ├── models/
│   │   ├── image_encoder.py    # EfficientNet-B0 -> R^256
│   │   ├── numeric_encoder.py  # MLP 80->128
│   │   ├── cross_modal_attention.py  # N1
│   │   ├── dynamic_fusion.py   # N2 MC-Dropout
│   │   ├── energy_ood.py       # N3 energy score
│   │   └── maxfuse.py          # Full model assembly
│   ├── losses/
│   │   └── energy_margin_loss.py  # CE + energy margin (Eq. 5)
│   ├── explain/
│   │   ├── gradcam.py          # Grad-CAM on EfficientNet last block
│   │   └── shap_explainer.py   # SHAP KernelExplainer on numeric branch
│   ├── train.py                # Two-phase training loop
│   ├── evaluate.py             # Closed-set + OOD evaluation
│   └── infer.py                # Single-sample inference + XAI
├── tests/                      # pytest unit tests (36 tests)
├── run_all.py                  # One-command full pipeline
└── requirements.txt
```

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 60 (phase 1: 1-20 frozen backbone, phase 2: 21-60 unfrozen) |
| Batch size | 64 |
| Optimiser | AdamW |
| LR (heads) | 1e-4 |
| LR (backbone, phase 2) | 1e-5 |
| Weight decay | 1e-4 |
| Warmup | 5 epochs linear |
| Scheduler | CosineAnnealingLR |
| Grad clip | 1.0 |
| MC passes (T) | 10 |
| Energy margins | m_in=-25, m_out=-7, alpha=0.1 |
| Label smoothing | 0.1 |

---

## Tests

```bash
pytest
```

36 unit tests covering all model components, data utilities, and OOD metrics.

---

## References

- Nataraj et al., *Malware Images: Visualization and Automatic Classification*, VizSec 2011
- Nazim et al., *Multimodal Malware Classification*, Scientific Reports 2025
- Liu et al., *Energy-Based Out-of-Distribution Detection*, NeurIPS 2020
- Gal & Ghahramani, *Dropout as a Bayesian Approximation*, ICML 2016
- Goldberger et al., *Neighbourhood Components Analysis*, NeurIPS 2004
