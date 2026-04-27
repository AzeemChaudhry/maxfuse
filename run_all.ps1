$ErrorActionPreference = "Stop"

Write-Host "========================================"
Write-Host "  MAXFUSE - Full Pipeline"
Write-Host "========================================"

Write-Host ""
Write-Host "[1/7] Installing dependencies..."
pip install -r requirements.txt

Write-Host ""
Write-Host "[2/7] Downloading Malimg dataset from Kaggle..."
python scripts/setup_data.py

Write-Host ""
Write-Host "[3/7] Running data preparation pipeline..."

New-Item -ItemType Directory -Force -Path "data\processed\features" | Out-Null
New-Item -ItemType Directory -Force -Path "data\splits" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null

Write-Host "  [3a] Extracting 508-dim image statistics..."
python src/data/pe_extractor.py --image_dir data/processed/images --out data/processed/features/pe_features_raw.csv --labels data/processed/features/labels.csv

Write-Host "  [3b] Building train/val/test splits..."
python -c "import sys; sys.path.insert(0, 'src'); from data.dataset import build_split_manifest; build_split_manifest('data/processed/images', 'data/splits', train=0.70, val=0.15, test=0.15, seed=42)"

Write-Host "  [3c] NCA dimensionality reduction (508 -> 80)..."
python src/data/nca_selector.py --in_csv data/processed/features/pe_features_raw.csv --labels_csv data/processed/features/labels.csv --train_csv data/splits/train.csv --out_csv data/processed/features/pe_features_nca.csv --model_path outputs/nca_model.pkl --n_components 80

Write-Host "  [3d] SMOTE class balancing..."
python src/data/smote_balancer.py --features_csv data/processed/features/pe_features_nca.csv --train_csv data/splits/train.csv --labels_csv data/processed/features/labels.csv --out_csv data/splits/train_balanced.csv

Write-Host ""
Write-Host "[4/7] Verifying dataloaders..."
python -c "import sys, yaml; sys.path.insert(0, 'src'); from data.dataset import get_dataloaders; cfg = yaml.safe_load(open('configs/base.yaml')); loaders = get_dataloaders(cfg); img, num, lbl = next(iter(loaders['train'])); print('img:', img.shape, ' num:', num.shape); assert img.shape[1:] == (1,224,224); assert num.shape[1] == 80; print('Dataloaders OK')"

Write-Host ""
Write-Host "[5/7] Running unit tests..."
pytest

Write-Host ""
Write-Host "[6/7] Training MAXFUSE (60 epochs)..."
python src/train.py --config configs/maxfuse_full.yaml

Write-Host ""
Write-Host "[7/7] Evaluating on test set..."
python src/evaluate.py --config configs/maxfuse_full.yaml --checkpoint outputs/checkpoints/maxfuse_full/best.pt

Write-Host ""
Write-Host "========================================"
Write-Host "  Done. Results in outputs/results/"
Write-Host "========================================"
