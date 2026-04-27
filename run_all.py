"""
MAXFUSE full pipeline runner.
Usage:
    python run_all.py
"""
import subprocess
import sys
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

def run(description, *args):
    print(f"\n{'='*50}")
    print(f"  {description}")
    print('='*50)
    cmd = [sys.executable] + list(args) if args[0].endswith('.py') else list(args)
    result = subprocess.run([sys.executable] + list(args))
    if result.returncode != 0:
        print(f"\nERROR: step failed -> {' '.join(args)}")
        sys.exit(result.returncode)

def run_inline(description, code):
    print(f"\n{'='*50}")
    print(f"  {description}")
    print('='*50)
    result = subprocess.run([sys.executable, '-c', code])
    if result.returncode != 0:
        print(f"\nERROR: step failed")
        sys.exit(result.returncode)

def run_cmd(description, cmd):
    print(f"\n{'='*50}")
    print(f"  {description}")
    print('='*50)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nERROR: step failed -> {cmd}")
        sys.exit(result.returncode)

# ── 1. Install dependencies
run_cmd("[1/7] Installing dependencies",
        f"{sys.executable} -m pip install -r requirements.txt")

# ── 2. Download + organise dataset
run("[2/7] Downloading Malimg dataset from Kaggle",
    "scripts/setup_data.py")

# ── 3. Extract 508-dim image statistics
Path("data/processed/features").mkdir(parents=True, exist_ok=True)
Path("data/splits").mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(parents=True, exist_ok=True)

run("[3a/7] Extracting 508-dim image statistics",
    "src/data/pe_extractor.py",
    "--image_dir", "data/processed/images",
    "--out",       "data/processed/features/pe_features_raw.csv",
    "--labels",    "data/processed/features/labels.csv")

# ── 4. Build splits
run_inline("[3b/7] Building train/val/test splits",
    "import sys; sys.path.insert(0,'src'); "
    "from data.dataset import build_split_manifest; "
    "build_split_manifest('data/processed/images','data/splits',"
    "train=0.70,val=0.15,test=0.15,seed=42)")

# ── 5. NCA reduction
run("[3c/7] NCA dimensionality reduction (508->80)",
    "src/data/nca_selector.py",
    "--in_csv",       "data/processed/features/pe_features_raw.csv",
    "--labels_csv",   "data/processed/features/labels.csv",
    "--train_csv",    "data/splits/train.csv",
    "--out_csv",      "data/processed/features/pe_features_nca.csv",
    "--model_path",   "outputs/nca_model.pkl",
    "--n_components", "80")

# ── 6. SMOTE balancing
run("[3d/7] SMOTE class balancing",
    "src/data/smote_balancer.py",
    "--features_csv", "data/processed/features/pe_features_nca.csv",
    "--train_csv",    "data/splits/train.csv",
    "--labels_csv",   "data/processed/features/labels.csv",
    "--out_csv",      "data/splits/train_balanced.csv")

# ── 7. Dataloader sanity check
run_inline("[4/7] Verifying dataloaders",
    "import sys, yaml; sys.path.insert(0,'src'); "
    "from data.dataset import get_dataloaders; "
    "cfg=yaml.safe_load(open('configs/base.yaml')); "
    "loaders=get_dataloaders(cfg); "
    "img,num,lbl=next(iter(loaders['train'])); "
    "print('img:',img.shape,'num:',num.shape); "
    "assert img.shape[1:]==(1,224,224); "
    "assert num.shape[1]==80; "
    "print('Dataloaders OK')")

# ── 8. Unit tests
run_cmd("[5/7] Running unit tests",
        f"{sys.executable} -m pytest")

# ── 9. Train
run("[6/7] Training MAXFUSE (60 epochs)",
    "src/train.py",
    "--config", "configs/maxfuse_full.yaml")

# ── 10. Evaluate
run("[7/7] Evaluating on test set",
    "src/evaluate.py",
    "--config",     "configs/maxfuse_full.yaml",
    "--checkpoint", "outputs/checkpoints/maxfuse_full/best.pt")

print("\n" + "="*50)
print("  Done. Results in outputs/results/")
print("="*50)
