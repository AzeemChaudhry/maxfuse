"""
MAXFUSE end-to-end pipeline runner.

Runs every stage in order with:
  - Skip-if-done fallbacks for all data stages
  - Per-step timeouts
  - Clear error messages on failure

Usage:
    python run.py                        # full pipeline
    python run.py --from-step train      # start from a specific step
    python run.py --ablation             # also run all ablation configs after main training
    python run.py --skip-tests           # skip pytest

Steps: install | download | features | splits | nca | smote | verify | tests | train | evaluate
"""

import subprocess
import sys
import os
import argparse
import time
from pathlib import Path

# ── Resolve project root regardless of where the script is invoked from ──────
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# ── Timeouts (seconds) ────────────────────────────────────────────────────────
TIMEOUTS = {
    'install':   300,      #  5 min  — pip
    'download':  1200,     # 20 min  — Kaggle download + copy
    'features':  7200,     #  2 hr   — 508-dim extraction over ~9k images
    'splits':    60,       #  1 min  — CSV splitting
    'nca':       3600,     #  1 hr   — NCA optimisation
    'smote':     600,      # 10 min  — SMOTE oversampling
    'verify':    120,      #  2 min  — dataloader sanity check
    'tests':     600,      # 10 min  — 36 unit tests
    'train':     172800,   # 48 hr   — 60-epoch GPU training
    'evaluate':  3600,     #  1 hr   — closed-set + OOD eval
}

# ── Fallback guards — step is skipped if ALL listed paths already exist ───────
SKIP_IF_EXISTS = {
    'download': ['data/processed/images'],
    'features': ['data/processed/features/pe_features_raw.csv'],
    'splits':   ['data/splits/train.csv'],
    'nca':      ['data/processed/features/pe_features_nca.csv'],
    'smote':    ['data/splits/train_balanced.csv'],
}


def _has_images(directory: str) -> bool:
    d = ROOT / directory
    return d.is_dir() and any(d.rglob('*/*.png'))


def _should_skip(step: str) -> bool:
    guards = SKIP_IF_EXISTS.get(step, [])
    if not guards:
        return False
    if step == 'download':
        return _has_images(guards[0])
    return all((ROOT / p).exists() for p in guards)


def _run(step: str, cmd: list, timeout: int):
    """Run a subprocess command with a timeout. Exit on failure."""
    print(f"\n{'='*56}")
    print(f"  {step.upper()}")
    print(f"  cmd : {' '.join(cmd)}")
    print(f"  timeout: {timeout}s")
    print('='*56)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        elapsed = int(time.time() - t0)
        print(f"\n[TIMEOUT] Step '{step}' exceeded {timeout}s ({elapsed}s elapsed). Aborting.")
        sys.exit(2)
    elapsed = int(time.time() - t0)
    if result.returncode != 0:
        print(f"\n[FAILED] Step '{step}' exited with code {result.returncode} after {elapsed}s.")
        sys.exit(result.returncode)
    print(f"  done in {elapsed}s")


def _run_inline(step: str, code: str, timeout: int):
    _run(step, [sys.executable, '-c', code], timeout)


def step_install():
    _run('install', [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
         TIMEOUTS['install'])


def step_download():
    if _should_skip('download'):
        n = sum(1 for _ in (ROOT / 'data/processed/images').rglob('*/*.png'))
        print(f"\n[SKIP] download — {n} images already in data/processed/images/")
        return
    _run('download', [sys.executable, 'scripts/setup_data.py'], TIMEOUTS['download'])


def step_features():
    if _should_skip('features'):
        print("\n[SKIP] features — pe_features_raw.csv already exists")
        return
    Path('data/processed/features').mkdir(parents=True, exist_ok=True)
    _run('features',
         [sys.executable, 'src/data/pe_extractor.py',
          '--image_dir', 'data/processed/images',
          '--out',       'data/processed/features/pe_features_raw.csv',
          '--labels',    'data/processed/features/labels.csv'],
         TIMEOUTS['features'])


def step_splits():
    if _should_skip('splits'):
        print("\n[SKIP] splits — train.csv already exists")
        return
    Path('data/splits').mkdir(parents=True, exist_ok=True)
    _run_inline('splits',
        "import sys; sys.path.insert(0,'src'); "
        "from data.dataset import build_split_manifest; "
        "build_split_manifest('data/processed/images','data/splits',"
        "train=0.70,val=0.15,test=0.15,seed=42)",
        TIMEOUTS['splits'])


def step_nca():
    if _should_skip('nca'):
        print("\n[SKIP] nca — pe_features_nca.csv already exists")
        return
    Path('outputs').mkdir(parents=True, exist_ok=True)
    _run('nca',
         [sys.executable, 'src/data/nca_selector.py',
          '--in_csv',       'data/processed/features/pe_features_raw.csv',
          '--labels_csv',   'data/processed/features/labels.csv',
          '--train_csv',    'data/splits/train.csv',
          '--out_csv',      'data/processed/features/pe_features_nca.csv',
          '--model_path',   'outputs/nca_model.pkl',
          '--n_components', '80'],
         TIMEOUTS['nca'])


def step_smote():
    if _should_skip('smote'):
        print("\n[SKIP] smote — train_balanced.csv already exists")
        return
    _run('smote',
         [sys.executable, 'src/data/smote_balancer.py',
          '--features_csv', 'data/processed/features/pe_features_nca.csv',
          '--train_csv',    'data/splits/train.csv',
          '--labels_csv',   'data/processed/features/labels.csv',
          '--out_csv',      'data/splits/train_balanced.csv'],
         TIMEOUTS['smote'])


def step_verify():
    _run_inline('verify',
        "import sys, yaml; sys.path.insert(0,'src'); "
        "from data.dataset import get_dataloaders; "
        "cfg=yaml.safe_load(open('configs/base.yaml')); "
        "loaders=get_dataloaders(cfg); "
        "img,num,lbl=next(iter(loaders['train'])); "
        "print('  img:',img.shape,'  num:',num.shape); "
        "assert img.shape[1:]==(1,224,224); "
        "assert num.shape[1]==80; "
        "print('  Dataloaders OK')",
        TIMEOUTS['verify'])


def step_tests():
    _run('tests', [sys.executable, '-m', 'pytest', '-v', '--tb=short'],
         TIMEOUTS['tests'])


def step_train(config: str = 'configs/maxfuse_full.yaml', resume: bool = False):
    cmd = [sys.executable, 'src/train.py', '--config', config]
    if resume:
        cmd.append('--resume')
    _run('train', cmd, TIMEOUTS['train'])


def step_evaluate(config: str = 'configs/maxfuse_full.yaml'):
    ckpt = f"outputs/checkpoints/{Path(config).stem}/best.pt"
    if not (ROOT / ckpt).exists():
        print(f"\n[WARN] evaluate skipped — checkpoint not found: {ckpt}")
        return
    _run('evaluate',
         [sys.executable, 'src/evaluate.py', '--config', config, '--checkpoint', ckpt],
         TIMEOUTS['evaluate'])


# ── Step ordering ─────────────────────────────────────────────────────────────

ALL_STEPS = ['install', 'download', 'features', 'splits', 'nca', 'smote',
             'verify', 'tests', 'train', 'evaluate']

ABLATION_CONFIGS = [
    'configs/base.yaml',
    'configs/ablation_no_n1.yaml',
    'configs/ablation_no_n2.yaml',
    'configs/ablation_no_n3.yaml',
]


def main():
    parser = argparse.ArgumentParser(
        description='MAXFUSE end-to-end pipeline runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available steps: {', '.join(ALL_STEPS)}"
    )
    parser.add_argument('--from-step', metavar='STEP', default='install',
                        choices=ALL_STEPS,
                        help='Start execution from this step (default: install)')
    parser.add_argument('--skip-tests', action='store_true',
                        help='Skip the pytest step')
    parser.add_argument('--ablation', action='store_true',
                        help='After main training, also train all ablation configs')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last saved checkpoint (last.pt)')
    args = parser.parse_args()

    start_idx = ALL_STEPS.index(args.from_step)
    steps_to_run = ALL_STEPS[start_idx:]

    print()
    print('='*56)
    print('  MAXFUSE — Full Pipeline')
    print(f"  Starting from : {args.from_step}")
    print(f"  Resume        : {'yes' if args.resume else 'no'}")
    print(f"  Ablation      : {'yes' if args.ablation else 'no'}")
    print('='*56)

    step_fns = {
        'install':  step_install,
        'download': step_download,
        'features': step_features,
        'splits':   step_splits,
        'nca':      step_nca,
        'smote':    step_smote,
        'verify':   step_verify,
        'tests':    step_tests,
        'train':    lambda: step_train('configs/maxfuse_full.yaml', resume=args.resume),
        'evaluate': lambda: step_evaluate('configs/maxfuse_full.yaml'),
    }

    for step in steps_to_run:
        if step == 'tests' and args.skip_tests:
            print(f"\n[SKIP] tests — --skip-tests flag set")
            continue
        step_fns[step]()

    if args.ablation:
        print('\n' + '='*56)
        print('  ABLATION STUDY')
        print('='*56)
        for cfg in ABLATION_CONFIGS:
            step_train(cfg, resume=args.resume)
            step_evaluate(cfg)

    print()
    print('='*56)
    print('  Pipeline complete. Results in outputs/results/')
    print('='*56)


if __name__ == '__main__':
    main()
