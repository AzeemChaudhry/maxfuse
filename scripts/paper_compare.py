"""
One-click comparison runner for paper-style reporting.

Runs baseline and MAXFUSE end-to-end (train + evaluate), then generates:
- comparison_metrics.csv
- comparison_metrics.md
- comparison_bar_metrics.png
- run_summary.txt

Usage:
    python scripts/paper_compare.py --mode smoke
    python scripts/paper_compare.py --mode full
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def _header(title: str):
    print("\n" + "=" * 88)
    print(f"{title}")
    print("=" * 88)


def _run(cmd: list[str], step_name: str, env: dict, timeout: int):
    _header(f"[RUN] {step_name}")
    print(f"Command  : {' '.join(cmd)}")
    print(f"Timeout  : {timeout}s")
    print(f"Work Dir : {ROOT}")

    t0 = datetime.now()
    proc = subprocess.run(cmd, env=env, timeout=timeout)
    dt = (datetime.now() - t0).total_seconds()

    print(f"Duration : {dt:.1f}s")
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name} (exit code={proc.returncode})")


def _metrics_table(metrics_rows: list[dict], out_dir: Path):
    df = pd.DataFrame(metrics_rows)
    df = df[[
        'model_name', 'config', 'checkpoint',
        'accuracy_pct', 'macro_f1', 'macro_auc', 'ood_auroc', 'fpr95',
        'results_json', 'confusion_plot', 'energy_plot'
    ]]
    df = df.sort_values('accuracy_pct', ascending=False)

    csv_path = out_dir / 'comparison_metrics.csv'
    md_path = out_dir / 'comparison_metrics.md'
    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)

    print(f"[SAVE] Comparison CSV : {csv_path}")
    print(f"[SAVE] Comparison MD  : {md_path}")

    return df


def _plot_bar(df: pd.DataFrame, out_dir: Path, mode: str):
    plot_df = df.copy()
    x = np.arange(len(plot_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, plot_df['accuracy_pct'], width, label='Accuracy (%)')
    ax.bar(x, plot_df['macro_f1'] * 100.0, width, label='Macro F1 (%)')
    ax.bar(x + width, plot_df['macro_auc'] * 100.0, width, label='Macro AUC (%)')

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['model_name'], rotation=0)
    ax.set_ylabel('Score (%)')
    ax.set_xlabel('Model')
    ax.set_ylim(0, 100)
    ax.set_title(f'Model Comparison ({mode.upper()}): Closed-Set Metrics')
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    for i, v in enumerate(plot_df['accuracy_pct']):
        ax.text(i - width, v + 0.8, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    out_path = out_dir / 'comparison_bar_metrics.png'
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[SAVE] Comparison plot: {out_path}")


def _write_summary(df: pd.DataFrame, out_dir: Path, mode: str):
    best = df.iloc[0]
    lines = [
        f"Comparison mode: {mode}",
        f"Run timestamp  : {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Top model by closed-set accuracy:",
        f"  - Model      : {best['model_name']}",
        f"  - Accuracy   : {best['accuracy_pct']:.2f}%",
        f"  - Macro F1   : {best['macro_f1']:.4f}",
        f"  - Macro AUC  : {best['macro_auc']:.4f}",
        "",
        "Artifacts:",
        f"  - comparison_metrics.csv",
        f"  - comparison_metrics.md",
        f"  - comparison_bar_metrics.png",
    ]
    path = out_dir / 'run_summary.txt'
    path.write_text("\n".join(lines), encoding='utf-8')
    print(f"[SAVE] Summary text   : {path}")


def run_comparison(mode: str, resume: bool = False):
    mode = mode.lower()
    if mode not in {'smoke', 'full'}:
        raise ValueError("mode must be one of: smoke, full")

    experiments = [
        {
            'model_name': 'Baseline (CNN + RUSBoost Late Fusion)',
            'config': 'configs/baseline_smoke.yaml' if mode == 'smoke' else 'configs/baseline.yaml',
            'train_timeout': 4 * 3600 if mode == 'smoke' else 72 * 3600,
        },
        {
            'model_name': 'MAXFUSE (N1+N2+N3)',
            'config': 'configs/smoke_test.yaml' if mode == 'smoke' else 'configs/maxfuse_full.yaml',
            'train_timeout': 4 * 3600 if mode == 'smoke' else 72 * 3600,
        },
    ]

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = ROOT / 'outputs' / 'paper_compare' / f"{mode}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _header(f"MAXFUSE PAPER COMPARISON PIPELINE ({mode.upper()})")
    print("This run will train + evaluate baseline and MAXFUSE, then build paper-ready comparison artifacts.")
    print(f"All summary artifacts will be written to: {out_dir}")

    env = os.environ.copy()
    env.setdefault('WANDB_MODE', 'disabled')
    env.setdefault('WANDB_SILENT', 'true')

    rows = []
    for i, exp in enumerate(experiments, start=1):
        cfg = exp['config']
        stem = Path(cfg).stem
        suffix = f"paper_{mode}_{stem}"
        ckpt = ROOT / 'outputs' / 'checkpoints' / stem / 'best.pt'

        _header(f"EXPERIMENT {i}/{len(experiments)}: {exp['model_name']}")
        print(f"Config     : {cfg}")
        print(f"Checkpoint : {ckpt}")

        train_cmd = [sys.executable, 'src/train.py', '--config', cfg]
        if resume:
            train_cmd.append('--resume')
        _run(train_cmd, f"TRAIN [{stem}]", env, timeout=exp['train_timeout'])

        eval_cmd = [
            sys.executable, 'src/evaluate.py',
            '--config', cfg,
            '--checkpoint', str(ckpt),
            '--output_suffix', suffix,
        ]
        _run(eval_cmd, f"EVALUATE [{stem}]", env, timeout=3 * 3600)

        result_json = ROOT / 'outputs' / 'results' / f"{stem}_{suffix}.json"
        confusion_plot = ROOT / 'outputs' / 'results' / f"{stem}_{suffix}_confusion.png"
        energy_plot = ROOT / 'outputs' / 'results' / f"{stem}_{suffix}_energy_histogram.png"
        energy_plot_noise = ROOT / 'outputs' / 'results' / f"{stem}_{suffix}_energy_histogram_noise.png"
        if not energy_plot.exists() and energy_plot_noise.exists():
            energy_plot = energy_plot_noise

        if not result_json.exists():
            raise FileNotFoundError(f"Expected result JSON not found: {result_json}")

        with open(result_json, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        row = {
            'model_name': exp['model_name'],
            'config': cfg,
            'checkpoint': str(ckpt),
            'accuracy_pct': float(metrics['accuracy']) * 100.0,
            'macro_f1': float(metrics['macro_f1']),
            'macro_auc': float(metrics['macro_auc']),
            'ood_auroc': float(metrics.get('ood_auroc', np.nan)),
            'fpr95': float(metrics.get('fpr95', np.nan)),
            'results_json': str(result_json),
            'confusion_plot': str(confusion_plot),
            'energy_plot': str(energy_plot),
        }
        rows.append(row)

        print("[METRICS]")
        print(f"  Accuracy : {row['accuracy_pct']:.2f}%")
        print(f"  Macro F1 : {row['macro_f1']:.4f}")
        print(f"  Macro AUC: {row['macro_auc']:.4f}")
        if not np.isnan(row['ood_auroc']):
            print(f"  OOD AUROC: {row['ood_auroc']:.4f}")
            print(f"  FPR@95   : {row['fpr95']:.4f}")

    _header('BUILDING COMPARISON TABLES AND GRAPHS')
    df = _metrics_table(rows, out_dir)
    _plot_bar(df, out_dir, mode=mode)
    _write_summary(df, out_dir, mode=mode)

    for p in df['results_json'].tolist() + df['confusion_plot'].tolist() + df['energy_plot'].tolist():
        pp = Path(p)
        if pp.exists():
            shutil.copy2(pp, out_dir / pp.name)

    _header('PIPELINE COMPLETE')
    print(f"Comparison artifacts are ready in: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='One-click MAXFUSE vs Baseline comparison pipeline')
    parser.add_argument('--mode', default='smoke', choices=['smoke', 'full'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run_comparison(mode=args.mode, resume=args.resume)
