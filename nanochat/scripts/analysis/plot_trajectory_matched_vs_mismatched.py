#!/usr/bin/env python3
"""
Plot the matched-vs-mismatched fine-tuning PPL trajectory (Full only).

Loads the eval_log of the 4 best-LR Full runs (Full-Muon / Full-Adam on
each of the two pretrains) from the meta_*.json files saved by
scripts.wikitext_finetune in $NANOCHAT_BASE_DIR/wikitext_checkpoints/.
Averages matched (Muon-Full-Muon, Adam-Full-Adam) and mismatched
(Muon-Full-Adam, Adam-Full-Muon) curves across the two pretrain
scenarios and plots two relative-PPL curves over training progress.

Usage:
    python scripts/analysis/plot_trajectory_matched_vs_mismatched.py
"""

import json
import sys
from pathlib import Path

# Make `nanochat` importable even when the package is not installed as a
# wheel (e.g. when pip install -e . skipped the Python package because of
# the maturin module-name vs project-name mismatch).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})
import matplotlib.pyplot as plt

from nanochat.common import get_base_dir

OUTPUT_DIR = REPO_ROOT  # save figures to repo root

# 4 best-LR Full-finetune runs grouped by category, keyed on the
# wikitext_checkpoints/<run_name>/ directory name. The pretrain tag is
# only kept here for orientation; baselines are read from the meta
# itself (init_ppl) so they stay in sync with whatever the run produced.
EXPERIMENTS = {
    "Matched": [
        "d20_muon_full-muon_lr0.9_ep1_seed0",
        "d20_adam_lr0.001_full-adam_lr0.03_ep1_seed0",
    ],
    "Mismatched": [
        "d20_muon_full-adam_lr0.009_ep1_seed0",
        "d20_adam_lr0.001_full-muon_lr0.5_ep1_seed0",
    ],
}

COLORS = {
    "Matched": '#c82423',
    "Mismatched": '#2878b5',
}


def load_run(run_name: str):
    """Return (eval_log, init_ppl) from the latest meta_*.json of this run."""
    run_dir = Path(get_base_dir()) / "wikitext_checkpoints" / run_name
    metas = sorted(run_dir.glob("meta_*.json"))
    if not metas:
        raise FileNotFoundError(f"No meta_*.json under {run_dir}")
    with open(metas[-1]) as f:
        meta = json.load(f)
    return meta["eval_log"], meta["init_ppl"]


def compute_relative_curve(eval_log, baseline, total_steps):
    """Convert eval_log to a list of (training_progress_%, relative_ppl)."""
    curve = []
    for entry in eval_log:
        frac = 0.0 if entry["step"] <= 0 else entry["step"] / total_steps * 100
        curve.append((frac, entry["perplexity"] / baseline))
    return curve


def main():
    # Load every run once and figure out the total number of steps from
    # the highest step recorded across all eval_logs.
    runs = {}
    total_steps = 0
    for cat, names in EXPERIMENTS.items():
        for name in names:
            eval_log, baseline = load_run(name)
            runs[name] = (eval_log, baseline)
            total_steps = max(total_steps, max(e["step"] for e in eval_log))

    # Build per-category curves and average across pretrains
    category_curves = {}
    for cat_name, names in EXPERIMENTS.items():
        rel_curves = [compute_relative_curve(*runs[n], total_steps) for n in names]
        n_points = min(len(c) for c in rel_curves)
        avg = []
        for i in range(n_points):
            frac = rel_curves[0][i][0]
            mean_rel = float(np.mean([c[i][1] for c in rel_curves]))
            avg.append((frac, mean_rel))
        category_curves[cat_name] = avg

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat_name, curve in category_curves.items():
        fracs = [f for f, _ in curve if f >= 0]
        ppls = [p for f, p in curve if f >= 0]
        # Drop the duplicate frac=0 entry produced by step=-1 + step=0
        if len(fracs) > 1 and fracs[0] == fracs[1]:
            fracs = fracs[1:]
            ppls = ppls[1:]
        ax.plot(fracs, ppls,
                label=cat_name, color=COLORS[cat_name],
                linestyle='-', marker='o',
                markersize=12, linewidth=2.5,
                markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel('Training Progress (%)', fontsize=33)
    ax.set_ylabel('Relative Perplexity', fontsize=33)
    ax.tick_params(axis='both', labelsize=25)
    ax.legend(fontsize=28, loc='upper right', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 102)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'finetune_ppl_trajectory.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'finetune_ppl_trajectory.pdf', bbox_inches='tight')
    print("Saved: finetune_ppl_trajectory.png and finetune_ppl_trajectory.pdf")


if __name__ == "__main__":
    main()
