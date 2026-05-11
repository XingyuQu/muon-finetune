#!/usr/bin/env python3
"""
Plot per-pretrain fine-tuning PPL trajectory (one figure per pretrain).

Loads the eval_log of the 8 best-LR runs from the meta_*.json files
saved by scripts.wikitext_finetune in
$NANOCHAT_BASE_DIR/wikitext_checkpoints/. For each of the two pretrains
(d20_muon and d20_adam_lr0.001) draws 4 curves
(Full / LoRA x Matched / Mismatched optimiser).

Usage:
    python scripts/analysis/plot_trajectory_per_pretrain.py
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

# Colors / line widths (matching the LR-sweep figure)
muon_dark, muon_light = '#2878b5', '#9ac9db'
adam_dark, adam_light = '#c82423', '#ff8884'
LW_FULL = 2.5
LW_LORA = 2.5

# 4 modes per pretrain. Each entry maps the legend label to the
# wikitext_checkpoints/<run_name>/ directory the meta_*.json lives in.
MUON_PRETRAIN_RUNS = {
    "Full-Muon": {"dir": "d20_muon_full-muon_lr0.9_ep1_seed0",       "color": muon_dark,  "lw": LW_FULL},
    "Full-Adam": {"dir": "d20_muon_full-adam_lr0.009_ep1_seed0",     "color": adam_dark,  "lw": LW_FULL},
    "LoRA-Muon": {"dir": "d20_muon_lora-muon_r8_lr0.9_ep1_seed0",    "color": muon_light, "lw": LW_LORA},
    "LoRA-Adam": {"dir": "d20_muon_lora-adam_r8_lr0.1_ep1_seed0",    "color": adam_light, "lw": LW_LORA},
}
ADAM_PRETRAIN_RUNS = {
    "Full-Adam": {"dir": "d20_adam_lr0.001_full-adam_lr0.03_ep1_seed0",   "color": adam_dark,  "lw": LW_FULL},
    "Full-Muon": {"dir": "d20_adam_lr0.001_full-muon_lr0.5_ep1_seed0",    "color": muon_dark,  "lw": LW_FULL},
    "LoRA-Adam": {"dir": "d20_adam_lr0.001_lora-adam_r8_lr0.3_ep1_seed0", "color": adam_light, "lw": LW_LORA},
    "LoRA-Muon": {"dir": "d20_adam_lr0.001_lora-muon_r8_lr0.7_ep1_seed0", "color": muon_light, "lw": LW_LORA},
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


def plot_trajectory(experiments, output_name, legend_title=""):
    # Load every run and use the largest-step value across them as the
    # x-axis denominator (training progress = step / total_steps * 100).
    runs = {}
    total_steps = 0
    for label, cfg in experiments.items():
        eval_log, baseline = load_run(cfg["dir"])
        runs[label] = (eval_log, baseline, cfg)
        total_steps = max(total_steps, max(e["step"] for e in eval_log))

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (eval_log, baseline, cfg) in runs.items():
        curve = compute_relative_curve(eval_log, baseline, total_steps)
        fracs = [f for f, _ in curve if f >= 0]
        ppls = [p for f, p in curve if f >= 0]
        # Drop the duplicate frac=0 entry produced by step=-1 + step=0
        if len(fracs) > 1 and fracs[0] == fracs[1]:
            fracs = fracs[1:]
            ppls = ppls[1:]

        ax.plot(fracs, ppls,
                label=label, color=cfg["color"],
                linestyle='-', marker='o',
                markersize=12, linewidth=cfg["lw"],
                markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel('Training Progress (%)', fontsize=45)
    ax.set_ylabel('Relative PPL', fontsize=45)
    ax.tick_params(axis='both', labelsize=30)
    leg = ax.legend(loc='upper right', fontsize=25, ncol=2, framealpha=0.9, edgecolor='none',
                    title=legend_title, title_fontsize=25)
    leg._legend_box.align = "center"
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 102)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{output_name}.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{output_name}.pdf', bbox_inches='tight')
    print(f"Saved: {output_name}.png and {output_name}.pdf")
    plt.close()


def main():
    plot_trajectory(MUON_PRETRAIN_RUNS, 'finetune_ppl_trajectory_d20_muon', "Muon-Pretrained")
    plot_trajectory(ADAM_PRETRAIN_RUNS, 'finetune_ppl_trajectory_d20_adam', "Adam-Pretrained")


if __name__ == "__main__":
    main()
