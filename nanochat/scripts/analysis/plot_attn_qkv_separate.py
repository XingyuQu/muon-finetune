#!/usr/bin/env python3
"""
Plot AttnQKV (average of AttnQ, AttnK, AttnV) SVD entropy and stable
rank as separate single-panel figures.

Reads the per-step metrics produced by scripts/analysis/run_svd_metrics.py
(under svd_results/summary.json) and averages the entropy and stable
rank of the three attention projections at each step.

Generates:
- attn_qkv_entropy.{png,pdf}
- attn_qkv_stable_rank.{png,pdf}

Usage:
    python scripts/analysis/plot_attn_qkv_separate.py
"""

import json
from pathlib import Path

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

SCRIPT_DIR = Path(__file__).resolve().parent
_ANALYSIS_DIR = SCRIPT_DIR.parents[1] / "analysis_results"
RESULTS_DIR = _ANALYSIS_DIR / "svd_results"
OUTPUT_DIR = _ANALYSIS_DIR / "svd_plots"

QKV_GROUPS = ["AttnQ", "AttnK", "AttnV"]

OPTIMIZERS = {
    "d20_adam_lr0.001": {"label": "Adam", "color": "#c82423", "marker": "o"},
    "d20_muon": {"label": "Muon", "color": "#2878b5", "marker": "o"},
}


def load_data():
    """Load summary.json produced by run_svd_metrics.py."""
    with open(RESULTS_DIR / "summary.json") as f:
        return json.load(f)


def attn_qkv_series(data, metric_key: str):
    """
    Return per-optimizer (steps, values) where each value is the mean
    over AttnQ, AttnK, AttnV of the per-group "mean" field.

    metric_key is "svd_entropy" or "stable_rank".
    """
    series = {}
    for opt_name in OPTIMIZERS:
        if opt_name not in data:
            continue
        opt_data = data[opt_name]
        steps = sorted(int(s) for s in opt_data.keys())
        xs, ys = [], []
        for step in steps:
            step_data = opt_data.get(str(step), {})
            metric_data = step_data.get(metric_key, {})
            qkv = [metric_data.get(g, {}).get("mean") for g in QKV_GROUPS]
            if any(v is None for v in qkv):
                continue
            xs.append(step)
            ys.append(sum(qkv) / len(qkv))
        series[opt_name] = (xs, ys)
    return series


def save_figure(output_path: Path, dpi: int = 150):
    """Save figure in both PNG and PDF formats."""
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")


def plot_single(series, ylabel: str, output_name: str, annotation_offset: float):
    """Render one AttnQKV figure (entropy or stable rank)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for opt_name, (steps, values) in series.items():
        if not steps:
            continue
        opt_config = OPTIMIZERS[opt_name]
        steps_k = [s / 1000 for s in steps]
        ax.plot(
            steps_k, values,
            label=opt_config["label"],
            color=opt_config["color"],
            marker=opt_config["marker"],
            markersize=10,
            linewidth=2.5,
            markeredgecolor='white',
            markeredgewidth=0.5,
        )

    ax.set_xlabel("Step (k)", fontsize=45)
    ax.set_ylabel(ylabel, fontsize=45)
    ax.tick_params(axis='both', labelsize=30)
    ax.legend(fontsize=25, framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)

    # Highlight the very first checkpoint (step 10) -- it sits before
    # the steady-state regime and is annotated rather than dropped.
    ax.axvline(x=0.01, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ymax = ax.get_ylim()[1]
    ax.annotate('Step 10',
                xy=(0.01, ymax),
                xytext=(1.0, ymax - annotation_offset),
                fontsize=25, color='black', ha='left', va='top')

    plt.tight_layout()
    save_figure(OUTPUT_DIR / f"{output_name}.png", dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    print("Plotting AttnQKV entropy...")
    plot_single(
        attn_qkv_series(data, "svd_entropy"),
        ylabel="Entropy",
        output_name="attn_qkv_entropy",
        annotation_offset=0.005,
    )

    print("Plotting AttnQKV stable rank...")
    plot_single(
        attn_qkv_series(data, "stable_rank"),
        ylabel="Stable Rank",
        output_name="attn_qkv_stable_rank",
        annotation_offset=10,
    )

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
