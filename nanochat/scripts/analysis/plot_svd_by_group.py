#!/usr/bin/env python3
"""
Plot SVD entropy and stable rank by layer group.

Reads the per-step metrics produced by scripts/analysis/run_svd_metrics.py
(under svd_results/summary.json) and renders one 2x3 subplot grid per
metric (one subplot per layer group).

Generates:
- svd_entropy_by_group.{png,pdf}
- stable_rank_by_group.{png,pdf}

Usage:
    python scripts/analysis/plot_svd_by_group.py
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

GROUPS = ["AttnQ", "AttnK", "AttnV", "AttnProj", "MLP_FC", "MLP_Proj"]

OPTIMIZERS = {
    "d20_adam_lr0.001": {"label": "Adam", "color": "#c82423", "marker": "o"},
    "d20_muon": {"label": "Muon", "color": "#2878b5", "marker": "o"},
}


def load_data():
    """Load summary.json produced by run_svd_metrics.py."""
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path) as f:
        return json.load(f)


def extract_per_group(data, metric_key: str):
    """
    Extract per-group time series for each optimizer from summary.json.

    metric_key is either "svd_entropy" or "stable_rank" (the keys
    written by run_svd_metrics.py for each step).

    Returns: dict[optimizer][group] = {"steps": [...], "values": [...]}
    """
    out = {}
    for opt_name in OPTIMIZERS:
        if opt_name not in data:
            continue
        out[opt_name] = {}
        opt_data = data[opt_name]
        steps = sorted(int(s) for s in opt_data.keys())
        for group in GROUPS:
            xs, ys = [], []
            for step in steps:
                step_data = opt_data.get(str(step), {})
                metric_data = step_data.get(metric_key, {})
                value = metric_data.get(group, {}).get("mean")
                if value is not None:
                    xs.append(step)
                    ys.append(value)
            out[opt_name][group] = {"steps": xs, "values": ys}
    return out


def save_figure(output_path: Path, dpi: int = 150):
    """Save figure in both PNG and PDF formats."""
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")


def plot_by_group(per_group, ylabel: str, filename: str):
    """Plot per-group time series in a 2x3 subplot grid.

    Skips the very first checkpoint per series (typically step 10) because
    its values reflect initialization noise rather than steady-state
    training behaviour. Y-axis range is set per subplot with 10% padding.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, group in enumerate(GROUPS):
        ax = axes[idx]
        all_values = []
        for opt_name, opt_config in OPTIMIZERS.items():
            if opt_name not in per_group:
                continue
            group_data = per_group[opt_name].get(group, {})
            steps = group_data.get("steps", [])
            values = group_data.get("values", [])
            if len(steps) < 2 or len(values) < 2:
                continue
            # Drop the first point (step 10 init-noise spike)
            steps_plot = steps[1:]
            values_plot = values[1:]
            all_values.extend(values_plot)
            steps_k = [s / 1000 for s in steps_plot]
            ax.plot(
                steps_k, values_plot,
                label=opt_config["label"],
                color=opt_config["color"],
                marker=opt_config["marker"],
                markersize=8,
                linewidth=2.5,
            )

        if all_values:
            ymin, ymax = min(all_values), max(all_values)
            padding = (ymax - ymin) * 0.1
            ax.set_ylim(ymin - padding, ymax + padding)

        ax.set_title(group, fontsize=35, fontweight="bold")
        ax.set_xlabel("Step (k)", fontsize=30)
        ax.set_ylabel(ylabel, fontsize=30)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=25)

    axes[0].legend(fontsize=25, loc="best")
    plt.tight_layout()
    save_figure(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    print("Plotting SVD Entropy by group...")
    plot_by_group(
        extract_per_group(data, "svd_entropy"),
        ylabel="Entropy",
        filename="svd_entropy_by_group.png",
    )

    print("Plotting Stable Rank by group...")
    plot_by_group(
        extract_per_group(data, "stable_rank"),
        ylabel="Stable Rank",
        filename="stable_rank_by_group.png",
    )

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
