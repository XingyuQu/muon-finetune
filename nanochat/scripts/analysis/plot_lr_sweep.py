#!/usr/bin/env python3
"""Plot relative-PPL vs LR curves for the WikiText fine-tuning sweep.

Reads the aggregated CSV produced by collect_all_seeds.py
(default $REPO/wikitext_results.csv) and writes:
- lr_sweep_muon.{png,pdf}   (Muon-finetune panel of paper Fig. 4)
- lr_sweep_adam.{png,pdf}   (Adam-finetune panel of paper Fig. 4)
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[2]  # repo root
DEFAULT_CSV = OUTPUT_DIR / "wikitext_results.csv"

# Baseline (zero-shot WikiText init_ppl) for each pretrain
BASELINE_MUON_PRETRAIN = 20.090944851075893  # d20_muon
BASELINE_ADAM_PRETRAIN = 21.051515383635365  # d20_adam_lr0.001

# Style: Muon = blue, Adam = red. Solid = matched pretrain/finetune,
# dashed = mismatched. Full = dark color, LoRA = light color.
muon_dark, muon_light = '#2878b5', '#9ac9db'
adam_dark, adam_light = '#c82423', '#ff8884'
LW_FULL = 2.5
LW_LORA = 2.5


def load_sweep_data(csv_path: Path):
    """Read wikitext_results.csv and pivot into (model_tag, mode) -> {lr: mean_ppl}.

    Only keeps rank=8 rows (the rank used for paper Fig. 4 / Table 11).
    """
    series: dict[tuple[str, str], dict[float, float]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip blank lines and any non-data trailers some older CSVs
            # picked up from collect_all_seeds.py's stdout output.
            try:
                rank = int(row["rank"])
                lr = float(row["lr"])
                mean_ppl = float(row["mean_ppl"])
            except (TypeError, ValueError, KeyError):
                continue
            if rank != 8:
                continue
            key = (row["model_tag"], row["mode"])
            series.setdefault(key, {})[lr] = mean_ppl
    return series


def plot_data(ax, data, label, color, marker, linestyle='-', baseline=1.0, linewidth=2.5):
    """Plot sorted dict on given axes, normalised by baseline."""
    if not data:
        return
    lrs = sorted(data.keys())
    ppls = [data[lr] / baseline for lr in lrs]
    dash_style = (6, 3) if linestyle == '--' else ()
    ax.plot(lrs, ppls,
            label=label, color=color, marker=marker,
            linestyle=linestyle, dashes=dash_style,
            markersize=8, linewidth=linewidth,
            markeredgecolor='white', markeredgewidth=0.5)


def make_lr_sweep_panel(panel: str, sweeps, *, output_dir: Path):
    """Render one of the two LR-sweep panels in paper Fig. 4.

    panel == "muon" -> the curves where the *finetune* optimiser is Muon.
    panel == "adam" -> the curves where the *finetune* optimiser is Adam.
    Solid = matched pretrain (same as finetune), dashed = mismatched.
    """
    if panel == "muon":
        matched_full = sweeps.get(("d20_muon", "full-muon"), {})
        matched_lora = sweeps.get(("d20_muon", "lora-muon"), {})
        mismatched_full = sweeps.get(("d20_adam_lr0.001", "full-muon"), {})
        mismatched_lora = sweeps.get(("d20_adam_lr0.001", "lora-muon"), {})
        matched_baseline, mismatched_baseline = BASELINE_MUON_PRETRAIN, BASELINE_ADAM_PRETRAIN
        c_dark, c_light = muon_dark, muon_light
        out_name = "lr_sweep_muon"
    elif panel == "adam":
        matched_full = sweeps.get(("d20_adam_lr0.001", "full-adam"), {})
        matched_lora = sweeps.get(("d20_adam_lr0.001", "lora-adam"), {})
        mismatched_full = sweeps.get(("d20_muon", "full-adam"), {})
        mismatched_lora = sweeps.get(("d20_muon", "lora-adam"), {})
        matched_baseline, mismatched_baseline = BASELINE_ADAM_PRETRAIN, BASELINE_MUON_PRETRAIN
        c_dark, c_light = adam_dark, adam_light
        out_name = "lr_sweep_adam"
    else:
        raise ValueError(panel)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data(ax, matched_full, 'Full (Matched)', c_dark, 'o', '-', matched_baseline, LW_FULL)
    plot_data(ax, matched_lora, 'LoRA (Matched)', c_light, 'o', '-', matched_baseline, LW_LORA)
    plot_data(ax, mismatched_full, 'Full (Mismatched)', c_dark, 'o', '--', mismatched_baseline, LW_FULL)
    plot_data(ax, mismatched_lora, 'LoRA (Mismatched)', c_light, 'o', '--', mismatched_baseline, LW_LORA)

    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate', fontsize=30)
    ax.set_ylabel('Relative Perplexity', fontsize=30)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(loc='lower left', fontsize=15, ncol=2, framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.68, 1.05)

    if panel == "muon":
        # Inset zoom on the bottom-right region (close to the optimum)
        axins = ax.inset_axes([0.1, 0.28, 0.35, 0.35])
        plot_data(axins, matched_full, None, c_dark, 'o', '-', matched_baseline, LW_FULL)
        plot_data(axins, matched_lora, None, c_light, 'o', '-', matched_baseline, LW_LORA)
        plot_data(axins, mismatched_full, None, c_dark, 'o', '--', mismatched_baseline, LW_FULL)
        plot_data(axins, mismatched_lora, None, c_light, 'o', '--', mismatched_baseline, LW_LORA)
        axins.set_xscale('log')
        axins.set_xlim(0.2, 2.0)
        axins.set_ylim(0.70, 0.79)
        axins.tick_params(axis='both', which='both', left=False, bottom=False,
                          labelleft=False, labelbottom=False)
        axins.grid(True, alpha=0.3)
        ax.indicate_inset_zoom(axins, edgecolor='gray', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_dir / f"{out_name}.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f"{out_name}.pdf", bbox_inches='tight')
    print(f"Saved: {out_name}.png and {out_name}.pdf")
    plt.close(fig)


def main():
    if not DEFAULT_CSV.exists():
        raise FileNotFoundError(
            f"{DEFAULT_CSV} not found. Run "
            f"`python scripts/analysis/collect_all_seeds.py > wikitext_results.csv` "
            f"first to aggregate the per-seed meta files."
        )
    sweeps = load_sweep_data(DEFAULT_CSV)
    print(f"Loaded {len(sweeps)} (model, mode) sweep curves from {DEFAULT_CSV}")

    make_lr_sweep_panel("muon", sweeps, output_dir=OUTPUT_DIR)
    make_lr_sweep_panel("adam", sweeps, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
