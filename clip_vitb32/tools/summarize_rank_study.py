#!/usr/bin/env python
"""Summarize LoRA rank best-lr study logs and plot avg test acc vs rank."""
from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

VARIANT_ORDER = [
    "lora_muon_pe",
    "lora_muon",
    "lora_adamw",
]


@dataclass(frozen=True)
class RunMeta:
    """Identifiers parsed from a rank-study log filename."""
    variant: str
    dataset: str
    lora_rank: int
    lr_str: str
    lr: Optional[float]
    seed: int


@dataclass
class RunResult:
    """A run's metadata plus its parsed test metrics."""
    meta: RunMeta
    log_path: Path
    test_acc_pct: Optional[float] = None
    test_loss: Optional[float] = None
    source: str = "log"


def _try_parse_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except Exception:
        return None


def _parse_meta_from_log_filename(path: Path) -> Optional[RunMeta]:
    name = path.name
    m = re.match(r"^(?P<prefix>.+)_r(?P<rank>\d+)_lr(?P<lr>[^_]+)_seed(?P<seed>\d+)\.log$", name)
    if not m:
        return None

    prefix = m.group("prefix")
    rank = int(m.group("rank"))
    lr_str = m.group("lr")
    seed = int(m.group("seed"))

    variant = None
    dataset = None
    for v in sorted(VARIANT_ORDER, key=len, reverse=True):
        if prefix.startswith(v + "_"):
            variant = v
            dataset = prefix[len(v) + 1:]
            break
    if variant is None or dataset is None:
        parts = prefix.split("_", 1)
        variant = parts[0]
        dataset = parts[1] if len(parts) > 1 else "unknown"

    lr = _try_parse_float(lr_str)
    return RunMeta(variant=variant, dataset=dataset, lora_rank=rank, lr_str=lr_str, lr=lr, seed=seed)


_RE_TEST_EVAL = re.compile(
    r"\[test-eval\]\s+test_loss=(?P<loss>[-+0-9.eE]+|nan|inf|None)\s+test_acc=(?P<acc>[0-9.]+)%",
    re.IGNORECASE,
)
_RE_SAVE_HISTORY = re.compile(r"^\[save\]\s+history\s+->\s+(?P<path>.+\.history\.json)\s*$")


def _extract_test_metrics_from_log(log_path: Path) -> tuple[Optional[float], Optional[float], str]:
    last_acc = None
    last_loss = None
    history_path = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _RE_TEST_EVAL.search(line)
            if m:
                acc = _try_parse_float(m.group("acc"))
                loss_raw = m.group("loss")
                loss = _try_parse_float(loss_raw) if loss_raw not in {"None"} else None
                if acc is not None:
                    last_acc = acc
                    last_loss = loss
            m2 = _RE_SAVE_HISTORY.match(line.strip())
            if m2:
                history_path = m2.group("path")

    if last_acc is not None:
        return last_acc, last_loss, "log[test-eval]"

    if history_path:
        hp = Path(history_path)
        if not hp.is_absolute():
            hp = (log_path.parent / hp).resolve()
        if hp.exists():
            try:
                import json
                data = json.loads(hp.read_text(encoding="utf-8"))
                acc = data.get("best_test_acc")
                loss = data.get("best_test_loss")
                acc_pct = float(acc) * 100.0 if acc is not None else None
                loss_val = float(loss) if loss is not None else None
                return acc_pct, loss_val, "history.json"
            except Exception:
                pass

    return None, None, "missing"


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _format_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    widths = {c: len(c) for c in columns}
    for r in rows:
        for c in columns:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    lines = [header, sep]
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)


def _iter_log_files(log_dir: Path) -> Iterable[Path]:
    if not log_dir.exists():
        return []
    return sorted(p for p in log_dir.glob("*.log") if p.is_file())


def load_results(log_dir: Path) -> list[RunResult]:
    """Scan ``log_dir`` for rank-study logs and return parsed RunResult records."""
    results: list[RunResult] = []
    for log_path in _iter_log_files(log_dir):
        meta = _parse_meta_from_log_filename(log_path)
        if meta is None:
            continue
        acc_pct, loss, source = _extract_test_metrics_from_log(log_path)
        results.append(RunResult(meta=meta, log_path=log_path, test_acc_pct=acc_pct, test_loss=loss, source=source))
    return results


def aggregate(results: list[RunResult]) -> list[dict]:
    """Average runs per (dataset, variant, lr, rank) and return sorted summary rows."""
    by_key: dict[tuple[str, str, str, int], list[RunResult]] = defaultdict(list)
    for r in results:
        if r.test_acc_pct is None:
            continue
        k = (r.meta.dataset, r.meta.variant, r.meta.lr_str, r.meta.lora_rank)
        by_key[k].append(r)

    rows = []
    for (dataset, variant, lr_str, rank), runs in by_key.items():
        accs = [float(x.test_acc_pct) for x in runs if x.test_acc_pct is not None]
        mean_acc, std_acc = _mean_std(accs)
        seeds = ",".join(str(x.meta.seed) for x in sorted(runs, key=lambda z: z.meta.seed))
        rows.append({
            "dataset": dataset,
            "variant": variant,
            "lr": lr_str,
            "rank": int(rank),
            "n_seeds": len(accs),
            "avg_test_acc_pct": float(mean_acc),
            "avg_test_acc(%)": f"{mean_acc:.2f}",
            "std": f"{std_acc:.2f}",
            "seeds": seeds,
        })

    def _sort_key(row: dict):
        v = row["variant"]
        v_idx = VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 999
        lr_num = _try_parse_float(row["lr"])
        lr_num = lr_num if lr_num is not None else math.inf
        return (row["dataset"], v_idx, v, int(row["rank"]), lr_num, str(row["lr"]))

    rows.sort(key=_sort_key)
    return rows


def plot(agg_rows: list[dict], out_path: Path, dataset: Optional[str], title: Optional[str]) -> None:
    """Plot avg test acc vs LoRA rank with one curve per variant (lr annotated per point)."""
    import matplotlib.pyplot as plt

    filtered = [r for r in agg_rows if (dataset is None or r["dataset"] == dataset)]
    datasets = sorted({r["dataset"] for r in filtered})
    if not datasets:
        raise RuntimeError("no aggregated rows to plot")

    n = len(datasets)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5 * ncols, 4.8 * nrows), squeeze=False)

    variant_to_color = {v: f"C{i % 10}" for i, v in enumerate(VARIANT_ORDER)}

    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        ds_rows = [r for r in filtered if r["dataset"] == ds]
        series: dict[str, list[dict]] = defaultdict(list)
        for r in ds_rows:
            series[r["variant"]].append(r)

        for variant, points in sorted(
            series.items(),
            key=lambda kv: (VARIANT_ORDER.index(kv[0]) if kv[0] in VARIANT_ORDER else 999, kv[0]),
        ):
            points.sort(key=lambda r: int(r["rank"]))
            xs = [int(p["rank"]) for p in points]
            ys = [float(p["avg_test_acc_pct"]) for p in points]
            lrs = [str(p["lr"]) for p in points]
            lr_const = len(set(lrs)) == 1
            label = f"{variant} (lr={lrs[0]})" if lr_const else f"{variant} (lr varies)"

            color = variant_to_color.get(variant, None)
            ax.plot(xs, ys, marker="o", linewidth=2, linestyle="-", color=color, label=label)

            for p in points:
                ax.annotate(
                    str(p["lr"]),
                    xy=(int(p["rank"]), float(p["avg_test_acc_pct"])),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=color,
                )

        ax.set_title(ds)
        ax.set_xlabel("LoRA rank (r)")
        ax.set_ylabel("Average test accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)


def main() -> int:
    """CLI entry point: aggregate rank-study logs, print tables, and save a plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("experiments/logs/vitb32_lora_rank_bestlr"))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--out-fig", type=Path, default=Path("experiments/logs/vitb32_lora_rank_bestlr/summary_rank_vs_avg_acc.png"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--title", type=str, default="ViT-B/32 LoRA rank best-lr: avg test acc vs rank")
    parser.add_argument("--per-run", action="store_true")
    args = parser.parse_args()

    results = load_results(args.log_dir)
    if not results:
        print(f"[error] No parsable log files found under: {args.log_dir}")
        return 2

    if args.per_run:
        per_run_rows = []
        for r in sorted(
            results,
            key=lambda x: (
                x.meta.dataset,
                VARIANT_ORDER.index(x.meta.variant) if x.meta.variant in VARIANT_ORDER else 999,
                x.meta.variant,
                x.meta.lora_rank,
                x.meta.lr if x.meta.lr is not None else math.inf,
                x.meta.seed,
            ),
        ):
            acc = "" if r.test_acc_pct is None else f"{r.test_acc_pct:.2f}"
            per_run_rows.append({
                "dataset": r.meta.dataset,
                "variant": r.meta.variant,
                "lr": r.meta.lr_str,
                "rank": r.meta.lora_rank,
                "seed": r.meta.seed,
                "test_acc(%)": acc,
                "source": r.source,
                "log": r.log_path.name,
            })
        print("\n[per-run]")
        print(_format_table(per_run_rows, ["dataset", "variant", "lr", "rank", "seed", "test_acc(%)", "source", "log"]))

    agg_rows = aggregate(results)
    if not agg_rows:
        print("[error] No runs with parsable test accuracy found in logs.")
        return 3

    filtered = [r for r in agg_rows if (args.dataset is None or r["dataset"] == args.dataset)]
    if not filtered:
        print(f"[error] No aggregated rows match dataset={args.dataset!r}.")
        return 4

    print("\n[aggregated-by-seed-mean]")
    print(_format_table(filtered, ["dataset", "variant", "lr", "rank", "n_seeds", "avg_test_acc(%)", "std", "seeds"]))

    if not args.no_plot:
        plot(agg_rows, args.out_fig, dataset=args.dataset, title=args.title)
        print(f"\n[plot] saved -> {args.out_fig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
