#!/usr/bin/env python
"""Summarize per-(variant, dataset, lr) training-log results from the standard sweep."""
import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DEFAULT_DATASETS = [
    "stanford_cars",
    "dtd",
    "gtsrb",
    "resisc45",
    "sun397",
    "svhn",
]
DEFAULT_VARIANTS = [
    "full_adamw",
    "full_muon",
    "full_muon_pe",
    "lora_adamw",
    "lora_muon",
    "lora_muon_pe",
]


@dataclass
class RunRecord:
    """Parsed metrics + identifiers for a single completed run log."""
    variant: str
    dataset: str
    lr: str
    seed: str
    path: str
    total_steps: int
    max_step: int
    last_train_loss: float
    pre_test_acc: float
    test_loss: Optional[float]
    test_acc: float


def _parse_id_from_filename(
    path: str, datasets: List[str], variants: List[str]
) -> Optional[Tuple[str, str, str, str]]:
    name = os.path.basename(path)
    if not name.endswith(".log"):
        return None
    stem = name[:-4]
    variants_set = set(variants)
    for dataset in sorted(datasets, key=len, reverse=True):
        marker = f"_{dataset}_lr"
        if marker not in stem:
            continue
        variant, rest = stem.split(marker, 1)
        if not variant or variant not in variants_set:
            return None
        if "_seed" not in rest:
            return None
        lr, seed = rest.split("_seed", 1)
        return variant, dataset, lr, seed
    return None


RE_TRAIN_STEP = re.compile(r"\[train\] step (\d+)/(\d+) loss=([0-9.+-eE]+)")
RE_PRE_EVAL = re.compile(r"\[pre-eval\].*?test_loss=([0-9.+-eENon]+)\s+test_acc=([0-9.+-eE]+)%")
RE_TEST = re.compile(r"\[test-eval\] test_loss=([0-9.+-eENon]+) test_acc=([0-9.+-eE]+)%")


def _parse_log(path: str, datasets: List[str], variants: List[str]) -> Optional[RunRecord]:
    parsed = _parse_id_from_filename(path, datasets, variants)
    if not parsed:
        return None
    variant, dataset, lr, seed = parsed

    total_steps = None
    max_step = 0
    last_train_loss = None
    pre_test_acc = None
    test_loss = None
    test_acc = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RE_PRE_EVAL.search(line)
            if m:
                pre_test_acc = float(m.group(2)) / 100.0
                continue

            m = RE_TRAIN_STEP.search(line)
            if m:
                step = int(m.group(1))
                total = int(m.group(2))
                total_steps = total
                max_step = max(max_step, step)
                last_train_loss = float(m.group(3))
                continue

            m = RE_TEST.search(line)
            if m:
                loss_str = m.group(1)
                test_loss = None if loss_str == "None" else float(loss_str)
                test_acc = float(m.group(2)) / 100.0

    if any(v is None for v in (last_train_loss, test_acc)):
        return None
    if total_steps is None or max_step < total_steps:
        return None

    return RunRecord(
        variant=variant,
        dataset=dataset,
        lr=lr,
        seed=seed,
        path=path,
        total_steps=total_steps,
        max_step=max_step,
        last_train_loss=last_train_loss,
        pre_test_acc=pre_test_acc if pre_test_acc is not None else 0.0,
        test_loss=test_loss,
        test_acc=test_acc,
    )


@dataclass
class LRGroup:
    """All seeds for one (variant, dataset, lr) combination plus their averages."""
    lr: str
    records: List[RunRecord]
    avg_test_acc: float
    avg_test_loss: Optional[float]

    @staticmethod
    def from_records(lr: str, records: List[RunRecord]) -> "LRGroup":
        """Aggregate ``records`` (same lr) into an LRGroup with mean test acc / loss."""
        avg_acc = sum(r.test_acc for r in records) / len(records)
        losses = [r.test_loss for r in records if r.test_loss is not None]
        avg_loss = sum(losses) / len(losses) if losses else None
        return LRGroup(
            lr=lr,
            records=sorted(records, key=lambda r: r.seed),
            avg_test_acc=avg_acc,
            avg_test_loss=avg_loss,
        )


def group_by_variant_dataset_lr(
    records: List[RunRecord],
) -> Dict[str, Dict[str, List[LRGroup]]]:
    """Group records into ``{variant: {dataset: [LRGroup, ...]}}`` sorted best-LR-first."""
    temp: Dict[str, Dict[str, Dict[str, List[RunRecord]]]] = {}
    for rec in records:
        by_ds = temp.setdefault(rec.variant, {})
        by_lr = by_ds.setdefault(rec.dataset, {})
        by_lr.setdefault(rec.lr, []).append(rec)

    result: Dict[str, Dict[str, List[LRGroup]]] = {}
    for variant, ds_dict in temp.items():
        result[variant] = {}
        for ds, lr_dict in ds_dict.items():
            lr_groups = [LRGroup.from_records(lr, recs) for lr, recs in lr_dict.items()]
            lr_groups.sort(key=lambda g: g.avg_test_acc, reverse=True)
            result[variant][ds] = lr_groups
    return result


def fmt_acc(acc: Optional[float]) -> str:
    """Format a fractional accuracy as a 2-decimal percentage (or ``NA``)."""
    return "NA" if acc is None else f"{acc * 100:.2f}"


def print_summary(grouped, datasets, variants):
    """Print a per-variant breakdown of all LRs / seeds with the best LR starred."""
    for variant in variants:
        print("=" * 95)
        print(f"{variant}")
        print("-" * 95)
        rows = []
        best_accs = []
        for ds in datasets:
            lr_groups = grouped.get(variant, {}).get(ds, [])
            if not lr_groups:
                rows.append((ds, "-", "NA", "NA", "NA", "NA", "NA", ""))
                continue
            for i, grp in enumerate(lr_groups):
                is_best = i == 0
                for rec in grp.records:
                    rows.append((
                        ds,
                        f"seed{rec.seed}",
                        grp.lr,
                        f"{rec.test_loss:.4f}" if rec.test_loss is not None else "None",
                        f"{rec.test_acc * 100:.2f}",
                        f"{rec.last_train_loss:.4f}",
                        f"{rec.pre_test_acc * 100:.2f}",
                        "*" if is_best else "",
                    ))
                avg_loss_str = f"{grp.avg_test_loss:.4f}" if grp.avg_test_loss is not None else "None"
                marker = "* BEST" if is_best else ""
                rows.append((ds, "avg", grp.lr, avg_loss_str, f"{grp.avg_test_acc * 100:.2f}", "-", "-", marker))
                if is_best:
                    best_accs.append(grp.avg_test_acc)
            rows.append(("", "", "", "", "", "", "", ""))
        header = ("dataset", "seed", "lr", "test_loss", "test_acc", "last_train", "pre_test_acc", "")
        widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
        for r in rows:
            print("  " + "  ".join(r[i].ljust(widths[i]) for i in range(len(r))))
        if best_accs:
            avg = sum(best_accs) / len(best_accs)
            print(f"  avg best-lr test acc ({len(datasets)} datasets): {fmt_acc(avg)}")
        else:
            print(f"  avg best-lr test acc ({len(datasets)} datasets): NA")
        print()


def print_acc_table(grouped, datasets, variants):
    """Print a compact best-LR average test-acc table (variants x datasets, with row averages)."""
    header = ["algorithm"] + datasets + ["avg"]
    rows = []
    for variant in variants:
        vals = []
        all_accs = []
        for ds in datasets:
            lr_groups = grouped.get(variant, {}).get(ds, [])
            if not lr_groups:
                vals.append("NA")
            else:
                best_grp = lr_groups[0]
                vals.append(fmt_acc(best_grp.avg_test_acc))
                all_accs.append(best_grp.avg_test_acc)
        avg = fmt_acc(sum(all_accs) / len(all_accs)) if all_accs else "NA"
        rows.append([variant] + vals + [avg])

    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    print("=" * 95)
    print("Best-LR Average Test Acc Table (percent, best LR selected by avg acc across seeds)")
    print("-" * 95)
    print("  " + "  ".join(header[i].ljust(widths[i]) for i in range(len(header))))
    for row in rows:
        print("  " + "  ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    print()


def write_csv(grouped, datasets, variants, path):
    """Dump grouped per-run / per-LR results to a CSV at ``path``."""
    header = [
        "variant", "dataset", "seed", "lr", "is_best_lr",
        "test_loss", "test_acc", "avg_test_acc", "last_train_loss", "pre_test_acc",
    ]
    rows = []
    for variant in variants:
        for ds in datasets:
            lr_groups = grouped.get(variant, {}).get(ds, [])
            if not lr_groups:
                rows.append([variant, ds, "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"])
                continue
            for i, grp in enumerate(lr_groups):
                is_best = "Y" if i == 0 else "N"
                for rec in grp.records:
                    rows.append([
                        variant, ds, rec.seed, grp.lr, is_best,
                        f"{rec.test_loss:.4f}" if rec.test_loss is not None else "None",
                        f"{rec.test_acc * 100:.2f}",
                        f"{grp.avg_test_acc * 100:.2f}",
                        f"{rec.last_train_loss:.4f}",
                        f"{rec.pre_test_acc * 100:.2f}",
                    ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    """CLI entry point: scan ``--logs-dir`` for completed runs and print summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=os.path.join("logs", "std_vitb32_bestlr"))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    log_dir = args.logs_dir
    if not os.path.isdir(log_dir):
        raise SystemExit(f"logs dir not found: {log_dir}")

    records = []
    total = 0
    for name in os.listdir(log_dir):
        if not name.endswith(".log"):
            continue
        total += 1
        rec = _parse_log(os.path.join(log_dir, name), datasets, variants)
        if rec:
            records.append(rec)

    print(f"[scan] logs_dir={log_dir} logs={total} complete_runs={len(records)}")
    if not records:
        return

    grouped = group_by_variant_dataset_lr(records)
    print_summary(grouped, datasets, variants)
    print_acc_table(grouped, datasets, variants)
    csv_path = args.csv or os.path.join(log_dir, "summary.csv")
    write_csv(grouped, datasets, variants, csv_path)
    print(f"[csv] wrote {csv_path}")


if __name__ == "__main__":
    main()
