#!/usr/bin/env python3
"""Compute SVD-based metrics (svd_entropy, stable_rank) on LoRA A/B matrices."""
import argparse
import json
import time
from pathlib import Path

import torch

from config_utils import resolve_checkpoint
from logging_utils import setup_logger
from model_utils import load_tokenizer_and_model
from svd_metrics import (
    SVD_GROUP_ORDER,
    collect_svd_metrics,
    expand_lora_groups,
    format_group_counts,
    format_group_values,
    parse_lora_group_name,
    summarize_grouping,
)
from utils import default_dtype_for_device, mean_or_none, parse_dtype


SCRIPT_DIR = Path(__file__).resolve().parent
ALL_METRICS = ["svd_entropy", "stable_rank"]


def parse_metrics(value: str) -> list[str]:
    """Parse a comma/space-separated metric list (or 'all') into ALL_METRICS order."""
    raw = [item.strip() for item in value.replace(" ", ",").split(",") if item.strip()]
    if not raw:
        return []
    if "all" in raw:
        return ALL_METRICS.copy()
    invalid = [item for item in raw if item not in ALL_METRICS]
    if invalid:
        raise ValueError(f"Unknown metrics: {', '.join(invalid)}")
    return [item for item in ALL_METRICS if item in raw]


def parse_svd_groups(value: str) -> list[str]:
    """Parse a group spec (or 'all') and expand bare base groups (e.g. 'AttnQO' -> '_A','_B')."""
    raw = [item.strip() for item in value.replace(" ", ",").split(",") if item.strip()]
    if not raw or "all" in raw:
        return expand_lora_groups(SVD_GROUP_ORDER)
    invalid = [
        item
        for item in raw
        if item not in SVD_GROUP_ORDER and parse_lora_group_name(item) is None
    ]
    if invalid:
        raise ValueError(f"Unknown SVD groups: {', '.join(invalid)}")
    ordered: list[str] = []
    for item in SVD_GROUP_ORDER:
        if item in raw:
            ordered.append(item)
    for item in raw:
        if parse_lora_group_name(item) is not None and item not in ordered:
            ordered.append(item)
    return expand_lora_groups(ordered)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argparser."""
    parser = argparse.ArgumentParser(description="SVD metrics for LoRA checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default=None, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--output", type=str, default=None, help="Optional output json path.")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path.")
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(ALL_METRICS),
        help="Comma-separated metrics to run (or 'all').",
    )
    parser.add_argument(
        "--svd-groups",
        type=str,
        default="all",
        help="Comma-separated LoRA SVD groups (e.g. 'AttnQO_A,AttnQO_B') or 'all'.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Load a LoRA checkpoint, compute selected SVD metrics, and write JSON + log."""
    run_timestamp = int(time.time())
    args = build_parser().parse_args(argv)

    log_path = Path(args.log_file) if args.log_file else SCRIPT_DIR / f"metrics_{run_timestamp}.log"
    log = setup_logger(log_path)

    if args.dtype is None:
        args.dtype = default_dtype_for_device(args.device)

    metrics_enabled = parse_metrics(args.metrics)
    if not metrics_enabled:
        raise ValueError("No metrics selected. Use --metrics or 'all'.")
    metrics_set = set(metrics_enabled)
    svd_groups = parse_svd_groups(args.svd_groups)
    svd_group_set = set(svd_groups)

    args.checkpoint = resolve_checkpoint(args.checkpoint)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    dtype = parse_dtype(args.dtype)
    log.info("Checkpoint: %s", args.checkpoint)
    log.info("Device: %s | dtype=%s", args.device, args.dtype)
    log.info("Log file: %s", log_path)
    log.info("Metrics: %s", ", ".join(metrics_enabled))
    log.info("SVD groups: %s", ", ".join(svd_groups))

    _tokenizer, model = load_tokenizer_and_model(args.checkpoint, dtype, device)

    log.info("Measuring %s...", ", ".join(metrics_enabled))
    svd_entropy, stable_rank_groups, svd_skipped, svd_group_counts = collect_svd_metrics(
        model,
        svd_group_set,
    )

    if "svd_entropy" in metrics_set:
        mean_entropy = mean_or_none(
            [value["mean"] for value in svd_entropy.values() if value["mean"] is not None]
        )
        log.info(
            "svd_entropy.groups=%d/%d | mean=%.6f | skipped=%d",
            len(svd_entropy),
            len(svd_groups),
            mean_entropy if mean_entropy is not None else 0.0,
            sum(svd_skipped.values()),
        )
        log.info("svd_entropy.group_means=%s", format_group_values(svd_entropy, order=svd_groups))
    if "stable_rank" in metrics_set:
        mean_rank = mean_or_none(
            [value["mean"] for value in stable_rank_groups.values() if value["mean"] is not None]
        )
        log.info(
            "stable_rank.groups=%d/%d | mean=%.6f | skipped=%d",
            len(stable_rank_groups),
            len(svd_groups),
            mean_rank if mean_rank is not None else 0.0,
            sum(svd_skipped.values()),
        )
        log.info("stable_rank.group_means=%s", format_group_values(stable_rank_groups, order=svd_groups))
    log.info("svd_group_counts.total=%s", format_group_counts(svd_group_counts["total"], order=svd_groups))
    log.info("svd_group_counts.selected=%s", format_group_counts(svd_group_counts["selected"], order=svd_groups))

    group_summary, ungrouped_summary = summarize_grouping(model.named_parameters(), svd_group_set)
    for group in svd_groups:
        summary = group_summary.get(group, [])
        log.info("svd_group_map.%s=%s", group, ", ".join(summary) if summary else "None")
    log.info(
        "svd_group_map.ungrouped=%s",
        ", ".join(ungrouped_summary) if ungrouped_summary else "None",
    )

    results = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "dtype": args.dtype,
        "metrics_requested": metrics_enabled,
    }
    if "svd_entropy" in metrics_set:
        results["svd_entropy_groups"] = svd_entropy
    if "stable_rank" in metrics_set:
        results["stable_rank_groups"] = stable_rank_groups
    results["svd_skipped"] = svd_skipped
    results["svd_group_counts"] = svd_group_counts
    results["svd_groups_requested"] = svd_groups
    results["svd_group_summary"] = group_summary
    results["svd_ungrouped_summary"] = ungrouped_summary

    payload = json.dumps(results, ensure_ascii=True, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n==== RESULTS_JSON ====\n")
        log_file.write(payload)
        log_file.write("\n")
    log.info("Results JSON written to %s", args.output if args.output else "stdout")


if __name__ == "__main__":
    main()
