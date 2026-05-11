#!/usr/bin/env python3
"""
Run SVD metrics on multiple checkpoints and all steps, with multi-GPU support.

Usage:
    # Run all tasks across 8 GPUs
    python scripts/analysis/run_svd_metrics.py

    # Run specific GPU (for manual parallel execution)
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_svd_metrics.py --gpu_id 0 --num_gpus 8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add nanochat to path (parents[1] = repo root, where the nanochat/ package lives)
SCRIPT_DIR = Path(__file__).resolve().parent
NANOCHAT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(NANOCHAT_ROOT))

import torch
import torch.multiprocessing as mp

from nanochat.common import get_base_dir
from svd_metrics import load_model_for_metrics, collect_svd_metrics, GROUP_ORDER, mean_or_none

# Checkpoints to analyze (the two pretrains used in the paper).
BASE_CKPT_DIR = Path(get_base_dir()) / "base_checkpoints"
CHECKPOINTS = [
    str(BASE_CKPT_DIR / "d20_adam_lr0.001"),
    str(BASE_CKPT_DIR / "d20_muon"),
]

# Output directory
OUTPUT_DIR = SCRIPT_DIR.parents[1] / "analysis_results" / "svd_results"


def get_all_steps(checkpoint_dir: Path) -> List[int]:
    """Get all available steps for a checkpoint."""
    model_files = list(checkpoint_dir.glob("model_*.pt"))
    steps = []
    for f in model_files:
        step_str = f.stem.split("_")[-1]
        steps.append(int(step_str))
    return sorted(steps)


def get_all_tasks() -> List[Tuple[str, int]]:
    """Get all (checkpoint_dir, step) pairs."""
    tasks = []
    for ckpt_path in CHECKPOINTS:
        ckpt_dir = Path(ckpt_path)
        if not ckpt_dir.exists():
            print(f"WARNING: {ckpt_dir} not found, skipping.")
            continue
        steps = get_all_steps(ckpt_dir)
        for step in steps:
            tasks.append((str(ckpt_dir), step))
    return tasks


def process_single_task(checkpoint_dir: str, step: int, gpu_id: int) -> dict:
    """Process a single checkpoint/step on a specific GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    ckpt_name = Path(checkpoint_dir).name

    print(f"[GPU {gpu_id}] Processing {ckpt_name} step {step}...")

    try:
        # Load model
        model, meta_data = load_model_for_metrics(checkpoint_dir, step, device)

        # Compute metrics
        svd_entropy, stable_rank_groups, skipped, _ = collect_svd_metrics(
            model,
            save_singular_values=False,
        )

        # Compute overall metrics
        overall_entropy = mean_or_none([
            v["mean"] for v in svd_entropy.values() if v["mean"] is not None
        ])
        overall_rank = mean_or_none([
            v["mean"] for v in stable_rank_groups.values() if v["mean"] is not None
        ])

        result = {
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_name": ckpt_name,
            "step": step,
            "svd_entropy": svd_entropy,
            "stable_rank": stable_rank_groups,
            "overall_entropy": overall_entropy,
            "overall_stable_rank": overall_rank,
            "skipped": skipped,
            "model_config": meta_data.get("model_config", {}),
            "user_config": meta_data.get("user_config", {}),
        }

        # Clean up
        del model
        torch.cuda.empty_cache()

        print(f"[GPU {gpu_id}] Done {ckpt_name} step {step}: entropy={overall_entropy:.4f}, rank={overall_rank:.2f}")
        return result

    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR processing {ckpt_name} step {step}: {e}")
        return {
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_name": ckpt_name,
            "step": step,
            "error": str(e),
        }


def worker_fn(gpu_id: int, tasks: List[Tuple[str, int]], output_dir: Path):
    """Process every assigned (checkpoint, step) on this GPU and write one JSON per step under output_dir."""
    torch.cuda.set_device(gpu_id)

    for checkpoint_dir, step in tasks:
        result = process_single_task(checkpoint_dir, step, gpu_id)

        # Save individual result
        ckpt_name = Path(checkpoint_dir).name
        output_file = output_dir / f"{ckpt_name}_step{step:06d}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)


def run_parallel(num_gpus: int = 8):
    """Run all tasks in parallel across multiple GPUs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = get_all_tasks()
    print(f"Total tasks: {len(tasks)}")
    print(f"Using {num_gpus} GPUs")

    # Distribute tasks across GPUs
    gpu_tasks = [[] for _ in range(num_gpus)]
    for i, task in enumerate(tasks):
        gpu_tasks[i % num_gpus].append(task)

    for gpu_id, gpu_task_list in enumerate(gpu_tasks):
        print(f"GPU {gpu_id}: {len(gpu_task_list)} tasks")

    # Spawn workers
    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id in range(num_gpus):
        if not gpu_tasks[gpu_id]:
            continue
        p = mp.Process(target=worker_fn, args=(gpu_id, gpu_tasks[gpu_id], OUTPUT_DIR))
        p.start()
        processes.append(p)

    # Wait for all to finish
    for p in processes:
        p.join()

    print(f"\nAll tasks completed. Results saved to: {OUTPUT_DIR}")

    # Aggregate results
    aggregate_results()


def run_single_gpu(gpu_id: int, num_gpus: int):
    """Run tasks assigned to a specific GPU (for manual parallel execution)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = get_all_tasks()

    # Get tasks for this GPU
    my_tasks = [task for i, task in enumerate(tasks) if i % num_gpus == gpu_id]
    print(f"GPU {gpu_id}: processing {len(my_tasks)} tasks out of {len(tasks)} total")

    torch.cuda.set_device(gpu_id)

    for checkpoint_dir, step in my_tasks:
        result = process_single_task(checkpoint_dir, step, gpu_id)

        # Save individual result
        ckpt_name = Path(checkpoint_dir).name
        output_file = OUTPUT_DIR / f"{ckpt_name}_step{step:06d}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    print(f"GPU {gpu_id}: Done!")


def aggregate_results():
    """Combine the per-step JSONs in OUTPUT_DIR into summary.json and print an entropy/rank table."""
    results = {}

    for json_file in OUTPUT_DIR.glob("*.json"):
        if json_file.name == "summary.json":
            continue
        with open(json_file) as f:
            data = json.load(f)

        ckpt_name = data.get("checkpoint_name", "unknown")
        step = data.get("step", 0)

        if ckpt_name not in results:
            results[ckpt_name] = {}
        results[ckpt_name][step] = data

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Overall SVD Entropy by Checkpoint and Step")
    print("=" * 80)

    ckpt_names = sorted(results.keys())
    all_steps = sorted(set(step for ckpt in results.values() for step in ckpt.keys()))

    # Header
    header = f"{'Step':>8}" + "".join(f"{name:>25}" for name in ckpt_names)
    print(header)
    print("-" * len(header))

    for step in all_steps:
        row = f"{step:>8}"
        for ckpt_name in ckpt_names:
            if step in results[ckpt_name]:
                entropy = results[ckpt_name][step].get("overall_entropy")
                if entropy is not None:
                    row += f"{entropy:>25.6f}"
                else:
                    row += f"{'ERROR':>25}"
            else:
                row += f"{'N/A':>25}"
        print(row)

    print("\n" + "=" * 80)
    print("SUMMARY: Overall Stable Rank by Checkpoint and Step")
    print("=" * 80)
    print(header)
    print("-" * len(header))

    for step in all_steps:
        row = f"{step:>8}"
        for ckpt_name in ckpt_names:
            if step in results[ckpt_name]:
                rank = results[ckpt_name][step].get("overall_stable_rank")
                if rank is not None:
                    row += f"{rank:>25.2f}"
                else:
                    row += f"{'ERROR':>25}"
            else:
                row += f"{'N/A':>25}"
        print(row)

    # Save summary
    summary_file = OUTPUT_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


def main():
    """CLI entry: dispatch to aggregate-only / single-GPU / parallel mode based on flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=None, help="Run on specific GPU only")
    parser.add_argument("--num_gpus", type=int, default=8, help="Total number of GPUs")
    parser.add_argument("--aggregate_only", action="store_true", help="Only aggregate existing results")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_results()
        return

    if args.gpu_id is not None:
        # Single GPU mode (for manual parallel)
        run_single_gpu(args.gpu_id, args.num_gpus)
    else:
        # Automatic multi-GPU mode
        run_parallel(args.num_gpus)


if __name__ == "__main__":
    main()
