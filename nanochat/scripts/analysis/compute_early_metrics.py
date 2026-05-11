#!/usr/bin/env python3
"""
Compute SVD metrics for early checkpoints (step 10) using 2 GPUs in parallel.

Usage:
    python scripts/analysis/compute_early_metrics.py
"""

import json
import os
import sys
from pathlib import Path

# Add nanochat to path (parents[1] = repo root, where the nanochat/ package lives)
SCRIPT_DIR = Path(__file__).resolve().parent
NANOCHAT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(NANOCHAT_ROOT))

import torch
import torch.multiprocessing as mp

from nanochat.common import get_base_dir
from svd_metrics import load_model_for_metrics, collect_svd_metrics, mean_or_none

# Checkpoints to analyze (early-step ckpts of the paper's pretrains).
BASE_CKPT_DIR = Path(get_base_dir()) / "base_checkpoints"
CHECKPOINTS = [
    (str(BASE_CKPT_DIR / "d20_adam_lr0.001_early"), 10),
    (str(BASE_CKPT_DIR / "d20_muon_early"), 10),
]

# Output directory
OUTPUT_DIR = SCRIPT_DIR.parents[1] / "analysis_results" / "svd_results"


def process_checkpoint(gpu_id: int, checkpoint_dir: str, step: int) -> dict:
    """Process a single checkpoint on a specific GPU."""
    torch.cuda.set_device(gpu_id)
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
        import traceback
        traceback.print_exc()
        return {
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_name": ckpt_name,
            "step": step,
            "error": str(e),
        }


def worker_fn(gpu_id: int, checkpoint_dir: str, step: int, result_queue):
    """Run process_checkpoint in a subprocess and push the result onto the queue."""
    result = process_checkpoint(gpu_id, checkpoint_dir, step)
    result_queue.put((checkpoint_dir, result))


def main():
    """Spawn one worker per early checkpoint, then merge results into svd_results/summary.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(CHECKPOINTS)} early checkpoints...")

    # Use multiprocessing to run on 2 GPUs
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()

    processes = []
    for gpu_id, (ckpt_dir, step) in enumerate(CHECKPOINTS):
        p = mp.Process(target=worker_fn, args=(gpu_id, ckpt_dir, step, result_queue))
        p.start()
        processes.append(p)

    # Wait for all to finish
    for p in processes:
        p.join()

    # Collect results
    results = {}
    while not result_queue.empty():
        ckpt_dir, result = result_queue.get()
        ckpt_name = Path(ckpt_dir).name
        results[ckpt_name] = result

        # Save individual result
        output_file = OUTPUT_DIR / f"{ckpt_name}_step{result['step']:06d}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {output_file}")

    # Update summary.json
    summary_path = OUTPUT_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    # Add early checkpoint data to summary
    # Map early checkpoints to their corresponding main runs
    mapping = {
        "d20_adam_lr0.001_early": "d20_adam_lr0.001",
        "d20_muon_early": "d20_muon",
    }

    for early_name, result in results.items():
        main_name = mapping.get(early_name)
        if main_name and main_name in summary:
            step = result["step"]
            summary[main_name][str(step)] = result
            print(f"Added step {step} to {main_name}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nUpdated summary.json")

    # Print summary
    print("\n" + "=" * 60)
    print("Early Checkpoint SVD Metrics:")
    print("=" * 60)
    for name, result in results.items():
        if "error" not in result:
            print(f"{name}: entropy={result['overall_entropy']:.4f}, rank={result['overall_stable_rank']:.2f}")
        else:
            print(f"{name}: ERROR - {result['error']}")


if __name__ == "__main__":
    main()
