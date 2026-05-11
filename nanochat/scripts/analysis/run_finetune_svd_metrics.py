#!/usr/bin/env python3
"""
Run SVD metrics on fine-tuned WikiText checkpoints (with intermediate saves).

Discovers all model_*.pt files in wikitext_checkpoints directories for the
8 best-LR experiments, computes SVD entropy, stable rank, spectral norm,
and max-norm, and saves results.

Usage:
    # Run all tasks across 8 GPUs
    python scripts/analysis/run_finetune_svd_metrics.py

    # Run on specific GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_finetune_svd_metrics.py --gpu_id 0 --num_gpus 8

    # Only aggregate existing results
    python scripts/analysis/run_finetune_svd_metrics.py --aggregate_only
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
NANOCHAT_ROOT = SCRIPT_DIR.parents[1]  # repo root
sys.path.insert(0, str(NANOCHAT_ROOT))

import torch
import torch.multiprocessing as mp

from nanochat.common import get_base_dir
from svd_metrics import load_model_for_metrics, collect_svd_metrics, GROUP_ORDER, mean_or_none

try:
    from safetensors.torch import save_file
except ImportError:
    save_file = None

# The 8 best-LR fine-tuning experiment directory names
EXPERIMENT_DIRS = [
    # Muon pretrain
    "d20_muon_full-muon_lr0.9_ep1_seed0",
    "d20_muon_full-adam_lr0.009_ep1_seed0",
    "d20_muon_lora-muon_r8_lr0.9_ep1_seed0",
    "d20_muon_lora-adam_r8_lr0.1_ep1_seed0",
    # Adam pretrain
    "d20_adam_lr0.001_full-adam_lr0.03_ep1_seed0",
    "d20_adam_lr0.001_full-muon_lr0.5_ep1_seed0",
    "d20_adam_lr0.001_lora-adam_r8_lr0.3_ep1_seed0",
    "d20_adam_lr0.001_lora-muon_r8_lr0.7_ep1_seed0",
]

# Base directory for wikitext checkpoints (uses NANOCHAT_BASE_DIR via get_base_dir,
# matching the rest of the project rather than the older NANOCHAT_CACHE convention).
WIKITEXT_DIR = Path(get_base_dir()) / "wikitext_checkpoints"

# Output directory
OUTPUT_DIR = SCRIPT_DIR.parents[1] / "analysis_results" / "svd_results_finetune"


def find_checkpoint_dirs() -> List[Path]:
    """Find the checkpoint directories for the 8 experiments."""
    dirs = []
    for dirname in EXPERIMENT_DIRS:
        dirpath = WIKITEXT_DIR / dirname
        if dirpath.exists():
            dirs.append(dirpath)
        else:
            print(f"WARNING: {dirpath} not found, skipping.")
    return dirs


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
    dirs = find_checkpoint_dirs()
    for ckpt_dir in dirs:
        steps = get_all_steps(ckpt_dir)
        print(f"  {ckpt_dir.name}: steps = {steps}")
        for step in steps:
            tasks.append((str(ckpt_dir), step))
    return tasks


def classify_weight(name: str) -> Optional[str]:
    """Classify a weight parameter into a group."""
    if "c_q" in name:
        return "AttnQ"
    elif "c_k" in name:
        return "AttnK"
    elif "c_v" in name:
        return "AttnV"
    elif "c_proj" in name:
        return "AttnProj"
    elif "c_fc" in name:
        return "MLP_FC"
    elif name.endswith("c_proj.weight") and "mlp" not in name.lower():
        return "AttnProj"
    elif "mlp" in name.lower() and "proj" in name.lower():
        return "MLP_Proj"
    elif "wte" in name:
        return "Embedding"
    elif "lm_head" in name:
        return "LMHead"
    return None


def compute_spectral_norm(model, device) -> Dict:
    """Compute spectral norm for all weight matrices."""
    results = {}
    for group in GROUP_ORDER:
        results[group] = {"values": [], "count": 0}

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        group = classify_weight(name)
        if group is None:
            continue

        matrix = param.data.float().to(device)
        if matrix.dim() > 2:
            matrix = matrix.reshape(matrix.shape[0], -1)

        s_max = torch.linalg.svdvals(matrix)[0].item()
        results[group]["values"].append(s_max)
        results[group]["count"] += 1

    # Aggregate
    for group in GROUP_ORDER:
        vals = results[group]["values"]
        if vals:
            results[group]["mean"] = sum(vals) / len(vals)
            results[group]["max"] = max(vals)
            results[group]["min"] = min(vals)
        else:
            results[group]["mean"] = None
            results[group]["max"] = None
            results[group]["min"] = None
        del results[group]["values"]

    return results


def compute_max_norm(model) -> Dict:
    """Compute max-norm for all weight matrices."""
    results = {}
    for group in GROUP_ORDER:
        results[group] = {"values": [], "count": 0}

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        group = classify_weight(name)
        if group is None:
            continue

        max_val = param.data.float().abs().max().item()
        results[group]["values"].append(max_val)
        results[group]["count"] += 1

    for group in GROUP_ORDER:
        vals = results[group]["values"]
        if vals:
            results[group]["mean"] = sum(vals) / len(vals)
            results[group]["max"] = max(vals)
            results[group]["min"] = min(vals)
        else:
            results[group]["mean"] = None
            results[group]["max"] = None
            results[group]["min"] = None
        del results[group]["values"]

    return results


def process_single_task(checkpoint_dir: str, step: int, gpu_id: int) -> dict:
    """Process a single checkpoint/step on a specific GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    ckpt_name = Path(checkpoint_dir).name

    print(f"[GPU {gpu_id}] Processing {ckpt_name} step {step}...")

    try:
        model, meta_data = load_model_for_metrics(checkpoint_dir, step, device)

        # SVD entropy + stable rank + singular values
        svd_entropy, stable_rank_groups, skipped, singular_values_data = collect_svd_metrics(
            model, save_singular_values=True,
        )

        # Spectral norm
        spectral_norm = compute_spectral_norm(model, device)

        # Max-norm
        max_norm = compute_max_norm(model)

        # Overall metrics
        overall_entropy = mean_or_none([
            v["mean"] for v in svd_entropy.values() if v["mean"] is not None
        ])
        overall_rank = mean_or_none([
            v["mean"] for v in stable_rank_groups.values() if v["mean"] is not None
        ])
        overall_spectral = mean_or_none([
            v["mean"] for v in spectral_norm.values() if v["mean"] is not None
        ])
        overall_maxnorm = mean_or_none([
            v["mean"] for v in max_norm.values() if v["mean"] is not None
        ])

        result = {
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_name": ckpt_name,
            "step": step,
            "svd_entropy": svd_entropy,
            "stable_rank": stable_rank_groups,
            "spectral_norm": spectral_norm,
            "max_norm": max_norm,
            "overall_entropy": overall_entropy,
            "overall_stable_rank": overall_rank,
            "overall_spectral_norm": overall_spectral,
            "overall_max_norm": overall_maxnorm,
            "skipped": skipped,
            "model_config": meta_data.get("model_config", {}),
            "user_config": meta_data.get("user_config", {}),
        }

        # Save singular values
        if singular_values_data and save_file is not None:
            sv_path = OUTPUT_DIR / f"{ckpt_name}_step{step:06d}_singular_values.safetensors"
            save_file({k: v.cpu() for k, v in singular_values_data.items()}, str(sv_path))

        del model, singular_values_data
        torch.cuda.empty_cache()

        print(f"[GPU {gpu_id}] Done {ckpt_name} step {step}: "
              f"entropy={overall_entropy:.4f}, rank={overall_rank:.2f}, "
              f"spectral={overall_spectral:.4f}, maxnorm={overall_maxnorm:.4f}")
        return result

    except Exception as e:
        import traceback
        print(f"[GPU {gpu_id}] ERROR processing {ckpt_name} step {step}: {e}")
        traceback.print_exc()
        return {
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_name": ckpt_name,
            "step": step,
            "error": str(e),
        }


def worker_fn(gpu_id: int, tasks: List[Tuple[str, int]], output_dir: Path):
    """Process every assigned (checkpoint, step) on this GPU; skip outputs that already exist."""
    torch.cuda.set_device(gpu_id)

    for checkpoint_dir, step in tasks:
        ckpt_name = Path(checkpoint_dir).name
        output_file = output_dir / f"{ckpt_name}_step{step:06d}.json"

        # Skip if already computed
        if output_file.exists():
            print(f"[GPU {gpu_id}] Skipping {ckpt_name} step {step} (already exists)")
            continue

        result = process_single_task(checkpoint_dir, step, gpu_id)

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)


def run_parallel(num_gpus: int = 8):
    """Run all tasks in parallel across multiple GPUs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = get_all_tasks()
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Using {num_gpus} GPUs")

    if not tasks:
        print("No tasks found. Run the fine-tuning experiments first.")
        return

    gpu_tasks = [[] for _ in range(num_gpus)]
    for i, task in enumerate(tasks):
        gpu_tasks[i % num_gpus].append(task)

    for gpu_id, gpu_task_list in enumerate(gpu_tasks):
        print(f"GPU {gpu_id}: {len(gpu_task_list)} tasks")

    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id in range(num_gpus):
        if not gpu_tasks[gpu_id]:
            continue
        p = mp.Process(target=worker_fn, args=(gpu_id, gpu_tasks[gpu_id], OUTPUT_DIR))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"\nAll tasks completed. Results saved to: {OUTPUT_DIR}")
    aggregate_results()


def run_single_gpu(gpu_id: int, num_gpus: int):
    """Run tasks assigned to a specific GPU."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = get_all_tasks()
    my_tasks = [task for i, task in enumerate(tasks) if i % num_gpus == gpu_id]
    print(f"GPU {gpu_id}: processing {len(my_tasks)} tasks out of {len(tasks)} total")

    torch.cuda.set_device(gpu_id)

    for checkpoint_dir, step in my_tasks:
        ckpt_name = Path(checkpoint_dir).name
        output_file = OUTPUT_DIR / f"{ckpt_name}_step{step:06d}.json"

        if output_file.exists():
            print(f"[GPU {gpu_id}] Skipping {ckpt_name} step {step} (already exists)")
            continue

        result = process_single_task(checkpoint_dir, step, gpu_id)

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    print(f"GPU {gpu_id}: Done!")


def aggregate_results():
    """Combine the per-step JSONs in OUTPUT_DIR into summary.json and print a metrics table per checkpoint."""
    results = {}

    for json_file in sorted(OUTPUT_DIR.glob("*.json")):
        if json_file.name == "summary.json":
            continue
        with open(json_file) as f:
            data = json.load(f)

        ckpt_name = data.get("checkpoint_name", "unknown")
        step = data.get("step", 0)

        if ckpt_name not in results:
            results[ckpt_name] = {}
        results[ckpt_name][step] = data

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY: Fine-tuning SVD Metrics by Checkpoint and Step")
    print("=" * 100)

    for ckpt_name in sorted(results.keys()):
        print(f"\n--- {ckpt_name} ---")
        print(f"  {'Step':>6}  {'Entropy':>10}  {'StableRank':>12}  {'SpectralNorm':>14}  {'MaxNorm':>10}")
        for step in sorted(results[ckpt_name].keys()):
            d = results[ckpt_name][step]
            if "error" in d:
                print(f"  {step:>6}  ERROR: {d['error']}")
                continue
            e = d.get("overall_entropy", 0)
            r = d.get("overall_stable_rank", 0)
            s = d.get("overall_spectral_norm", 0)
            m = d.get("overall_max_norm", 0)
            print(f"  {step:>6}  {e:>10.6f}  {r:>12.2f}  {s:>14.4f}  {m:>10.4f}")

    # Save summary
    summary_file = OUTPUT_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


def main():
    """CLI entry: dispatch to aggregate-only / single-GPU / parallel mode based on flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_results()
        return

    if args.gpu_id is not None:
        run_single_gpu(args.gpu_id, args.num_gpus)
    else:
        run_parallel(args.num_gpus)


if __name__ == "__main__":
    main()
