#!/usr/bin/env python3
"""
Compute max-norm (element-wise maximum absolute value) for nanochat checkpoints.

This script computes the max-norm for each weight matrix and adds the results
to the existing summary.json file.

Usage:
    python scripts/analysis/compute_max_norm.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Add nanochat to path (parents[1] = repo root, where the nanochat/ package lives)
SCRIPT_DIR = Path(__file__).resolve().parent
NANOCHAT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(NANOCHAT_ROOT))

from nanochat.checkpoint_manager import load_checkpoint
from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig

# Group definitions
GROUP_ORDER = ["AttnQ", "AttnK", "AttnV", "AttnProj", "MLP_FC", "MLP_Proj", "Embedding", "LMHead"]

# Checkpoints to analyze. By default we run on the two pretrains used in the
# paper across all training steps; uncomment the early-step entries to also
# include the early checkpoints used for the appendix figure.
BASE_CKPT_DIR = Path(get_base_dir()) / "base_checkpoints"
CHECKPOINTS = [
    (str(BASE_CKPT_DIR / "d20_adam_lr0.001"), None),  # all steps
    (str(BASE_CKPT_DIR / "d20_muon"), None),          # all steps
    # (str(BASE_CKPT_DIR / "d20_adam_lr0.001_early"), 10),
    # (str(BASE_CKPT_DIR / "d20_muon_early"), 10),
]

OUTPUT_DIR = SCRIPT_DIR.parents[1] / "analysis_results" / "svd_results"


def classify_group(name: str) -> Optional[str]:
    """Classify a parameter name into a group."""
    lower = name.lower()
    if "attn.c_q" in lower:
        return "AttnQ"
    if "attn.c_k" in lower:
        return "AttnK"
    if "attn.c_v" in lower:
        return "AttnV"
    if "attn.c_proj" in lower:
        return "AttnProj"
    if "mlp.c_fc" in lower:
        return "MLP_FC"
    if "mlp.c_proj" in lower:
        return "MLP_Proj"
    if "wte" in lower or "embedding" in lower:
        return "Embedding"
    if "lm_head" in lower:
        return "LMHead"
    return None


def mean_or_none(values: List[float]) -> Optional[float]:
    """Compute mean or return None if empty."""
    if not values:
        return None
    return sum(values) / len(values)


def compute_max_norm(model: torch.nn.Module) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute max-norm (element-wise maximum absolute value) for all 2D weight matrices.

    Returns:
        Dict mapping group name to {"mean": float, "count": int, "max": float, "min": float}
    """
    group_max_norms: Dict[str, List[float]] = {}

    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        group = classify_group(name)
        if group is None:
            continue

        # Compute max-norm: max(|W|)
        max_norm = torch.abs(param).max().item()
        group_max_norms.setdefault(group, []).append(max_norm)

    # Aggregate results
    results = {}
    for group, values in group_max_norms.items():
        results[group] = {
            "mean": mean_or_none(values),
            "max": max(values) if values else None,
            "min": min(values) if values else None,
            "count": len(values),
        }

    return results


def load_model_for_metrics(checkpoint_dir: str, step: int, device: torch.device) -> Tuple[GPT, dict]:
    """Load model from checkpoint for metrics computation."""
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)

    # Fix torch compile issue
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}

    # Build model
    model_config_kwargs = meta_data["model_config"]
    model_config = GPTConfig(**model_config_kwargs)

    with torch.device("meta"):
        model = GPT(model_config)

    model.to_empty(device=device)
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    return model, meta_data


def get_available_steps(checkpoint_dir: str) -> List[int]:
    """Get all available checkpoint steps."""
    ckpt_path = Path(checkpoint_dir)
    steps = []
    # Look for model_XXXXXX.pt files
    for f in ckpt_path.glob("model_*.pt"):
        try:
            step = int(f.stem.split("_")[1])
            steps.append(step)
        except (ValueError, IndexError):
            continue
    return sorted(steps)


def process_checkpoint(worker_id: int, checkpoint_dir: str, step: int, device: torch.device) -> dict:
    """Process a single checkpoint."""
    ckpt_name = Path(checkpoint_dir).name

    print(f"[Worker {worker_id}] Processing {ckpt_name} step {step} on {device}...")

    try:
        # Load model
        model, _ = load_model_for_metrics(checkpoint_dir, step, device)

        # Compute max-norm
        max_norm_groups = compute_max_norm(model)

        # Compute overall max-norm
        overall_max_norm = mean_or_none([
            v["mean"] for v in max_norm_groups.values() if v["mean"] is not None
        ])

        result = {
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_name": ckpt_name,
            "step": step,
            "max_norm": max_norm_groups,
            "overall_max_norm": overall_max_norm,
        }

        # Clean up
        del model

        print(f"[Worker {worker_id}] Done {ckpt_name} step {step}: overall_max_norm={overall_max_norm:.4f}")
        return result

    except Exception as e:
        print(f"[Worker {worker_id}] ERROR processing {ckpt_name} step {step}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_name": ckpt_name,
            "step": step,
            "error": str(e),
        }


def main():
    """Iterate over all (checkpoint, step) tasks on CPU and merge max-norm into svd_results/summary.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all tasks
    all_tasks = []
    for checkpoint_dir, specific_step in CHECKPOINTS:
        if specific_step is not None:
            all_tasks.append((checkpoint_dir, specific_step))
        else:
            steps = get_available_steps(checkpoint_dir)
            for step in steps:
                all_tasks.append((checkpoint_dir, step))

    print(f"Total tasks: {len(all_tasks)}")
    for ckpt_dir, step in all_tasks:
        print(f"  - {Path(ckpt_dir).name} step {step}")

    # Use CPU for max-norm computation (fast, doesn't need GPU)
    device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    # Process sequentially (max-norm is very fast)
    results = []
    for i, (checkpoint_dir, step) in enumerate(all_tasks):
        result = process_checkpoint(i, checkpoint_dir, step, device)
        results.append(result)

    # Load existing summary.json
    summary_path = OUTPUT_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    # Map early checkpoints to main runs
    mapping = {
        "d20_adam_lr0.001_early": "d20_adam_lr0.001",
        "d20_muon_early": "d20_muon",
    }

    # Update summary with max-norm results
    for result in results:
        if "error" in result:
            continue

        ckpt_name = result["checkpoint_name"]
        step = result["step"]

        # Determine the main run name
        main_name = mapping.get(ckpt_name, ckpt_name)

        if main_name not in summary:
            summary[main_name] = {}

        step_key = str(step)
        if step_key not in summary[main_name]:
            summary[main_name][step_key] = {}

        # Add max-norm data
        summary[main_name][step_key]["max_norm"] = result["max_norm"]
        summary[main_name][step_key]["overall_max_norm"] = result["overall_max_norm"]

        print(f"Updated {main_name} step {step} with max_norm")

    # Save updated summary
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nUpdated summary.json")

    # Print summary
    print("\n" + "=" * 60)
    print("Max-Norm Results:")
    print("=" * 60)
    for result in sorted(results, key=lambda x: (x["checkpoint_name"], x["step"])):
        if "error" not in result:
            ckpt_name = result["checkpoint_name"]
            step = result["step"]
            overall = result["overall_max_norm"]
            print(f"{ckpt_name} step {step}: overall_max_norm={overall:.4f}")
        else:
            print(f"{result['checkpoint_name']} step {result['step']}: ERROR - {result['error']}")


if __name__ == "__main__":
    main()
