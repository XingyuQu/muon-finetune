#!/usr/bin/env python3
"""
Compute spectral norm (largest singular value) for nanochat checkpoints.

Usage:
    python scripts/analysis/compute_spectral_norm.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

# Add nanochat to path (parents[1] = repo root, where the nanochat/ package lives)
SCRIPT_DIR = Path(__file__).resolve().parent
NANOCHAT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(NANOCHAT_ROOT))

from nanochat.checkpoint_manager import load_checkpoint
from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig

try:
    from safetensors.torch import save_file
except ImportError:
    save_file = None

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


def compute_spectral_norm_and_svd(
    model: torch.nn.Module,
    device: torch.device,
    save_singular_values: bool = True,
) -> Tuple[Dict[str, Dict[str, Optional[float]]], Optional[Dict[str, torch.Tensor]]]:
    """
    Compute spectral norm (largest singular value) and optionally save all singular values.

    Returns:
        Tuple of:
        - Dict mapping group name to {"mean": float, "max": float, "min": float, "count": int}
        - Dict mapping "name__group" to singular values tensor (if save_singular_values=True)
    """
    group_spectral_norms: Dict[str, List[float]] = {}
    singular_values_data: Optional[Dict[str, torch.Tensor]] = {} if save_singular_values else None

    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        group = classify_group(name)
        if group is None:
            continue

        # Compute all singular values
        matrix = param.detach().to(device=device, dtype=torch.float32)
        try:
            s = torch.linalg.svdvals(matrix)
            spectral_norm = s[0].item()  # Largest singular value
            group_spectral_norms.setdefault(group, []).append(spectral_norm)

            # Save all singular values
            if singular_values_data is not None:
                key = f"{name}__{group}"
                singular_values_data[key] = s.detach().cpu().float()
        except Exception as e:
            print(f"  Warning: Failed to compute SVD for {name}: {e}")
            continue

    # Aggregate results
    results = {}
    for group, values in group_spectral_norms.items():
        results[group] = {
            "mean": mean_or_none(values),
            "max": max(values) if values else None,
            "min": min(values) if values else None,
            "count": len(values),
        }

    return results, singular_values_data


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


def process_checkpoint(gpu_id: int, checkpoint_dir: str, step: int, save_sv: bool = True) -> dict:
    """Process a single checkpoint on a specific GPU."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    ckpt_name = Path(checkpoint_dir).name

    print(f"[GPU {gpu_id}] Processing {ckpt_name} step {step}...")

    try:
        # Load model
        model, _ = load_model_for_metrics(checkpoint_dir, step, device)

        # Compute spectral norm and singular values
        spectral_norm_groups, singular_values_data = compute_spectral_norm_and_svd(
            model, device, save_singular_values=save_sv
        )

        # Compute overall spectral norm
        overall_spectral_norm = mean_or_none([
            v["mean"] for v in spectral_norm_groups.values() if v["mean"] is not None
        ])

        # Save singular values to safetensors
        if save_sv and singular_values_data and save_file is not None:
            sv_path = OUTPUT_DIR / f"{ckpt_name}_step{step:06d}_singular_values.safetensors"
            save_file(singular_values_data, str(sv_path))
            print(f"[GPU {gpu_id}] Saved singular values to {sv_path}")

        result = {
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_name": ckpt_name,
            "step": step,
            "spectral_norm": spectral_norm_groups,
            "overall_spectral_norm": overall_spectral_norm,
        }

        # Clean up
        del model
        if singular_values_data:
            del singular_values_data
        torch.cuda.empty_cache()

        print(f"[GPU {gpu_id}] Done {ckpt_name} step {step}: overall_spectral_norm={overall_spectral_norm:.4f}")
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


def worker_fn(gpu_id: int, tasks: List[Tuple[str, int]], result_queue, save_sv: bool = True):
    """Process every assigned (checkpoint, step) pair on a single GPU and push results to the queue."""
    for checkpoint_dir, step in tasks:
        result = process_checkpoint(gpu_id, checkpoint_dir, step, save_sv=save_sv)
        result_queue.put(result)


def main():
    """Distribute (checkpoint, step) tasks across visible GPUs and merge spectral_norm into svd_results/summary.json."""
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

    # Use multiple GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\nUsing {num_gpus} GPUs")

    if num_gpus == 0:
        raise RuntimeError("No GPUs available. Spectral norm computation requires GPU.")

    # Distribute tasks across GPUs
    gpu_tasks = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        gpu_tasks[i % num_gpus].append(task)

    # Use multiprocessing
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()

    processes = []
    for gpu_id in range(num_gpus):
        if gpu_tasks[gpu_id]:
            p = mp.Process(target=worker_fn, args=(gpu_id, gpu_tasks[gpu_id], result_queue, True))
            p.start()
            processes.append(p)

    # Wait for all to finish
    for p in processes:
        p.join()

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

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

    # Update summary with spectral norm results
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

        # Add spectral norm data
        summary[main_name][step_key]["spectral_norm"] = result["spectral_norm"]
        summary[main_name][step_key]["overall_spectral_norm"] = result["overall_spectral_norm"]

        print(f"Updated {main_name} step {step} with spectral_norm")

    # Save updated summary
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nUpdated summary.json")

    # Print summary
    print("\n" + "=" * 60)
    print("Spectral Norm Results:")
    print("=" * 60)
    for result in sorted(results, key=lambda x: (x["checkpoint_name"], x["step"])):
        if "error" not in result:
            ckpt_name = result["checkpoint_name"]
            step = result["step"]
            overall = result["overall_spectral_norm"]
            print(f"{ckpt_name} step {step}: overall_spectral_norm={overall:.4f}")
        else:
            print(f"{result['checkpoint_name']} step {result['step']}: ERROR - {result['error']}")


if __name__ == "__main__":
    main()
