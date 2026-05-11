#!/usr/bin/env python3
"""
Compute SVD-based metrics (entropy, stable rank, singular values) for nanochat checkpoints.

Usage:
    python scripts/analysis/svd_metrics.py --checkpoint_dir /path/to/checkpoint --step 21400
    python scripts/analysis/svd_metrics.py --checkpoint_dir /path/to/checkpoint  # uses last step
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Add the nanochat top-level dir (parents[1]) to sys.path so `import nanochat` resolves.
SCRIPT_DIR = Path(__file__).resolve().parent
NANOCHAT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(NANOCHAT_ROOT))

from nanochat.checkpoint_manager import load_checkpoint, find_last_step
from nanochat.gpt import GPT, GPTConfig

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from safetensors.torch import save_file
except ImportError:
    save_file = None


# Group definitions for nanochat GPT model
# transformer.h.{layer}.attn.c_q, c_k, c_v, c_proj
# transformer.h.{layer}.mlp.c_fc, c_proj
GROUP_ORDER = ["AttnQ", "AttnK", "AttnV", "AttnProj", "MLP_FC", "MLP_Proj", "Embedding", "LMHead"]


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


def compute_svd_metrics_for_matrix(
    name: str,
    group: str,
    matrix: torch.Tensor,
    group_entropies: Dict[str, List[float]],
    group_stable_ranks: Dict[str, List[float]],
    singular_values_data: Optional[Dict[str, torch.Tensor]] = None,
    skipped: Optional[Dict[str, int]] = None,
) -> bool:
    """
    Compute SVD metrics for a single matrix.

    Returns True if successful, False otherwise.
    """
    if skipped is None:
        skipped = {}

    try:
        device = matrix.device
        if device.type != "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                raise RuntimeError("CUDA is required for SVD metrics.")

        matrix = matrix.detach().to(device=device, dtype=torch.float32)
        s = torch.linalg.svdvals(matrix)

        if s.numel() == 0:
            return False

        # Save singular values if requested
        if singular_values_data is not None:
            key = f"{name}__{group}"
            singular_values_data[key] = s.detach().cpu().float()

        # Compute stable rank: ||A||_F^2 / ||A||_2^2
        s2 = s.pow(2)
        denom = s.max().pow(2) + 1e-12
        stable_rank = s2.sum() / denom
        group_stable_ranks.setdefault(group, []).append(float(stable_rank.item()))

        # Compute normalized entropy
        p = s2 / s2.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-12)) / math.log(p.numel())
        group_entropies.setdefault(group, []).append(float(entropy.item()))

        return True
    except Exception as exc:
        skipped[str(exc)] = skipped.get(str(exc), 0) + 1
        return False


def collect_svd_metrics(
    model: torch.nn.Module,
    groups: Optional[List[str]] = None,
    save_singular_values: bool = False,
) -> Tuple[
    Dict[str, Dict[str, Optional[float]]],  # entropy by group
    Dict[str, Dict[str, Optional[float]]],  # stable rank by group
    Dict[str, int],  # skipped counts
    Optional[Dict[str, torch.Tensor]],  # singular values
]:
    """Run SVD on every 2D parameter, group results by GROUP_ORDER, and return per-group entropy/stable-rank
    along with skipped-error counts and (optionally) the raw singular value tensors.
    """
    selected_groups = set(groups or GROUP_ORDER)
    group_entropies: Dict[str, List[float]] = {}
    group_stable_ranks: Dict[str, List[float]] = {}
    skipped: Dict[str, int] = {}
    singular_values_data: Optional[Dict[str, torch.Tensor]] = {} if save_singular_values else None

    # Collect candidates
    candidates: List[Tuple[str, torch.nn.Parameter, str]] = []
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        group = classify_group(name)
        if group is None or group not in selected_groups:
            continue
        candidates.append((name, param, group))

    # Process with progress bar
    progress = None
    if tqdm is not None and candidates:
        progress = tqdm(
            total=len(candidates),
            desc="SVD",
            unit="mat",
            dynamic_ncols=True,
            ascii=True,
            file=sys.stdout,
        )

    for name, param, group in candidates:
        rows, cols = param.shape
        if progress is not None:
            progress.set_postfix(group=group, shape=f"{rows}x{cols}", refresh=False)

        compute_svd_metrics_for_matrix(
            name,
            group,
            param,
            group_entropies,
            group_stable_ranks,
            singular_values_data=singular_values_data,
            skipped=skipped,
        )

        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()

    # Aggregate results
    svd_entropy = {
        group: {"mean": mean_or_none(values), "count": len(values)}
        for group, values in group_entropies.items()
    }
    stable_rank_groups = {
        group: {"mean": mean_or_none(values), "count": len(values)}
        for group, values in group_stable_ranks.items()
    }

    return svd_entropy, stable_rank_groups, skipped, singular_values_data


def format_group_values(groups: Dict[str, Dict], key: str = "mean", order: Optional[List[str]] = None) -> str:
    """Format group values for display."""
    parts = []
    order = order or GROUP_ORDER
    for group in order:
        if group not in groups:
            continue
        value = groups[group].get(key)
        if value is None:
            continue
        parts.append(f"{group}={value:.6f}")
    return ", ".join(parts)


def load_model_for_metrics(checkpoint_dir: str, step: int, device: torch.device) -> GPT:
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


def main(argv: List[str] = None) -> None:
    """CLI: load a single checkpoint, print per-group SVD metrics, and dump JSON (and optional safetensors)."""
    parser = argparse.ArgumentParser(description="Compute SVD metrics for nanochat checkpoints.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint directory path.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: last step).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path.")
    parser.add_argument("--save_singular_values", action="store_true", help="Save singular values to safetensors.")
    parser.add_argument("--groups", type=str, default=None, help="Comma-separated groups to include.")
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Resolve step
    step = args.step
    if step is None:
        step = find_last_step(str(checkpoint_dir))
        print(f"Using last step: {step}")

    # Parse groups
    groups = None
    if args.groups:
        groups = [g.strip() for g in args.groups.split(",")]

    # Setup device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Step: {step}")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model, meta_data = load_model_for_metrics(str(checkpoint_dir), step, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Compute metrics
    print("Computing SVD metrics...")
    svd_entropy, stable_rank_groups, skipped, singular_values_data = collect_svd_metrics(
        model,
        groups=groups,
        save_singular_values=args.save_singular_values,
    )

    # Print results
    print("\n" + "=" * 60)
    print("SVD Entropy (normalized):")
    print("=" * 60)
    for group in GROUP_ORDER:
        if group in svd_entropy:
            data = svd_entropy[group]
            print(f"  {group:12s}: mean={data['mean']:.6f}, count={data['count']}")

    overall_entropy = mean_or_none([
        v["mean"] for v in svd_entropy.values() if v["mean"] is not None
    ])
    print(f"  {'Overall':12s}: mean={overall_entropy:.6f}" if overall_entropy else "")

    print("\n" + "=" * 60)
    print("Stable Rank:")
    print("=" * 60)
    for group in GROUP_ORDER:
        if group in stable_rank_groups:
            data = stable_rank_groups[group]
            print(f"  {group:12s}: mean={data['mean']:.6f}, count={data['count']}")

    overall_rank = mean_or_none([
        v["mean"] for v in stable_rank_groups.values() if v["mean"] is not None
    ])
    print(f"  {'Overall':12s}: mean={overall_rank:.6f}" if overall_rank else "")

    if skipped:
        print(f"\nSkipped: {skipped}")

    # Save results
    results = {
        "checkpoint_dir": str(checkpoint_dir),
        "step": step,
        "model_config": meta_data.get("model_config", {}),
        "user_config": meta_data.get("user_config", {}),
        "svd_entropy": svd_entropy,
        "stable_rank": stable_rank_groups,
        "skipped": skipped,
        "overall_entropy": overall_entropy,
        "overall_stable_rank": overall_rank,
    }

    output_path = args.output
    if output_path is None:
        output_path = checkpoint_dir / f"svd_metrics_{step:06d}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Save singular values if requested
    if args.save_singular_values and singular_values_data:
        if save_file is None:
            print("Warning: safetensors not available, skipping singular values save.")
        else:
            sv_path = checkpoint_dir / f"singular_values_{step:06d}.safetensors"
            save_file(singular_values_data, str(sv_path))
            print(f"Singular values saved to: {sv_path}")


if __name__ == "__main__":
    main()
