from __future__ import annotations

import math
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from utils import mean_or_none

SVD_GROUP_ORDER = ["AttnQO", "AttnKV", "Dense"]
LORA_GROUP_SUFFIXES = ["A", "B"]
LAYER_TOKENS = {"layers", "layer", "blocks", "block"}

LORA_A_TAGS = ("lora_A", "lora_embedding_A")
LORA_B_TAGS = ("lora_B", "lora_embedding_B")


def split_lora_param_name(name: str) -> Optional[Tuple[str, str, str]]:
    """Split a PEFT param name into (base_prefix, 'A'|'B', adapter); None if not LoRA."""
    for tag, part in [(tag, "A") for tag in LORA_A_TAGS] + [(tag, "B") for tag in LORA_B_TAGS]:
        token = f"{tag}."
        if token not in name:
            continue
        prefix, rest = name.split(token, 1)
        adapter = "default"
        if rest:
            adapter = rest.split(".", 1)[0]
            if adapter == "weight":
                adapter = "default"
        if not prefix.endswith("."):
            prefix = f"{prefix}."
        return prefix, part, adapter
    return None


def parse_lora_group_name(name: str) -> Optional[Tuple[str, str]]:
    """Parse '<Base>_A' / '<Base>_B' into (base, suffix); None if not a LoRA group name."""
    for suffix in LORA_GROUP_SUFFIXES:
        token = f"_{suffix}"
        if name.endswith(token):
            base = name[: -len(token)]
            if base in SVD_GROUP_ORDER:
                return base, suffix
    return None


def expand_lora_groups(groups: List[str]) -> List[str]:
    """Expand bare base groups (e.g. 'AttnQO') into ['AttnQO_A', 'AttnQO_B']; pass others through."""
    expanded: List[str] = []
    for group in groups:
        if group in SVD_GROUP_ORDER:
            expanded.extend([f"{group}_{suffix}" for suffix in LORA_GROUP_SUFFIXES])
        else:
            expanded.append(group)
    return expanded


def _svd_metrics_for_matrix(
    name: str,
    group: str,
    matrix: torch.Tensor,
    group_entropies: Dict[str, List[float]],
    group_stable_ranks: Dict[str, List[float]],
    skipped: Dict[str, int],
) -> bool:
    """Compute normalized spectral entropy + stable rank on `matrix`; append to group lists."""
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
        s2 = s.pow(2)
        denom = s.max().pow(2) + 1e-12
        stable_rank = s2.sum() / denom
        group_stable_ranks.setdefault(group, []).append(float(stable_rank.item()))
        p = s2 / s2.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-12)) / math.log(p.numel())
        group_entropies.setdefault(group, []).append(float(entropy.item()))
        return True
    except Exception as exc:
        skipped[str(exc)] = skipped.get(str(exc), 0) + 1
        return False


def format_group_values(
    groups: Dict[str, Any],
    key: str = "mean",
    order: Optional[List[str]] = None,
) -> str:
    """Render `{group: {key: value}}` as a 'GroupA=0.123, GroupB=0.456' string."""
    parts = []
    order = order or SVD_GROUP_ORDER
    for group in order:
        if group not in groups:
            continue
        value = groups[group].get(key)
        if value is None:
            continue
        parts.append(f"{group}={value:.6f}")
    return ", ".join(parts)


def format_group_counts(counts: Dict[str, int], order: Optional[List[str]] = None) -> str:
    """Render `{group: count}` as 'GroupA=12, GroupB=12' (zero-fills missing groups)."""
    parts = []
    order = order or SVD_GROUP_ORDER
    for group in order:
        parts.append(f"{group}={counts.get(group, 0)}")
    return ", ".join(parts)


def simplify_param_name(name: str) -> str:
    """Strip leading 'model.' and trailing '.weight'/'.bias' for compact log labels."""
    if name.startswith("model."):
        name = name[6:]
    if name.endswith(".weight") or name.endswith(".bias"):
        name = name.rsplit(".", 1)[0]
    return name


def _format_range(start: int, end: int) -> str:
    return f"{start}-{end}" if start != end else f"{start}"


def format_index_ranges(indices: Iterable[int]) -> str:
    """Compress a list of ints into a 'a-b,c,d-e' range string."""
    items = sorted(indices)
    if not items:
        return ""
    ranges = []
    start = prev = items[0]
    for idx in items[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(_format_range(start, prev))
        start = prev = idx
    ranges.append(_format_range(start, prev))
    return ",".join(ranges)


def summarize_names(names: Iterable[str]) -> List[str]:
    """Collapse repeated layer indices, e.g. 'layers.0.x'..'layers.31.x' -> 'layers.0-31.x'."""
    grouped: Dict[str, set[int]] = {}
    for name in names:
        tokens = name.split(".")
        key_tokens: List[str] = []
        layer_idx = None
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token in LAYER_TOKENS and idx + 1 < len(tokens) and tokens[idx + 1].isdigit():
                layer_idx = int(tokens[idx + 1])
                key_tokens.extend([token, "{layer}"])
                idx += 2
                continue
            key_tokens.append(token)
            idx += 1
        key = ".".join(key_tokens)
        layers = grouped.setdefault(key, set())
        if "{layer}" in key and layer_idx is not None:
            layers.add(layer_idx)

    summaries = []
    for key in sorted(grouped.keys()):
        text = key
        if "{layer}" in text:
            text = text.replace("{layer}", format_index_ranges(grouped[key]))
        summaries.append(text)
    return summaries


def summarize_grouping(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    groups: Optional[set[str]] = None,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Walk all 2D params and bucket each LoRA matrix into its group; return (grouped, ungrouped)."""
    selected = set(groups or SVD_GROUP_ORDER)
    grouped: Dict[str, List[str]] = {group: [] for group in selected}
    ungrouped: List[str] = []
    for name, param in named_parameters:
        if param.ndim != 2:
            continue
        simple = simplify_param_name(name)
        lora_info = split_lora_param_name(name)
        if lora_info is not None:
            base_prefix, part, _adapter = lora_info
            base_group = classify_svd_group(base_prefix.rstrip("."))
            if base_group is None:
                ungrouped.append(simple)
                continue
            lora_group = f"{base_group}_{part}"
            if lora_group in grouped:
                grouped[lora_group].append(simple)
            continue
        # Non-LoRA 2D param: silently skip if it's a tracked LoRA base layer
        # (e.g., q_proj.base_layer.weight); only flag truly unknown params.
        if classify_svd_group(name) is not None:
            continue
        ungrouped.append(simple)

    grouped_summary = {group: summarize_names(names) for group, names in grouped.items()}
    ungrouped_summary = summarize_names(ungrouped)
    return grouped_summary, ungrouped_summary


def classify_svd_group(name: str) -> Optional[str]:
    """Map a parameter name to one of {AttnQO, AttnKV, Dense} (LLaMA-style); None otherwise."""
    lower = name.lower()
    if ".self_attn.q_proj" in lower or ".self_attn.o_proj" in lower:
        return "AttnQO"
    if (
        ".self_attn.k_proj" in lower
        or ".self_attn.v_proj" in lower
    ):
        return "AttnKV"
    if ".mlp.gate_proj" in lower or ".mlp.up_proj" in lower or ".mlp.down_proj" in lower:
        return "Dense"
    return None


def collect_svd_metrics(
    model,
    groups: Optional[set[str]] = None,
) -> Tuple[
    Dict[str, Dict[str, Optional[float]]],
    Dict[str, Dict[str, Optional[float]]],
    Dict[str, int],
    Dict[str, Dict[str, int]],
]:
    """Compute svd_entropy + stable_rank per LoRA matrix, aggregated by group.

    Returns (svd_entropy_groups, stable_rank_groups, skipped_errors, group_counts).
    """
    selected_groups = set(groups or [])
    group_entropies: Dict[str, List[float]] = {}
    group_stable_ranks: Dict[str, List[float]] = {}
    skipped: Dict[str, int] = {}
    group_totals: Dict[str, int] = {group: 0 for group in selected_groups}
    group_selected: Dict[str, int] = {group: 0 for group in selected_groups}

    named_params = list(model.named_parameters())

    lora_group_specs: Dict[str, set[str]] = {}
    for group in selected_groups:
        parsed = parse_lora_group_name(group)
        if parsed is None:
            raise ValueError(
                f"Group {group!r} is not a LoRA group name (expected '<Base>_A' or '<Base>_B')."
            )
        base_group, suffix = parsed
        lora_group_specs.setdefault(base_group, set()).add(suffix)

    lora_a: Dict[Tuple[str, str], Tuple[str, torch.nn.Parameter]] = {}
    lora_b: Dict[Tuple[str, str], Tuple[str, torch.nn.Parameter]] = {}
    for name, param in named_params:
        if param.ndim != 2:
            continue
        lora_info = split_lora_param_name(name)
        if lora_info is None:
            continue
        base_prefix, part, adapter = lora_info
        key = (base_prefix, adapter)
        if part == "A":
            lora_a[key] = (name, param)
        else:
            lora_b[key] = (name, param)

    keys = sorted(set(lora_a.keys()) | set(lora_b.keys()))
    progress = None
    if tqdm is not None and keys:
        progress = tqdm(
            total=len(keys),
            desc="SVD-LoRA",
            unit="mat",
            dynamic_ncols=True,
            ascii=True,
            file=sys.stdout,
        )
    for base_prefix, adapter in keys:
        base_group = classify_svd_group(base_prefix.rstrip("."))
        if base_group is None or base_group not in lora_group_specs:
            if progress is not None:
                progress.update(1)
            continue
        suffixes = lora_group_specs[base_group]
        lora_a_entry = lora_a.get((base_prefix, adapter))
        lora_b_entry = lora_b.get((base_prefix, adapter))
        base_key = simplify_param_name(base_prefix.rstrip("."))
        if "A" in suffixes and lora_a_entry is not None:
            group_name = f"{base_group}_A"
            group_totals[group_name] = group_totals.get(group_name, 0) + 1
            group_selected[group_name] = group_selected.get(group_name, 0) + 1
            _svd_metrics_for_matrix(
                base_key,
                group_name,
                lora_a_entry[1],
                group_entropies,
                group_stable_ranks,
                skipped,
            )
        if "B" in suffixes and lora_b_entry is not None:
            group_name = f"{base_group}_B"
            group_totals[group_name] = group_totals.get(group_name, 0) + 1
            group_selected[group_name] = group_selected.get(group_name, 0) + 1
            _svd_metrics_for_matrix(
                base_key,
                group_name,
                lora_b_entry[1],
                group_entropies,
                group_stable_ranks,
                skipped,
            )
        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()

    svd_entropy = {
        group: {"mean": mean_or_none(values), "count": len(values)}
        for group, values in group_entropies.items()
    }
    stable_rank_groups = {
        group: {"mean": mean_or_none(values), "count": len(values)}
        for group, values in group_stable_ranks.items()
    }
    group_counts = {
        "total": group_totals,
        "selected": group_selected,
    }

    return svd_entropy, stable_rank_groups, skipped, group_counts
