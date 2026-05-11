from typing import Sequence

import torch

from .muon_builder import _build_muon_optimizer


def optimizer_tag(args) -> str:
    """Return a short slug for the optimizer choice (``muon``, ``muon_pe``, ``adamw``)."""
    opt = args.optimizer
    if opt == "muon" and args.ns_using_pe:
        return "muon_pe"
    return opt


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, Sequence):
        return list(x)
    return [x]


def apply_grad_clipping(
    model: torch.nn.Module,
    max_grad_norm: float,
) -> None:
    """Clip ``model`` gradients to ``max_grad_norm`` (no-op if not positive)."""
    if max_grad_norm is None or max_grad_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def build_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    """Construct the optimizer (Muon with AdamW fallback group, or plain AdamW)."""
    opt_name = args.optimizer.lower()

    if opt_name == "muon":
        return _build_muon_optimizer(model, args)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.wd,
    )
