from typing import Iterable, Optional

import torch


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    """Arithmetic mean of `values`, or None if empty."""
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def default_dtype_for_device(device: str) -> str:
    """Pick a sensible default dtype: bf16 if CUDA + bf16 supported, else fp16/fp32."""
    if device.startswith("cuda"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return "fp32"


def parse_dtype(dtype: str) -> torch.dtype:
    """Convert a dtype tag ('fp32'|'fp16'|'bf16') to the corresponding torch.dtype."""
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")
