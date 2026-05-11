from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def _resolve_adapter_dir(checkpoint: str) -> Optional[Path]:
    """Find the dir containing adapter_config.json (try `checkpoint` and one nested level)."""
    ckpt_path = Path(checkpoint)
    candidates = [ckpt_path, ckpt_path / ckpt_path.name]
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file():
            return candidate
    return None


def _resolve_base_model_name(adapter_dir: Path) -> str:
    """Read base_model_name_or_path from adapter_config.json (resolve relative paths)."""
    config_path = adapter_dir / "adapter_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = payload.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError(f"Missing base_model_name_or_path in {config_path}")
    base_path = Path(base_model)
    if not base_path.is_absolute():
        candidate = adapter_dir / base_model
        if candidate.exists():
            return str(candidate)
    return base_model


def _load_causal_lm(model_name: str, dtype: torch.dtype):
    """Load a CausalLM, falling back to torch_dtype= for older transformers versions."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )


def load_tokenizer_and_model(checkpoint: str, dtype: torch.dtype, device: torch.device):
    """Load base CausalLM + LoRA adapter from `checkpoint`; eval mode, KV cache off."""
    adapter_dir = _resolve_adapter_dir(checkpoint)
    if adapter_dir is None:
        raise RuntimeError(
            f"No LoRA adapter_config.json found under {checkpoint}; "
            "this script only supports LoRA checkpoints."
        )
    if PeftModel is None:
        raise RuntimeError("peft is required to load LoRA adapters.")
    base_model = _resolve_base_model_name(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    model = _load_causal_lm(base_model, dtype=dtype)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    return tokenizer, model
