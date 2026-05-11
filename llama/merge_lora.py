#!/usr/bin/env python3
"""
Merge LoRA adapter weights into base model for evaluation.

Usage:
    python merge_lora.py --adapter_path results/lora_adamw_meta_math_lr2e-4_seed0
    python merge_lora.py --adapter_path results/lora_adamw_meta_math_lr2e-4_seed0 --output_path results/merged_model
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(adapter_path: str, output_path: str | None = None, dtype: str = "float16"):
    """Merge LoRA adapter into base model and save."""

    # Read adapter config to get base model
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config["base_model_name_or_path"]
    print(f"Base model: {base_model_name}")
    print(f"Adapter path: {adapter_path}")

    # Set output path
    if output_path is None:
        output_path = adapter_path + "_merged"
    print(f"Output path: {output_path}")

    # Load dtype
    torch_dtype = getattr(torch, dtype)
    print(f"Using dtype: {torch_dtype}")

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge and unload
    print("Merging weights...")
    merged_model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("Done!")
    return output_path


def main():
    """CLI entry point: parse args and call merge_lora()."""
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for merged model (default: {adapter_path}_merged)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: float16)",
    )

    args = parser.parse_args()
    merge_lora(args.adapter_path, args.output_path, args.dtype)


if __name__ == "__main__":
    main()
