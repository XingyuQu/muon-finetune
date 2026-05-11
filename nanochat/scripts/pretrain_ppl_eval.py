"""
Evaluate model perplexity on pretraining data to measure forgetting.

Usage:
    # Evaluate a checkpoint directory
    torchrun --standalone --nproc_per_node=8 -m scripts.pretrain_ppl_eval \
        --ckpt-dir=/path/to/checkpoint

    # Evaluate base pretrained model
    torchrun --standalone --nproc_per_node=8 -m scripts.pretrain_ppl_eval \
        --model_tag=d20_muon

    # Use fewer samples for quick testing
    torchrun --standalone --nproc_per_node=8 -m scripts.pretrain_ppl_eval \
        --ckpt-dir=/path/to/checkpoint --num_samples=100
"""

import os
import math
import json
import random
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist
import pyarrow.parquet as pq

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model, find_last_step, build_model


def load_pretrain_data(base_dir, num_shards=5, seed=42):
    """Load text from pretraining data shards."""
    data_dir = os.path.join(base_dir, "base_data")

    # Get all shard files
    shard_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])

    # Randomly sample shards
    rng = random.Random(seed)
    selected_shards = rng.sample(shard_files, min(num_shards, len(shard_files)))

    texts = []
    for shard_file in selected_shards:
        shard_path = os.path.join(data_dir, shard_file)
        table = pq.read_table(shard_path)
        df = table.to_pandas()
        texts.extend(df['text'].tolist())

    return texts


def tokenize_and_chunk(texts, tokenizer, max_seq_len, num_samples, seed=42):
    """Tokenize texts and create fixed-length chunks for evaluation."""
    rng = random.Random(seed)
    rng.shuffle(texts)

    # Get BOS token id (used as document delimiter in nanochat)
    bos_token_id = tokenizer.bos_token_id

    # Concatenate all texts with BOS tokens as delimiters
    all_tokens = []
    for text in texts:
        all_tokens.append(bos_token_id)  # Document start
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

        # Stop if we have enough tokens
        if len(all_tokens) >= num_samples * max_seq_len * 2:
            break

    # Create chunks of max_seq_len
    chunks = []
    for i in range(0, len(all_tokens) - max_seq_len, max_seq_len):
        chunk = all_tokens[i:i + max_seq_len]
        chunks.append(chunk)
        if len(chunks) >= num_samples:
            break

    return chunks


@torch.no_grad()
def evaluate_ppl(model, chunks, device, batch_size=8):
    """Evaluate perplexity on the chunks."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    # Process in batches
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]

        # Convert to tensor
        input_ids = torch.tensor(batch_chunks, dtype=torch.long, device=device)

        # Forward pass
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)

        # Calculate loss (next token prediction)
        # logits: (B, T, V), targets: (B, T)
        # We predict token[i+1] from token[i], so shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Calculate cross entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum'
        )

        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return avg_loss, ppl, total_tokens


def main():
    """CLI entry: load a base / mid / sft / direct-path checkpoint, sample pretraining shards, and report PPL."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, default=None,
                        help='Direct path to checkpoint directory')
    parser.add_argument('--model_tag', type=str, default=None,
                        help='Model tag for base checkpoint (e.g., d20_muon)')
    parser.add_argument('--source', type=str, default='base', choices=['base', 'mid', 'sft'],
                        help='Source of the model: base|mid|sft')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of sequence chunks to evaluate')
    parser.add_argument('--num_shards', type=int, default=5,
                        help='Number of data shards to sample from')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    args = parser.parse_args()

    # Initialize distributed
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = (ddp_rank == 0)

    base_dir = get_base_dir()

    # Load model
    if args.ckpt_dir is not None:
        ckpt_dir = os.path.abspath(args.ckpt_dir)
        step = find_last_step(ckpt_dir)
        print0(f"Loading model from {ckpt_dir} at step {step}")
        model, tokenizer, meta = build_model(ckpt_dir, step, device, phase="eval")
        model_name = os.path.basename(ckpt_dir)
    else:
        print0(f"Loading model from {args.source}/{args.model_tag}")
        model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
        model_name = f"{args.source}_{args.model_tag}"

    # Get max sequence length from model
    max_seq_len = model.max_seq_len if hasattr(model, 'max_seq_len') else 1024
    print0(f"Using max_seq_len={max_seq_len}")

    # Load pretraining data
    print0(f"Loading pretraining data from {args.num_shards} shards...")
    texts = load_pretrain_data(base_dir, num_shards=args.num_shards, seed=args.seed)
    print0(f"Loaded {len(texts)} text samples")

    # Tokenize and chunk
    print0(f"Tokenizing and creating {args.num_samples} chunks...")
    chunks = tokenize_and_chunk(texts, tokenizer, max_seq_len, args.num_samples, seed=args.seed)
    print0(f"Created {len(chunks)} chunks of length {max_seq_len}")

    # Distribute chunks across GPUs if using DDP
    if ddp:
        chunks_per_gpu = len(chunks) // ddp_world_size
        start_idx = ddp_rank * chunks_per_gpu
        end_idx = start_idx + chunks_per_gpu if ddp_rank < ddp_world_size - 1 else len(chunks)
        local_chunks = chunks[start_idx:end_idx]
        print0(f"GPU {ddp_rank}: evaluating {len(local_chunks)} chunks")
    else:
        local_chunks = chunks

    # Evaluate
    print0("Evaluating perplexity...")
    avg_loss, ppl, total_tokens = evaluate_ppl(model, local_chunks, device, batch_size=args.batch_size)

    # Aggregate results across GPUs if using DDP
    if ddp:
        # Convert to tensors for all_reduce
        loss_tensor = torch.tensor([avg_loss * total_tokens], device=device)
        tokens_tensor = torch.tensor([total_tokens], device=device)

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)

        total_loss = loss_tensor.item()
        total_tokens = int(tokens_tensor.item())
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)

    # Print results
    print0("=" * 60)
    print0(f"Model: {model_name}")
    print0(f"Pretrain Data PPL: {ppl:.4f}")
    print0(f"Avg Loss: {avg_loss:.6f}")
    print0(f"Total Tokens Evaluated: {total_tokens:,}")
    print0("=" * 60)

    # Save results
    if master_process:
        result = {
            "model": model_name,
            "pretrain_ppl": ppl,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
            "num_samples": args.num_samples,
            "num_shards": args.num_shards,
            "seed": args.seed
        }

        # Save to checkpoint directory if available
        if args.ckpt_dir is not None:
            output_path = os.path.join(ckpt_dir, "pretrain_ppl.json")
        else:
            output_dir = os.path.join(base_dir, "pretrain_ppl_eval")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{model_name}.json")

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print0(f"Results saved to: {output_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
