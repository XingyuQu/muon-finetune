#!/usr/bin/env python3
"""Collect all experiment results and compute mean across seeds.

Reads every meta_*.json under $NANOCHAT_BASE_DIR/wikitext_checkpoints/
and prints (to stdout) one CSV-formatted row per (model_tag, mode, lr,
rank) with the mean and std of final_ppl across seeds, followed by a
Python-dict snippet useful for pasting into the plot scripts.

Usage:
    python scripts/analysis/collect_all_seeds.py
    python scripts/analysis/collect_all_seeds.py > wikitext_results.csv
"""

import os
import json
import re
from collections import defaultdict

from nanochat.common import get_base_dir

CHECKPOINT_DIR = os.path.join(get_base_dir(), "wikitext_checkpoints")

def parse_dir_name(dirname):
    """Parse directory name to extract model_tag, mode, lr, rank, seed."""
    # Patterns:
    # d20_muon_full-muon_lr0.9_ep1 (seed=0 implicit)
    # d20_muon_full-muon_lr0.9_ep1_seed0
    # d20_muon_lora-muon_lr0.9_ep1 (seed=0, rank=8 implicit)
    # d20_muon_lora-muon_r16_lr0.9_ep1_seed0

    # Skip directories without _ep1 suffix pattern
    if '_ep1' not in dirname:
        return None

    # Extract model_tag
    if dirname.startswith("d20_muon_"):
        model_tag = "d20_muon"
        rest = dirname[len("d20_muon_"):]
    elif dirname.startswith("d20_adam_lr0.001_"):
        model_tag = "d20_adam_lr0.001"
        rest = dirname[len("d20_adam_lr0.001_"):]
    else:
        return None

    # Extract mode
    mode_match = re.match(r'(full-muon|full-adam|lora-muon|lora-adam)', rest)
    if not mode_match:
        return None
    mode = mode_match.group(1)
    rest = rest[len(mode):]

    # Extract rank (for lora modes)
    rank = 8  # default
    rank_match = re.search(r'_r(\d+)', rest)
    if rank_match:
        rank = int(rank_match.group(1))

    # Extract lr
    lr_match = re.search(r'_lr([0-9.e-]+)', rest)
    if not lr_match:
        return None
    lr = float(lr_match.group(1))

    # Extract seed - if explicit seed in name, use it; otherwise seed=0
    # But skip dirs like "lr0.03_ep1" if "lr0.03_ep1_seed0" also exists (to avoid duplicates)
    seed = 0  # default
    seed_match = re.search(r'_seed(\d+)', rest)
    if seed_match:
        seed = int(seed_match.group(1))
    else:
        # This is an implicit seed=0 dir. We'll mark it specially and filter later
        pass

    return {
        'model_tag': model_tag,
        'mode': mode,
        'lr': lr,
        'rank': rank,
        'seed': seed,
        'has_explicit_seed': seed_match is not None,
        'dirname': dirname,
    }

def get_final_ppl(dirpath):
    """Get final_ppl from meta file."""
    meta_files = [f for f in os.listdir(dirpath) if f.startswith('meta_') and f.endswith('.json')]
    if not meta_files:
        return None
    meta_path = os.path.join(dirpath, meta_files[0])
    with open(meta_path) as f:
        data = json.load(f)
    return data.get('final_ppl')

# Collect all results
# Key: (model_tag, mode, lr, rank) -> list of (seed, ppl)
results = defaultdict(list)

# First pass: collect all parsed dirs
all_parsed = []
for dirname in os.listdir(CHECKPOINT_DIR):
    dirpath = os.path.join(CHECKPOINT_DIR, dirname)
    if not os.path.isdir(dirpath):
        continue

    parsed = parse_dir_name(dirname)
    if parsed is None:
        continue

    ppl = get_final_ppl(dirpath)
    if ppl is None:
        continue

    # Note: keeping diverged results (ppl > 100) for completeness

    parsed['ppl'] = ppl
    all_parsed.append(parsed)

# Check for duplicates: if both "lr0.03_ep1" and "lr0.03_ep1_seed0" exist, skip the implicit one
explicit_seed_keys = set()
for p in all_parsed:
    if p['has_explicit_seed']:
        key = (p['model_tag'], p['mode'], p['lr'], p['rank'], p['seed'])
        explicit_seed_keys.add(key)

for p in all_parsed:
    key = (p['model_tag'], p['mode'], p['lr'], p['rank'], p['seed'])
    # Skip implicit seed=0 if explicit seed=0 exists
    if not p['has_explicit_seed'] and key in explicit_seed_keys:
        continue

    result_key = (p['model_tag'], p['mode'], p['lr'], p['rank'])
    results[result_key].append((p['seed'], p['ppl']))

# Output one CSV row per (model_tag, mode, lr, rank) configuration.
print("model_tag,mode,lr,rank,n_seeds,mean_ppl,std_ppl,seeds")
for key in sorted(results.keys()):
    model_tag, mode, lr, rank = key
    seed_ppls = results[key]
    seeds = [s for s, p in seed_ppls]
    ppls = [p for s, p in seed_ppls]
    mean_ppl = sum(ppls) / len(ppls)
    if len(ppls) > 1:
        std_ppl = (sum((p - mean_ppl)**2 for p in ppls) / len(ppls)) ** 0.5
    else:
        std_ppl = 0
    print(f"{model_tag},{mode},{lr},{rank},{len(ppls)},{mean_ppl:.6f},{std_ppl:.6f},{seeds}")
