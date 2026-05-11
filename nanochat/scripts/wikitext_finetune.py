"""
Fine-tune nanochat on WikiText-2 to compare optimizers and measure forgetting.

Supports four modes:
- full-adam: Full fine-tuning with AdamW
- full-muon: Full fine-tuning with Muon
- lora-adam: LoRA fine-tuning with AdamW
- lora-muon: LoRA fine-tuning with Muon

Run on one GPU:
    python -m scripts.wikitext_finetune --mode=full-adam

Or torchrun for multi-GPU:
    torchrun --standalone --nproc_per_node=8 -m scripts.wikitext_finetune --mode=full-muon
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import random
from itertools import chain

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from datasets import load_dataset

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

# -----------------------------------------------------------------------------
# Minimal LoRA implementation: wraps an nn.Linear with a frozen base weight
# plus trainable low-rank factors A (r x in) and B (out x r). Forward adds
# (B @ A) @ x scaled by alpha/r; merged_weight() materialises the merged
# matrix for checkpointing without LoRA-specific keys.

class LoRALinear(nn.Linear):
    """nn.Linear wrapping a frozen base weight + trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        assert base.bias is None, "nanochat Linear layers are bias-free"
        super().__init__(base.in_features, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        with torch.no_grad():
            self.weight.copy_(base.weight)
        self.weight.requires_grad_(False)

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = (self.alpha / self.r) if self.r > 0 else 0.0
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
            self.lora_B = nn.Parameter(torch.empty(base.out_features, self.r, device=base.weight.device, dtype=base.weight.dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x):
        """Compute base @ x plus the scaled low-rank update (skipped when r == 0)."""
        y = F.linear(x, self.weight)
        if self.r > 0:
            x = self.dropout(x)
            y = y + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return y

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """Return the materialised weight (base + scaled B @ A) cast back to the base dtype."""
        if self.r <= 0:
            return self.weight
        delta = (self.lora_B.float() @ self.lora_A.float()) * float(self.scaling)
        return (self.weight.float() + delta).to(dtype=self.weight.dtype)

def inject_lora(model: nn.Module, r: int, alpha: float, dropout: float, target_names: set[str]) -> int:
    """Replace every nn.Linear child whose attribute name is in target_names with a LoRALinear; return the count."""
    replaced = 0
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    return replaced

def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Build a state_dict with LoRA factors folded back into base weights, so the result loads as a plain GPT."""
    state = {k: v for k, v in model.state_dict().items() if ".lora_A" not in k and ".lora_B" not in k}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.weight"] = module.merged_weight()
    return state

# -----------------------------------------------------------------------------
# Hyperparameters
wandb_name = "auto"  # wandb run name ("dummy" = no wandb, "auto" = auto-generate name)
seed = 42
# Model source
source = "base"  # base|mid
model_tag = "d20_adam"  # which checkpoint to load
step = None  # step to load (None = latest)
# Compute
device_type = ""  # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
device_batch_size = 4
# Training
mode = "full-adam"  # full-adam|full-muon|lora-adam|lora-muon
num_epochs = 1
num_iterations = -1  # override (-1 = use num_epochs)
target_examples_per_step = 32
block_size = 1024  # sequence length
# Learning rates (matching nanochat defaults)
# For Muon mode: matrix_lr for transformer layers, embedding_lr/unembedding_lr for embed/lm_head
matrix_lr = 0.02
embedding_lr = 0.2
unembedding_lr = 0.004
# For Adam mode: single lr for all params (fine-tuning typically uses smaller lr)
adam_lr = 5e-5
# For LoRA mode
lora_adam_lr = 2e-4
lora_muon_lr = 2e-4
weight_decay = 0.0
init_lr_frac = 0.02  # Initial LR fraction (like nanochat: start at 2% of target LR)
# LoRA settings
lora_r = 8
lora_alpha = 16.0
lora_dropout = 0.0
lora_targets = "c_q,c_k,c_v,c_proj,c_fc"
# Evaluation
eval_every = 50
output_dir = None
save_weights = True
num_saves = 0  # number of equally-spaced intermediate checkpoints (0 = no saves, 5 = save at 20%,40%,60%,80%,100%)
# CLI override
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))] + ["output_dir", "step"]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Random seed
seed_rank = int(seed) + ddp_rank
random.seed(seed_rank)
torch.manual_seed(seed_rank)
if device_type == "cuda":
    torch.cuda.manual_seed_all(seed_rank)

# Parse mode
use_lora = mode.startswith("lora")
use_muon = mode.endswith("muon")
print0(f"Mode: {mode} (LoRA={use_lora}, Muon={use_muon})")

# wandb logging init
if wandb_name == "auto":
    # Include LR and epochs in run name to distinguish different experiments
    if mode == "full-muon":
        lr_for_name = matrix_lr
    elif mode == "full-adam":
        lr_for_name = adam_lr
    elif mode == "lora-muon":
        lr_for_name = lora_muon_lr
    elif mode == "lora-adam":
        lr_for_name = lora_adam_lr
    else:
        lr_for_name = adam_lr
    # Include rank in name for LoRA experiments
    if mode.startswith("lora"):
        run_name = f"wikitext_{mode}_{model_tag}_r{lora_r}_lr{lr_for_name}_ep{num_epochs}_seed{seed}"
    else:
        run_name = f"wikitext_{mode}_{model_tag}_lr{lr_for_name}_ep{num_epochs}_seed{seed}"
else:
    run_name = wandb_name
use_dummy_wandb = wandb_name == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat_wikitext", name=run_name, config=user_config, save_code=True)

# Load model
print0(f"Loading model from {source}/{model_tag}...")
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
model_config = model.config

num_params = sum(p.numel() for p in model.parameters())
print0(f"Model has {num_params/1e6:.2f}M parameters")

# Setup training mode
if use_lora:
    # Freeze base model, inject LoRA
    for p in model.parameters():
        p.requires_grad_(False)
    target_names = {s.strip() for s in lora_targets.split(",") if s.strip()}
    num_replaced = inject_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target_names=target_names)
    print0(f"Injected LoRA into {num_replaced} layers")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print0(f"Trainable parameters: {num_trainable/1e6:.2f}M ({100*num_trainable/num_params:.2f}%)")
else:
    # Full fine-tuning
    for p in model.parameters():
        p.requires_grad_(True)
    trainable_params = list(model.parameters())
    print0("Full fine-tuning: all parameters trainable")

# -----------------------------------------------------------------------------
# Load WikiText-2
print0("Loading WikiText-2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Tokenize using nanochat tokenizer
def tokenize_function(examples):
    """HF datasets map fn: encode each non-empty `text` line to a list of token ids."""
    all_ids = []
    for text in examples["text"]:
        if text.strip():  # skip empty lines
            ids = tokenizer.encode(text)
            all_ids.append(ids)
    return {"input_ids": all_ids}

print0("Tokenizing dataset...")
tokenized_train = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing train",
)
tokenized_val = dataset["validation"].map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing validation",
)

# Group into blocks
def group_texts(examples):
    """HF datasets map fn: flatten token lists and chunk into fixed-length `block_size` blocks (drop remainder)."""
    # Flatten all input_ids
    all_ids = []
    for ids in examples["input_ids"]:
        all_ids.extend(ids)

    # Drop remainder
    total_length = (len(all_ids) // block_size) * block_size
    all_ids = all_ids[:total_length]

    # Split into chunks
    result = {
        "input_ids": [all_ids[i:i + block_size] for i in range(0, total_length, block_size)]
    }
    return result

print0(f"Grouping texts into blocks of {block_size}...")
train_data = tokenized_train.map(
    group_texts,
    batched=True,
    desc=f"Grouping train into {block_size}-token blocks",
)
val_data = tokenized_val.map(
    group_texts,
    batched=True,
    desc=f"Grouping val into {block_size}-token blocks",
)

print0(f"=> Train blocks: {len(train_data)}, Validation blocks: {len(val_data)}")

# -----------------------------------------------------------------------------
# Data generator
def data_generator(dataset, batch_size, shuffle=True):
    """Infinite (input, target) batch generator with DDP-rank striping; yields shifted causal-LM tensors."""
    def collate_batch(batch):
        input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long, device=device)
        # For causal LM: inputs = ids[:-1], targets = ids[1:]
        inputs = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()
        return inputs, targets

    indices = list(range(len(dataset)))
    batch = []

    while True:
        if shuffle:
            random.shuffle(indices)
        for i in indices:
            if i % ddp_world_size != ddp_rank:
                continue
            batch.append(dataset[i])
            if len(batch) == batch_size:
                yield collate_batch(batch)
                batch = []

# Calculate training parameters
examples_per_step = device_batch_size * ddp_world_size
if examples_per_step >= target_examples_per_step:
    grad_accum_steps = 1
    effective_batch_size = examples_per_step
else:
    assert target_examples_per_step % examples_per_step == 0
    grad_accum_steps = target_examples_per_step // examples_per_step
    effective_batch_size = target_examples_per_step

if num_iterations == -1:
    num_iterations = (len(train_data) // effective_batch_size) * num_epochs

print0(f"Effective batch size: {effective_batch_size}")
print0(f"Grad accum steps: {grad_accum_steps}")
print0(f"Number of iterations: {num_iterations}")

# Compute intermediate save steps from num_saves
save_steps = set()
if num_saves > 0:
    for i in range(1, num_saves + 1):
        frac = i / num_saves
        save_step = round(frac * (num_iterations - 1))
        save_steps.add(save_step)
    print0(f"Will save checkpoints at steps: {sorted(save_steps)} (num_saves={num_saves})")

train_loader = data_generator(train_data, batch_size=device_batch_size, shuffle=True)

# -----------------------------------------------------------------------------
# Optimizer setup
# Matching nanochat's setup_optimizers() exactly:
# - matrix_params (transformer.h) -> Muon (in Muon mode) or AdamW (in Adam mode)
# - embedding_params (transformer.wte) -> AdamW with embedding_lr
# - lm_head_params (lm_head) -> AdamW with unembedding_lr
# DistMuon and DistAdamW handle gradient synchronization internally

from functools import partial

# Separate parameters into 3 groups exactly like nanochat
matrix_params = list(model.transformer.h.parameters())
embedding_params = list(model.transformer.wte.parameters())
lm_head_params = list(model.lm_head.parameters())

# For LoRA mode, only the LoRA params are trainable
if use_lora:
    matrix_params = [p for p in matrix_params if p.requires_grad]
    embedding_params = [p for p in embedding_params if p.requires_grad]
    lm_head_params = [p for p in lm_head_params if p.requires_grad]

print0(f"Parameter groups: matrix={len(matrix_params)}, embedding={len(embedding_params)}, lm_head={len(lm_head_params)}")

# LR scaling like nanochat: scale AdamW LRs by 1/sqrt(d_model/768)
model_dim = model.config.n_embd
dmodel_lr_scale = (model_dim / 768) ** -0.5
print0(f"LR scale for AdamW params: 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

# AdamW kwargs matching nanochat
adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)

optimizers = []

if use_muon:
    # Muon mode: Muon for matrix params, AdamW for embedding/lm_head
    # Exactly like nanochat's setup_optimizers()

    lr = lora_muon_lr if use_lora else matrix_lr
    MuonFactory = DistMuon if ddp else Muon
    AdamFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)

    # AdamW for embedding and lm_head (with scaled LRs)
    adam_groups = []
    if lm_head_params:
        adam_groups.append(dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale))
    if embedding_params:
        adam_groups.append(dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale))

    if adam_groups:
        adamw_optimizer = AdamFactory(adam_groups, **adamw_kwargs)
        optimizers.append(adamw_optimizer)
        print0(f"AdamW optimizer: lm_head lr={unembedding_lr * dmodel_lr_scale:.6f}, embedding lr={embedding_lr * dmodel_lr_scale:.6f}")

    # Muon for matrix params
    if matrix_params:
        muon_kwargs = dict(lr=lr, momentum=0.95)
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        optimizers.append(muon_optimizer)
        print0(f"Muon optimizer: {len(matrix_params)} params, lr={lr}")

else:
    # Adam mode: AdamW for all params
    # Use same LR for all params (typical fine-tuning approach)

    lr = lora_adam_lr if use_lora else adam_lr
    AdamFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)

    # Group all params with same LR
    adam_groups = []
    if matrix_params:
        adam_groups.append(dict(params=matrix_params, lr=lr))
    if lm_head_params:
        adam_groups.append(dict(params=lm_head_params, lr=lr))
    if embedding_params:
        adam_groups.append(dict(params=embedding_params, lr=lr))

    if adam_groups:
        adamw_optimizer = AdamFactory(adam_groups, **adamw_kwargs)
        optimizers.append(adamw_optimizer)
        print0(f"AdamW optimizer: lr={lr}")

# Set initial_lr for all param groups (matching nanochat: scale by init_lr_frac)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

# Learning rate scheduler (matching nanochat: simple linear decay to 0)
def get_lr_multiplier(it):
    """Linear LR decay: 1.0 at step 0 -> 0.0 at num_iterations."""
    return 1.0 - it / num_iterations

# -----------------------------------------------------------------------------
# Evaluation function
def evaluate_perplexity(val_ds, max_batches=50):
    """Evaluate perplexity on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    val_loader = data_generator(val_ds, batch_size=device_batch_size, shuffle=False)
    max_batches = min(max_batches, len(val_ds) // device_batch_size // ddp_world_size or 1)

    for _ in range(max_batches):
        inputs, targets = next(val_loader)

        with torch.no_grad(), autocast_ctx:
            loss = model(inputs, targets)

        total_loss += loss.item()
        num_batches += 1

    # Aggregate across ranks
    if ddp:
        loss_tensor = torch.tensor([total_loss], dtype=torch.float, device=device)
        batches_tensor = torch.tensor([num_batches], dtype=torch.long, device=device)

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(batches_tensor, op=dist.ReduceOp.SUM)

        total_loss = loss_tensor.item()
        num_batches = batches_tensor.item()

    model.train()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity

# -----------------------------------------------------------------------------
# Training loop
print0("Starting training...")
print0(f"Initial evaluation...")
init_loss, init_ppl = evaluate_perplexity(val_data)
print0(f"Initial | Val loss: {init_loss:.4f} | Perplexity: {init_ppl:.2f}")
wandb_run.log({"step": -1, "val_loss": init_loss, "perplexity": init_ppl})

step = 0
train_iter = iter(train_loader)
best_ppl = init_ppl
eval_log = [{"step": -1, "val_loss": init_loss, "perplexity": init_ppl}]

while True:
    last_step = (step == num_iterations - 1)

    # Evaluation
    if step % eval_every == 0 or last_step:
        val_loss, ppl = evaluate_perplexity(val_data)
        if ppl < best_ppl:
            best_ppl = ppl
        print0(f"Step {step:05d}/{num_iterations:05d} | Val loss: {val_loss:.4f} | Perplexity: {ppl:.2f} | best: {best_ppl:.2f}")
        eval_log.append({"step": step, "val_loss": val_loss, "perplexity": ppl})
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
            "perplexity": ppl,
        })

    if last_step:
        break

    # Training step (matching nanochat order: backward -> lr_schedule -> step -> zero_grad)
    model.train()

    # Compute gradient
    total_loss = 0.0
    for _ in range(grad_accum_steps):
        inputs, targets = next(train_iter)
        with autocast_ctx:
            loss = model(inputs, targets)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        total_loss += loss.item()

    # Learning rate scheduler (matching nanochat)
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # Step the optimizers
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    if step % 10 == 0:
        print0(f"Step {step:05d}/{num_iterations:05d} | Train loss: {train_loss.item():.4f} | lrm: {lrm:.4f}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss.item(),
    })
    step += 1

    # Intermediate checkpoint saving
    if master_process and save_weights and save_steps and step in save_steps and not (step == num_iterations - 1):
        if output_dir:
            checkpoint_dir_save = output_dir
        else:
            base_dir = get_base_dir()
            if mode == "full-muon":
                lr_str = f"lr{matrix_lr}"
            elif mode == "full-adam":
                lr_str = f"lr{adam_lr}"
            elif mode == "lora-muon":
                lr_str = f"lr{lora_muon_lr}"
            elif mode == "lora-adam":
                lr_str = f"lr{lora_adam_lr}"
            else:
                lr_str = f"lr{adam_lr}"
            if use_lora:
                checkpoint_dir_save = os.path.join(base_dir, "wikitext_checkpoints", f"{model_tag}_{mode}_r{lora_r}_{lr_str}_ep{num_epochs}_seed{seed}")
            else:
                checkpoint_dir_save = os.path.join(base_dir, "wikitext_checkpoints", f"{model_tag}_{mode}_{lr_str}_ep{num_epochs}_seed{seed}")

        inter_meta = {
            "step": step,
            "mode": mode,
            "model_tag": model_tag,
            "num_epochs": num_epochs,
            "seed": seed,
            "model_config": model_config.__dict__,
            "user_config": user_config,
        }
        if use_lora:
            inter_state = merged_state_dict(model)
        else:
            inter_state = model.state_dict()
        save_checkpoint(checkpoint_dir_save, step, inter_state, None, inter_meta)
        print0(f"Saved intermediate checkpoint at step {step} to {checkpoint_dir_save}")
        del inter_state

# Final evaluation
final_loss, final_ppl = evaluate_perplexity(val_data)
print0(f"Final | Val loss: {final_loss:.4f} | Perplexity: {final_ppl:.2f}")
print0(f"PPL change: {init_ppl:.2f} -> {final_ppl:.2f} ({100*(final_ppl-init_ppl)/init_ppl:+.1f}%)")

# Save results
if master_process:
    if output_dir:
        checkpoint_dir = output_dir
    else:
        base_dir = get_base_dir()
        # Include LR and epochs in path to avoid overwriting between different experiments
        if mode == "full-muon":
            lr_str = f"lr{matrix_lr}"
        elif mode == "full-adam":
            lr_str = f"lr{adam_lr}"
        elif mode == "lora-muon":
            lr_str = f"lr{lora_muon_lr}"
        elif mode == "lora-adam":
            lr_str = f"lr{lora_adam_lr}"
        else:
            lr_str = f"lr{adam_lr}"
        # Include rank in path for LoRA experiments
        if use_lora:
            checkpoint_dir = os.path.join(base_dir, "wikitext_checkpoints", f"{model_tag}_{mode}_r{lora_r}_{lr_str}_ep{num_epochs}_seed{seed}")
        else:
            checkpoint_dir = os.path.join(base_dir, "wikitext_checkpoints", f"{model_tag}_{mode}_{lr_str}_ep{num_epochs}_seed{seed}")

    meta_data = {
        "step": step,
        "mode": mode,
        "init_loss": init_loss,
        "init_ppl": init_ppl,
        "final_loss": final_loss,
        "final_ppl": final_ppl,
        "best_ppl": best_ppl,
        "model_tag": model_tag,
        "num_epochs": num_epochs,
        "seed": seed,
        "model_config": model_config.__dict__,
        "user_config": user_config,
        "eval_log": eval_log,
    }

    if save_weights:
        # Save both model and meta
        if use_lora:
            state_dict = merged_state_dict(model)
        else:
            state_dict = model.state_dict()
        save_checkpoint(checkpoint_dir, step, state_dict, None, meta_data)
        print0(f"Saved model and meta to {checkpoint_dir}")
    else:
        # Save only meta (no model weights)
        import json
        os.makedirs(checkpoint_dir, exist_ok=True)
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)
        print0(f"Saved meta only (no weights) to {meta_path}")

wandb_run.finish()
compute_cleanup()
