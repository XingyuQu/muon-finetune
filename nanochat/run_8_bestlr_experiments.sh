#!/usr/bin/env bash
# Run the 8 best-LR WikiText fine-tuning experiments (4 modes x 2 pretrains)
# at the LRs reported in the paper. Each run writes an eval_log + meta only
# (--save_weights=False), which is what the trajectory-plotting analysis
# scripts consume; no model weights are persisted.

cd "$(dirname "$0")"

# --- Muon pretrain (d20_muon) ---

echo "=== [1/8] Muon pretrain + Full-Muon, LR=0.9 ==="
torchrun --standalone --nproc_per_node=8 -m scripts.wikitext_finetune \
    --mode=full-muon --model_tag=d20_muon --seed=0 \
    --matrix_lr=0.9 --num_epochs=1 --eval_every=15 \
    --save_weights=False --wandb_name=auto

echo "=== [2/8] Muon pretrain + Full-Adam, LR=0.009 ==="
torchrun --standalone --nproc_per_node=8 -m scripts.wikitext_finetune \
    --mode=full-adam --model_tag=d20_muon --seed=0 \
    --adam_lr=0.009 --num_epochs=1 --eval_every=15 \
    --save_weights=False --wandb_name=auto

echo "=== [3/8] Muon pretrain + LoRA-Muon, LR=0.9 ==="
torchrun --standalone --nproc_per_node=8 -m scripts.wikitext_finetune \
    --mode=lora-muon --model_tag=d20_muon --seed=0 \
    --lora_muon_lr=0.9 --num_epochs=1 --eval_every=15 \
    --save_weights=False --wandb_name=auto

echo "=== [4/8] Muon pretrain + LoRA-Adam, LR=0.1 ==="
torchrun --standalone --nproc_per_node=8 -m scripts.wikitext_finetune \
    --mode=lora-adam --model_tag=d20_muon --seed=0 \
    --lora_adam_lr=0.1 --num_epochs=1 --eval_every=15 \
    --save_weights=False --wandb_name=auto

# --- Adam pretrain (d20_adam_lr0.001) ---

echo "=== [5/8] Adam pretrain + Full-Adam, LR=0.03 ==="
torchrun --standalone --nproc_per_node=8 -m scripts.wikitext_finetune \
    --mode=full-adam --model_tag=d20_adam_lr0.001 --seed=0 \
    --adam_lr=0.03 --num_epochs=1 --eval_every=15 \
    --save_weights=False --wandb_name=auto

echo "=== [6/8] Adam pretrain + Full-Muon, LR=0.5 ==="
torchrun --standalone --nproc_per_node=8 -m scripts.wikitext_finetune \
    --mode=full-muon --model_tag=d20_adam_lr0.001 --seed=0 \
    --matrix_lr=0.5 --num_epochs=1 --eval_every=15 \
    --save_weights=False --wandb_name=auto

echo "=== [7/8] Adam pretrain + LoRA-Adam, LR=0.3 ==="
torchrun --standalone --nproc_per_node=8 -m scripts.wikitext_finetune \
    --mode=lora-adam --model_tag=d20_adam_lr0.001 --seed=0 \
    --lora_adam_lr=0.3 --num_epochs=1 --eval_every=15 \
    --save_weights=False --wandb_name=auto

echo "=== [8/8] Adam pretrain + LoRA-Muon, LR=0.7 ==="
torchrun --standalone --nproc_per_node=8 -m scripts.wikitext_finetune \
    --mode=lora-muon --model_tag=d20_adam_lr0.001 --seed=0 \
    --lora_muon_lr=0.7 --num_epochs=1 --eval_every=15 \
    --save_weights=False --wandb_name=auto

echo "=== All 8 experiments done ==="
