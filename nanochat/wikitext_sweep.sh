#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# WikiText-2 Fine-tuning Sweep Script
# =============================================================================
# Usage:
#   ./wikitext_sweep.sh                    # Run all experiments
#   ./wikitext_sweep.sh --dry-run          # Print commands without executing
#   DRY_RUN=1 ./wikitext_sweep.sh          # Same as above
#
# Environment variables:
#   NPROC       - Number of GPUs (default: 8)
#   NUM_EPOCHS  - Number of epochs (default: 3)
#   EVAL_EVERY  - Evaluation frequency (default: 10)
#   DRY_RUN     - Set to 1 for dry run
#   NO_WANDB    - Set to 1 to disable wandb
# =============================================================================

# Configuration
NPROC=${NPROC:-8}
NUM_EPOCHS=${NUM_EPOCHS:-3}
EVAL_EVERY=${EVAL_EVERY:-10}
DRY_RUN=${DRY_RUN:-0}
NO_WANDB=${NO_WANDB:-0}

# Parse command line args
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-wandb)
            NO_WANDB=1
            shift
            ;;
    esac
done

# =============================================================================
# Sweep Configuration - EDIT THIS SECTION
# =============================================================================

# Models to sweep over (paper uses d20_muon and d20_adam_lr0.001)
MODELS=(
    "d20_muon"
    "d20_adam_lr0.001"
)

# Modes (optimizers): full-* and lora-* are both supported by scripts.wikitext_finetune.
# Paper Table 11 sweeps all four; comment out any you don't need.
MODES=(
    "full-adam"
    "full-muon"
    "lora-adam"
    "lora-muon"
)

# Seeds
SEEDS=(0 1 2)

# Learning rates for each mode (override via env var, e.g. LORA_MUON_LRS="0.5 0.7 0.9")
# Defaults below are minimal starter points; the paper's sweep used a wider range
# (1e-5 to 9e-1, see appendix). Best LRs found in the paper:
#   full-adam: 0.009 (Muon pre), 0.03 (Adam pre)
#   full-muon: 0.9   (Muon pre), 0.5  (Adam pre)
#   lora-adam: 0.1   (Muon pre), 0.3  (Adam pre)
#   lora-muon: 0.9   (Muon pre), 0.7  (Adam pre)
ADAM_LRS=(${ADAM_LRS:-1e-5 5e-5 1e-4 1e-3 9e-3 3e-2})
MUON_LRS=(${MUON_LRS:-0.005 0.01 0.05 0.1 0.5 0.9})
LORA_ADAM_LRS=(${LORA_ADAM_LRS:-0.05 0.1 0.3 0.5})
LORA_MUON_LRS=(${LORA_MUON_LRS:-0.3 0.5 0.7 0.9})

# =============================================================================
# Helper functions
# =============================================================================

run_experiment() {
    local model_tag=$1
    local mode=$2
    local lr=$3
    local seed=$4

    # Determine LR parameter name
    local lr_param
    case "$mode" in
        full-adam)  lr_param="--adam_lr=$lr" ;;
        full-muon)  lr_param="--matrix_lr=$lr" ;;
        lora-adam)  lr_param="--lora_adam_lr=$lr" ;;
        lora-muon)  lr_param="--lora_muon_lr=$lr" ;;
        *)          echo "Unknown mode: $mode" >&2; exit 1 ;;
    esac

    # Wandb name
    local wandb_name
    if [[ "$NO_WANDB" == "1" ]]; then
        wandb_name="dummy"
    else
        wandb_name="wikitext_${model_tag}_${mode}_lr${lr}_seed${seed}"
    fi

    # Build command
    local cmd="torchrun --standalone --nproc_per_node=$NPROC -m scripts.wikitext_finetune"
    cmd="$cmd --mode=$mode"
    cmd="$cmd --model_tag=$model_tag"
    cmd="$cmd --seed=$seed"
    cmd="$cmd $lr_param"
    cmd="$cmd --num_epochs=$NUM_EPOCHS"
    cmd="$cmd --eval_every=$EVAL_EVERY"
    cmd="$cmd --wandb_name=$wandb_name"

    echo ""
    echo "========================================================================"
    echo ">>> $model_tag + $mode + lr=$lr + seed=$seed"
    echo "========================================================================"
    echo "$cmd"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN - skipping execution]"
    else
        $cmd
        echo "✅ Done: $model_tag + $mode + lr=$lr + seed=$seed"
    fi
}

# =============================================================================
# Main sweep loop
# =============================================================================

echo "=========================================="
echo "WikiText-2 Fine-tuning Sweep"
echo "=========================================="
echo "Models: ${MODELS[*]}"
echo "Modes: ${MODES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Adam LRs:      ${ADAM_LRS[*]}"
echo "Muon LRs:      ${MUON_LRS[*]}"
echo "LoRA-Adam LRs: ${LORA_ADAM_LRS[*]}"
echo "LoRA-Muon LRs: ${LORA_MUON_LRS[*]}"
echo "Num epochs: $NUM_EPOCHS"
echo "Eval every: $EVAL_EVERY"
echo "GPUs: $NPROC"
echo "Dry run: $DRY_RUN"
echo "=========================================="

# Count total experiments
total=0
for model in "${MODELS[@]}"; do
    for mode in "${MODES[@]}"; do
        case "$mode" in
            full-adam)  lrs=("${ADAM_LRS[@]}") ;;
            full-muon)  lrs=("${MUON_LRS[@]}") ;;
            lora-adam)  lrs=("${LORA_ADAM_LRS[@]}") ;;
            lora-muon)  lrs=("${LORA_MUON_LRS[@]}") ;;
            *)          echo "Unknown mode: $mode" >&2; exit 1 ;;
        esac
        for lr in "${lrs[@]}"; do
            for seed in "${SEEDS[@]}"; do
                ((total++))
            done
        done
    done
done
echo "Total experiments: $total"
echo ""

# Run experiments
count=0
for model in "${MODELS[@]}"; do
    for mode in "${MODES[@]}"; do
        # Select LRs based on mode
        case "$mode" in
            full-adam)  lrs=("${ADAM_LRS[@]}") ;;
            full-muon)  lrs=("${MUON_LRS[@]}") ;;
            lora-adam)  lrs=("${LORA_ADAM_LRS[@]}") ;;
            lora-muon)  lrs=("${LORA_MUON_LRS[@]}") ;;
            *)          echo "Unknown mode: $mode" >&2; exit 1 ;;
        esac

        for lr in "${lrs[@]}"; do
            for seed in "${SEEDS[@]}"; do
                ((count++))
                echo ""
                echo "[$count/$total]"
                run_experiment "$model" "$mode" "$lr" "$seed"
            done
        done
    done
done

echo ""
echo "=========================================="
echo "🎉 Sweep complete! ($count/$total experiments)"
echo "=========================================="
