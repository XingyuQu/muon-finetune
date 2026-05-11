#!/bin/bash
# Batch evaluate all checkpoints in results/ folder
# Each dataset type is evaluated on its corresponding benchmark:
#   - meta_math: gsm8k_qa (Q:/A: format)
#   - wizard_lm: commonsense benchmarks (arc, hellaswag, piqa, etc.)
#   - codefeedback: humaneval (training-aligned Alpaca format)
#
# Model type auto-detection (by prefix):
#   lora_* -> LoRA model, use vLLM with --merge
#   full_* -> Full fine-tuned model, use vLLM direct
#
# Checkpoint selection:
#   Each experiment dir is expected to contain a final-model subdir created by
#   model.save_pretrained() in run_exp.py, named after wandb_name (e.g. lora_*_noDS,
#   full_*_DDP). We pick the first such subdir whose prefix (lora_/full_) matches
#   the experiment's detected model type. If none exists, mark as incomplete.
#
# Usage:
#   # Evaluate all experiments (auto-detects lora vs full by prefix)
#   bash run_all_evals.sh
#
#   # Specify a different results directory
#   RESULTS_DIR=.//results_llama_sweep bash run_all_evals.sh
#
#   # Dry run - only show what would be evaluated, don't actually run
#   DRY_RUN=1 bash run_all_evals.sh
#
#   # Skip experiments matching a pattern (regex)
#   SKIP_PATTERN="codefeedback" bash run_all_evals.sh
#
#   # Only run experiments matching a pattern (regex)
#   ONLY_PATTERN="meta_math.*seed0" bash run_all_evals.sh
#
#   # Override the wandb project (defaults to run_lm_eval.sh's `lm-eval-ft-llama-7b`)
#   WANDB_PROJECT=lm-eval-llama-13b bash run_all_evals.sh

cd "$(dirname "$0")"

RESULTS_DIR="${RESULTS_DIR:-./results}"

# Track results
FAILED_CKPTS=()
INCOMPLETE_EXPS=()
SUCCESS_COUNT=0
SKIP_COUNT=0

# Environment settings
export CLEANUP_MERGED=${CLEANUP_MERGED:-1}  # Delete merged model after eval to save disk space
export NUM_GPUS=${NUM_GPUS:-8}              # TP=8 (lm-eval-harness official recommendation)
export DATA_PARALLEL=${DATA_PARALLEL:-1}    # DP=1
# Wandb project override. If unset, run_lm_eval.sh's own default (`lm-eval-ft-llama-7b`)
# kicks in — likely wrong for non-7B sweeps, so prefer to set it explicitly.
[[ -n "$WANDB_PROJECT" ]] && export WANDB_PROJECT

# Dry run mode: only show what would be evaluated, don't actually run
DRY_RUN=${DRY_RUN:-0}

# Force re-evaluation (skip "already evaluated" check)
FORCE_EVAL="${FORCE_EVAL:-0}"

# Pattern filtering:
#   SKIP_PATTERN: skip experiments matching this regex
#   ONLY_PATTERN: only run experiments matching this regex
SKIP_PATTERN="${SKIP_PATTERN:-}"
ONLY_PATTERN="${ONLY_PATTERN:-}"

# Count total experiment directories
TOTAL=$(ls -d $RESULTS_DIR/*/ 2>/dev/null | wc -l)
CURRENT=0

echo "=============================================="
echo "Batch Evaluation of All Checkpoints"
echo "=============================================="
echo "Results dir: $RESULTS_DIR"
echo "Wandb project: ${WANDB_PROJECT:-<run_lm_eval.sh default>}"
echo "Total experiments: $TOTAL"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "*** DRY RUN MODE - No actual evaluation will run ***"
fi
if [[ "$FORCE_EVAL" == "1" ]]; then
    echo "*** FORCE EVAL MODE - Will re-evaluate even if results exist ***"
fi
if [[ -n "$SKIP_PATTERN" ]]; then
    echo "Skip pattern: $SKIP_PATTERN"
fi
if [[ -n "$ONLY_PATTERN" ]]; then
    echo "Only pattern: $ONLY_PATTERN"
fi
echo "=============================================="
echo "Task mapping:"
echo "  meta_math:    gsm8k_qa"
echo "  wizard_lm:    arc_challenge,arc_easy,hellaswag,piqa,winogrande,boolq,openbookqa"
echo "  codefeedback: humaneval"
echo "=============================================="

# Helper function to check if a path is a valid checkpoint
# Match any number of safetensors / pytorch_model shards via glob (compgen).
has_full_weights() {
    local ckpt_path="$1"
    [[ -f "${ckpt_path}/config.json" ]] && \
        ([[ -f "${ckpt_path}/model.safetensors" ]] || \
         compgen -G "${ckpt_path}/model-*.safetensors" > /dev/null || \
         [[ -f "${ckpt_path}/pytorch_model.bin" ]] || \
         compgen -G "${ckpt_path}/pytorch_model-*.bin" > /dev/null)
}

is_valid_checkpoint() {
    local ckpt_path="$1"
    local is_lora="${2:-}"

    if [[ "$is_lora" == "true" ]]; then
        [[ -f "${ckpt_path}/adapter_config.json" ]] && return 0
    elif [[ "$is_lora" == "false" ]]; then
        has_full_weights "$ckpt_path" && return 0
    else
        [[ -f "${ckpt_path}/adapter_config.json" ]] && return 0
        has_full_weights "$ckpt_path" && return 0
    fi
    return 1
}

# Helper function to run evaluation
# Args: $1=checkpoint_path, $2=tasks, $3=extra_env, $4=wandb_run_name, $5=is_lora
run_single_eval() {
    local ckpt_path="$1"
    local tasks="$2"
    local extra_env="$3"
    local wandb_name="$4"
    local is_lora="$5"

    # Skip if already has eval results (unless FORCE_EVAL is set)
    if [[ "$FORCE_EVAL" != "1" ]] && ls -d "${ckpt_path}"/lm_eval_results* &>/dev/null; then
        echo "Skipping: Already has evaluation results at $ckpt_path"
        return 2
    fi

    if ! is_valid_checkpoint "$ckpt_path" "$is_lora"; then
        echo "No valid checkpoint found at $ckpt_path, skipping"
        return 1
    fi

    echo "Evaluating: $ckpt_path"

    local wandb_env=""
    if [[ -n "$wandb_name" ]]; then
        wandb_env="WANDB_RUN_NAME=$wandb_name"
    fi

    local eval_mode
    if [[ "$is_lora" == "true" ]]; then
        eval_mode="--merge"
    else
        eval_mode=""
    fi

    # Dry run mode
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] Would execute:"
        echo "  $extra_env $wandb_env bash run_lm_eval.sh \"$ckpt_path\" \"$tasks\" $eval_mode"
        return 0
    fi

    # Empty env strings expand away cleanly; eval handles leading/extra spaces.
    eval "$extra_env $wandb_env bash run_lm_eval.sh \"$ckpt_path\" \"$tasks\" $eval_mode"
}

for exp_dir in $RESULTS_DIR/*/; do
    exp_name=$(basename "$exp_dir")
    CURRENT=$((CURRENT + 1))

    # Pattern filtering
    if [[ -n "$SKIP_PATTERN" ]] && [[ "$exp_name" =~ $SKIP_PATTERN ]]; then
        echo "[$CURRENT/$TOTAL] Skipping (matches SKIP_PATTERN): $exp_name"
        continue
    fi
    if [[ -n "$ONLY_PATTERN" ]] && [[ ! "$exp_name" =~ $ONLY_PATTERN ]]; then
        echo "[$CURRENT/$TOTAL] Skipping (doesn't match ONLY_PATTERN): $exp_name"
        continue
    fi

    echo ""
    echo "=============================================="
    echo "[$CURRENT/$TOTAL] Processing experiment: $exp_name"
    echo "=============================================="

    # Determine model type by prefix
    if [[ "$exp_name" == lora_* ]]; then
        IS_LORA=true
        echo "Model type: LoRA (detected by prefix)"
    elif [[ "$exp_name" == full_* ]]; then
        IS_LORA=false
        echo "Model type: Full fine-tuned (detected by prefix)"
    else
        echo "WARNING: Unknown model type prefix in $exp_name (expected lora_* or full_*), skipping..."
        continue
    fi

    # Determine tasks based on dataset name
    if [[ "$exp_name" == *"meta_math"* ]]; then
        TASKS="gsm8k_qa"
        EXTRA_ENV=""
    elif [[ "$exp_name" == *"wizard_lm"* ]]; then
        TASKS="arc_challenge,arc_easy,hellaswag,piqa,winogrande,boolq,openbookqa"
        EXTRA_ENV=""
    elif [[ "$exp_name" == *"codefeedback"* ]]; then
        TASKS="humaneval"
        EXTRA_ENV="HF_ALLOW_CODE_EVAL=1"
    else
        echo "WARNING: Unknown dataset type in $exp_name, skipping..."
        continue
    fi

    echo "Tasks: $TASKS"
    if [[ "$IS_LORA" == "true" ]]; then
        echo "Backend: vLLM (--merge, for LoRA)"
    else
        echo "Backend: vLLM (direct, for full models)"
    fi

    # Find the saved final-model subdir (model.save_pretrained writes to <exp_dir>/<wandb_name>/).
    # The Trainer's intermediate `checkpoint-N/` dirs are removed by cleanup_outputs=True at
    # the end of training, so the lora_*/full_*-named subdir is the canonical eval target.
    target_ckpt=""
    ckpt_source=""
    for subdir in "${exp_dir}"*/; do
        subdir_name=$(basename "$subdir")
        if [[ "$subdir_name" == checkpoint-* ]]; then
            continue
        fi
        if [[ "$IS_LORA" == "true" && "$subdir_name" != lora_* ]]; then
            continue
        fi
        if [[ "$IS_LORA" == "false" && "$subdir_name" != full_* ]]; then
            continue
        fi
        if [[ -d "$subdir" ]] && is_valid_checkpoint "$subdir" "$IS_LORA"; then
            target_ckpt="$subdir"
            ckpt_source="$subdir_name"
            echo "Found checkpoint: $target_ckpt"
            break
        fi
    done

    # Check if we found a valid checkpoint
    if [[ -z "$target_ckpt" ]]; then
        echo "WARNING: Experiment $exp_name is INCOMPLETE - no valid checkpoint found"
        INCOMPLETE_EXPS+=("$exp_name")
        continue
    fi

    wandb_name="$exp_name"

    echo "Using checkpoint: $target_ckpt"
    echo "Checkpoint source: $ckpt_source"
    echo "Wandb run name: $wandb_name"

    # Run evaluation
    if run_single_eval "$target_ckpt" "$TASKS" "$EXTRA_ENV" "$wandb_name" "$IS_LORA"; then
        echo "Completed: $exp_name"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        ret=$?
        if [[ $ret -eq 2 ]]; then
            SKIP_COUNT=$((SKIP_COUNT + 1))
        else
            echo "FAILED: $exp_name"
            FAILED_CKPTS+=("$exp_name")
        fi
    fi
done

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "=============================================="
echo "Summary:"
echo "  Success: $SUCCESS_COUNT"
echo "  Skipped (already evaluated): $SKIP_COUNT"
echo "  Failed:  ${#FAILED_CKPTS[@]}"
echo "  Incomplete (no checkpoint): ${#INCOMPLETE_EXPS[@]}"

if [[ ${#INCOMPLETE_EXPS[@]} -gt 0 ]]; then
    echo ""
    echo "Incomplete experiments (no valid checkpoint found):"
    for exp in "${INCOMPLETE_EXPS[@]}"; do
        echo "  - $exp"
    done
fi

if [[ ${#FAILED_CKPTS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed checkpoints:"
    for ckpt in "${FAILED_CKPTS[@]}"; do
        echo "  - $ckpt"
    done
    exit 1
fi
