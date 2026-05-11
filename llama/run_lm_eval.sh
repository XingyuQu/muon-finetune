#!/bin/bash
# Evaluate models on benchmarks using lm-eval-harness with vLLM backend
#
# Supported tasks:
#   - gsm8k_qa: Math evaluation (Q:/A: format, matches meta_math training)
#   - humaneval: Code evaluation (Alpaca format, matches codefeedback training)
#   - arc_challenge,arc_easy,hellaswag,piqa,winogrande,boolq,openbookqa: Commonsense
#
# Usage:
#   # Evaluate merged model on gsm8k with vLLM (default, fast)
#   bash run_lm_eval.sh results/lora_adamw_meta_math_lr2e-4_seed0_merged gsm8k_qa
#
#   # Merge LoRA first, then evaluate with vLLM (recommended)
#   bash run_lm_eval.sh results/lora_adamw_meta_math_lr2e-4_seed0 gsm8k_qa --merge
#
#   # Custom number of GPUs
#   NUM_GPUS=1 bash run_lm_eval.sh results/model gsm8k_qa
#
#   # Multiple tasks (commonsense reasoning benchmark suite)
#   bash run_lm_eval.sh results/model "arc_challenge,arc_easy,hellaswag,piqa,winogrande,boolq,openbookqa" --merge
#
#   # Code evaluation
#   HF_ALLOW_CODE_EVAL=1 bash run_lm_eval.sh results/model humaneval --merge
#
#   # Custom fewshot (0-shot, 5-shot, 8-shot, etc.)
#   NUM_FEWSHOT=0 bash run_lm_eval.sh results/model gsm8k_qa --merge
#
#   # Delete merged model after evaluation to save disk space
#   CLEANUP_MERGED=1 bash run_lm_eval.sh results/model gsm8k_qa --merge
#
#   # Disable wandb logging
#   WANDB_DISABLED=1 bash run_lm_eval.sh results/model gsm8k_qa --merge


set -e

cd "$(dirname "$0")"

# =============================================================================
# Config
# =============================================================================
# Parallelism settings:
#   TP (tensor_parallel_size): split model across GPUs (works for any model size, default)
#   DP (data_parallel_size):   spawn N independent replicas (higher peak throughput on small
#     models with large datasets, but lm-eval-harness officially recommends TP)
NUM_GPUS=${NUM_GPUS:-8}        # Tensor parallelism (TP) - lm-eval-harness official recommendation
DATA_PARALLEL=${DATA_PARALLEL:-1}  # Data parallelism (DP)
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
# vLLM engine context window (prompt + generated tokens). NOT the generation length cap —
# per-task generation length is set in the task yaml as `generation_kwargs.max_gen_toks`.
# Only needs to be >= max(prompt_len + max_gen_toks) across all eval samples.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
BATCH_SIZE=${BATCH_SIZE:-auto}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}  # Default to 0-shot
CLEANUP_MERGED=${CLEANUP_MERGED:-0}  # Set to 1 to delete merged model after eval
WANDB_PROJECT=${WANDB_PROJECT:-lm-eval-ft-llama-7b}
WANDB_DISABLED=${WANDB_DISABLED:-0}  # Set to 1 to disable wandb logging
CUSTOM_TASKS_PATH=${CUSTOM_TASKS_PATH:-"$(dirname "$0")/lm_eval_tasks"}  # Path to custom task definitions
OUTPUT_PREFIX=${OUTPUT_PREFIX:-lm_eval_results}  # Output directory prefix

# =============================================================================
# Parse arguments
# =============================================================================
if [[ $# -lt 2 ]]; then
    echo "Usage: bash run_lm_eval.sh <model_path> <tasks> [--merge]"
    echo ""
    echo "Arguments:"
    echo "  model_path    Path to model or LoRA adapter"
    echo "  tasks         Comma-separated list of tasks (e.g., gsm8k_qa,hellaswag)"
    echo ""
    echo "Options:"
    echo "  (default)     Assume model_path is already a merged/full model, use vLLM"
    echo "  --merge       Merge LoRA adapter first, then evaluate with vLLM"
    echo ""
    echo "Environment variables:"
    echo "  NUM_GPUS=${NUM_GPUS}                    Tensor parallelism (TP) - split model across GPUs"
    echo "  DATA_PARALLEL=${DATA_PARALLEL}               Data parallelism (DP) - number of model replicas"
    echo "  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}  GPU memory utilization"
    echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}              Max sequence length"
    echo "  BATCH_SIZE=${BATCH_SIZE}                Batch size (auto recommended)"
    echo "  NUM_FEWSHOT=<n>            Number of few-shot examples (default: 0)"
    echo "                             Use comma-separated for multiple: NUM_FEWSHOT=\"0,5,8\""
    echo "  CLEANUP_MERGED=1           Delete merged model after evaluation to save disk space"
    echo "  HF_ALLOW_CODE_EVAL=1       Enable code execution (required for humaneval)"
    echo "  OUTPUT_PREFIX=<prefix>     Output directory prefix (default: lm_eval_results)"
    echo ""
    echo "Examples:"
    echo "  bash run_lm_eval.sh results/model gsm8k_qa --merge"
    echo "  bash run_lm_eval.sh results/model humaneval --merge"
    echo "  bash run_lm_eval.sh results/model \"arc_challenge,arc_easy,hellaswag\" --merge"
    exit 1
fi

MODEL_PATH="$1"
TASKS="$2"
MODE="${3:-direct}"

# =============================================================================
# Main logic
# =============================================================================

# Convert to absolute path
MODEL_PATH=$(realpath "$MODEL_PATH")

echo "=============================================="
echo "lm-eval-harness Evaluation"
echo "=============================================="
echo "Model path: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Mode: $MODE"
echo "Tensor Parallel (TP): $NUM_GPUS"
echo "Data Parallel (DP): $DATA_PARALLEL"
echo "Num fewshot: ${NUM_FEWSHOT:-0}"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Custom tasks path: $CUSTOM_TASKS_PATH"
echo "Wandb project: ${WANDB_PROJECT}"
echo "Wandb disabled: ${WANDB_DISABLED}"
echo "=============================================="

# Build wandb args
WANDB_ARGS=""
if [[ "$WANDB_DISABLED" != "1" ]]; then
    # Use WANDB_RUN_NAME if set, otherwise use basename of model path
    if [[ -n "$WANDB_RUN_NAME" ]]; then
        RUN_NAME="$WANDB_RUN_NAME"
    else
        RUN_NAME=$(basename "$MODEL_PATH")
    fi
    WANDB_ARGS="--wandb_args project=$WANDB_PROJECT,name=$RUN_NAME"
fi

# Auto-enable code eval for humaneval task
UNSAFE_CODE_ARG=""
if [[ "$TASKS" == *"humaneval"* ]]; then
    export HF_ALLOW_CODE_EVAL=1
    UNSAFE_CODE_ARG="--confirm_run_unsafe_code"
    echo "HF_ALLOW_CODE_EVAL=1 and --confirm_run_unsafe_code (auto-enabled for humaneval)"
fi

# Run lm_eval on `$2` (model path) with `$1`-shot; output goes under $MODEL_PATH/.
run_eval_vllm() {
    local fewshot="$1"
    local model_path="$2"

    local fewshot_args=""
    local output_path=""

    if [[ -n "$fewshot" ]]; then
        fewshot_args="--num_fewshot $fewshot"
        output_path="${MODEL_PATH}/${OUTPUT_PREFIX}_${fewshot}shot"
    else
        output_path="${MODEL_PATH}/${OUTPUT_PREFIX}"
    fi

    echo ""
    echo ">>> Running evaluation (vLLM): ${fewshot:-default}-shot"
    echo ">>> Output: $output_path"
    echo ""

    lm_eval --model vllm \
        --model_args pretrained="$model_path",tensor_parallel_size=$NUM_GPUS,data_parallel_size=$DATA_PARALLEL,dtype=auto,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,max_model_len=$MAX_MODEL_LEN \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --include_path "$CUSTOM_TASKS_PATH" \
        --log_samples \
        $fewshot_args \
        $UNSAFE_CODE_ARG \
        $WANDB_ARGS \
        --output_path "$output_path"
}

case "$MODE" in
    --merge)
        # Merge LoRA first, then evaluate, then cleanup
        MERGED_PATH="${MODEL_PATH}/lm_eval_merged"
        CREATED_MERGED=false

        if [[ ! -d "$MERGED_PATH" ]]; then
            echo "Merging LoRA adapter..."
            python merge_lora.py --adapter_path "$MODEL_PATH" --output_path "$MERGED_PATH"
            CREATED_MERGED=true
        else
            echo "Using existing merged model at $MERGED_PATH"
        fi

        if [[ -n "$NUM_FEWSHOT" && "$NUM_FEWSHOT" == *","* ]]; then
            # Multiple fewshot settings
            IFS=',' read -ra FEWSHOTS <<< "$NUM_FEWSHOT"
            for fs in "${FEWSHOTS[@]}"; do
                run_eval_vllm "$fs" "$MERGED_PATH"
            done
        else
            # Single fewshot setting
            run_eval_vllm "$NUM_FEWSHOT" "$MERGED_PATH"
        fi

        # Cleanup: delete merged model if requested
        if [[ "$CLEANUP_MERGED" == "1" ]]; then
            echo ""
            echo "Cleaning up merged model to save disk space..."
            rm -rf "$MERGED_PATH"
            echo "Deleted: $MERGED_PATH"
        fi
        ;;

    *)
        # Direct evaluation with vLLM (assume already merged or full model)
        if [[ -n "$NUM_FEWSHOT" && "$NUM_FEWSHOT" == *","* ]]; then
            # Multiple fewshot settings
            IFS=',' read -ra FEWSHOTS <<< "$NUM_FEWSHOT"
            for fs in "${FEWSHOTS[@]}"; do
                run_eval_vllm "$fs" "$MODEL_PATH"
            done
        else
            # Single fewshot setting
            run_eval_vllm "$NUM_FEWSHOT" "$MODEL_PATH"
        fi
        ;;
esac

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "=============================================="
