#!/bin/bash
# LLaMA-2-13B LoRA instruction-tuning sweep (multi-node queue).
# Each task uses 8 GPUs on one node; multiple nodes pull from a shared queue.
# All variants run under standard DDP (no DeepSpeed).
#
# Usage:
#   bash lora_13b_sft.sh                     # default sweep
#   NODE_NAME=node1 bash lora_13b_sft.sh     # tag logs with a node name
#   RESET_QUEUE=1 bash lora_13b_sft.sh       # wipe the shared queue and re-init

set -u
set -o pipefail

cd "$(dirname "$0")"

# =============================================================================
# Environment
# =============================================================================
export TRANSFORMERS_NO_TF=${TRANSFORMERS_NO_TF:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# =============================================================================
# Configurable Parameters
# =============================================================================

# Model
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-2-13b-hf"}
MODEL_DTYPE=${MODEL_DTYPE:-"bf16"}
FLASH_ATTENTION=${FLASH_ATTENTION:-"true"}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-"false"}
MAX_LENGTH=${MAX_LENGTH:-1024}

# Training
SEEDS=${SEEDS:-"0 1 2"}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
REAL_BATCH_SIZE=${REAL_BATCH_SIZE:-32}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
NUM_EPOCHS=${NUM_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}             # -1 = use NUM_EPOCHS
EVAL_BEFORE_TRAINING=${EVAL_BEFORE_TRAINING:-"false"}
EVAL_TIMES=${EVAL_TIMES:-5}
CLEANUP_OUTPUTS=${CLEANUP_OUTPUTS:-"false"}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-"null"}  # "null" | "auto" | <path>

# LoRA
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}

# Muon
USING_PE=${USING_PE:-"true"}
NS_DTYPE=${NS_DTYPE:-"bf16"}

# Wandb
WANDB_PROJECT=${WANDB_PROJECT:-"llama13b_instruction_tuning"}
WANDB_MODE=${WANDB_MODE:-"online"}

# GPU + multi-node queue
NUM_GPUS=${NUM_GPUS:-8}
NODE_NAME=${NODE_NAME:-$(hostname)}
QUEUE_DIR=${QUEUE_DIR:-"$PWD/.queue_llama_13b_sweep"}
QUEUE_FILE=${QUEUE_FILE:-"$QUEUE_DIR/tasks.queue"}
LOCK_FILE=${LOCK_FILE:-"$QUEUE_DIR/lock"}
RESET_QUEUE=${RESET_QUEUE:-0}

# Logging / output
LOG_DIR=${LOG_DIR:-"logs/llama_13b_sweep"}
OUTPUT_DIR=${OUTPUT_DIR:-"./results_llama_13b_sweep"}
mkdir -p "$LOG_DIR"

# =============================================================================
# Queue Functions (with file locking for multi-node safety)
# =============================================================================

init_queue() {
  mkdir -p "$QUEUE_DIR"
  (
    flock -x 200
    if [[ "$RESET_QUEUE" == "1" ]]; then
      rm -f "$QUEUE_FILE" "$QUEUE_DIR/initialized"
      echo "[$NODE_NAME] Queue reset requested"
    fi
    if [[ ! -f "$QUEUE_DIR/initialized" ]]; then
      : > "$QUEUE_FILE"
      for i in "${!task_list[@]}"; do
        # Format: index|task_name (e.g., "0|lora_muon_codefeedback_lr7e-4_seed0_ep1")
        echo "${i}|${task_names[$i]}" >> "$QUEUE_FILE"
      done
      echo "initialized" > "$QUEUE_DIR/initialized"
      echo "[$NODE_NAME] Queue initialized with $total_tasks tasks"
    else
      remaining=$(wc -l < "$QUEUE_FILE" 2>/dev/null || echo "0")
      echo "[$NODE_NAME] Queue already exists, $remaining tasks remaining"
    fi
  ) 200>"$LOCK_FILE"
}

get_next_task() {
  (
    flock -x 200
    if [[ -s "$QUEUE_FILE" ]]; then
      # Returns format: index|task_name
      head -n 1 "$QUEUE_FILE"
      tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp" && mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
    fi
  ) 200>"$LOCK_FILE"
}

get_remaining_tasks() {
  (
    flock -s 200
    wc -l < "$QUEUE_FILE" 2>/dev/null || echo "0"
  ) 200>"$LOCK_FILE"
}

# =============================================================================
# Command Builder
# =============================================================================

# Single source of truth for the run-name suffix (used in exp_name + task_name).
if [[ "$MAX_STEPS" -gt 0 ]]; then
  EXP_SUFFIX="steps${MAX_STEPS}"
else
  EXP_SUFFIX="ep${NUM_EPOCHS}"
fi

build_cmd() {
  local variant="$1"
  local ds="$2"
  local lr="$3"
  local seed="$4"

  local exp_name="${variant}_${ds}_lr${lr}_seed${seed}_${EXP_SUFFIX}"

  local base_args="model.name=${MODEL_NAME} \
        model.dtype=${MODEL_DTYPE} \
        model.flash_attention=${FLASH_ATTENTION} \
        model.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
        dataset.name=${ds} \
        dataset.max_length=${MAX_LENGTH} \
        training.learning_rate=${lr} \
        training.num_epochs=${NUM_EPOCHS} \
        training.max_steps=${MAX_STEPS} \
        training.per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
        training.real_batch_size=${REAL_BATCH_SIZE} \
        training.max_grad_norm=${MAX_GRAD_NORM} \
        training.warmup_ratio=${WARMUP_RATIO} \
        training.weight_decay=${WEIGHT_DECAY} \
        evaluation.eval_before_training=${EVAL_BEFORE_TRAINING} \
        evaluation.eval_times=${EVAL_TIMES} \
        experiment.seed=${seed} \
        experiment.output_dir=${OUTPUT_DIR}/${exp_name} \
        experiment.cleanup_outputs=${CLEANUP_OUTPUTS} \
        experiment.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT} \
        training.save_total_limit=${SAVE_TOTAL_LIMIT} \
        wandb.project=${WANDB_PROJECT} \
        wandb.mode=${WANDB_MODE}"

  local lora_args="peft.lora_r=${LORA_R} peft.lora_alpha=${LORA_ALPHA}"

  local launch_cmd="torchrun --nproc_per_node=${NUM_GPUS} run_exp.py"

  case "$variant" in
    lora_adamw)
      echo "${launch_cmd} \
        init_method=lora \
        ${base_args} \
        ${lora_args} \
        optimizer.name=adamw \
        wandb.name=lora_adamw_${ds}_lr${lr}_seed${seed}_${EXP_SUFFIX}"
      ;;
    lora_muon_pe)
      echo "${launch_cmd} \
        init_method=lora \
        ${base_args} \
        ${lora_args} \
        optimizer.name=muon \
        optimizer.ns_using_pe=${USING_PE} \
        optimizer.ns_dtype=${NS_DTYPE} \
        wandb.name=lora_muon_pe_${ds}_lr${lr}_seed${seed}_${EXP_SUFFIX}"
      ;;
    lora_muon)
      echo "${launch_cmd} \
        init_method=lora \
        ${base_args} \
        ${lora_args} \
        optimizer.name=muon \
        optimizer.ns_using_pe=false \
        optimizer.ns_dtype=${NS_DTYPE} \
        wandb.name=lora_muon_${ds}_lr${lr}_seed${seed}_${EXP_SUFFIX}"
      ;;
    *)
      echo "echo 'Unknown variant: $variant'" >&2
      return 1
      ;;
  esac
}

# =============================================================================
# Generate Task List
# =============================================================================

task_list=()
task_names=()

# =============================================================================
# Sweep: codefeedback @ lr=7e-4, lora_adamw + lora_muon, across seeds
# =============================================================================

echo "Generating task list..."
echo ""

add_task() {
  local variant="$1" ds="$2" lr="$3" seed="$4"
  cmd=$(build_cmd "$variant" "$ds" "$lr" "$seed")
  task_list+=("$cmd")
  task_names+=("${variant}_${ds}_lr${lr}_seed${seed}_${EXP_SUFFIX}")
  echo "  [${#task_list[@]}] ${variant} | ${ds} | lr=${lr} | seed=${seed}"
}

for seed in ${SEEDS}; do
  add_task lora_adamw codefeedback 7e-4 "$seed"
  add_task lora_muon  codefeedback 7e-4 "$seed"
done

echo ""
total_tasks=${#task_list[@]}

# =============================================================================
# Print Configuration Summary
# =============================================================================

echo "=============================================="
echo "LLaMA-2-13B LoRA Sweep"
echo "=============================================="
echo "Node:       ${NODE_NAME}"
echo "Model:      ${MODEL_NAME} (${MODEL_DTYPE}, flash=${FLASH_ATTENTION}, grad_ckpt=${GRADIENT_CHECKPOINTING})"
echo "Max length: ${MAX_LENGTH}"
echo "Epochs:     ${NUM_EPOCHS}   max_steps=${MAX_STEPS}"
echo "Eval:       before=${EVAL_BEFORE_TRAINING}, times=${EVAL_TIMES}"
echo "Batch:      ${PER_DEVICE_BATCH_SIZE} (per device) / ${REAL_BATCH_SIZE} (effective)"
echo "LoRA:       r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "Muon:       PE=${USING_PE}, ns_dtype=${NS_DTYPE}"
echo "Seeds:      ${SEEDS}"
echo "GPUs/node:  ${NUM_GPUS}"
echo "Total tasks: $total_tasks"
echo "Queue dir:  ${QUEUE_DIR}"
echo "Log dir:    ${LOG_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=============================================="

if [[ $total_tasks -eq 0 ]]; then
  echo "No tasks to run. Check your configuration."
  exit 0
fi

# =============================================================================
# Initialize Queue and Run Worker
# =============================================================================

init_queue

echo "[$NODE_NAME] Starting worker..."

completed=0
failed=0

# Track the in-flight task so a SIGINT/SIGTERM can return it to the head of
# the queue. Empty between iterations and after a task finishes.
current_task_entry=""

return_in_flight_task() {
  if [[ -n "$current_task_entry" ]]; then
    (
      flock -x 200
      tmp=$(mktemp "${QUEUE_FILE}.XXXXXX")
      echo "$current_task_entry" > "$tmp"
      [[ -s "$QUEUE_FILE" ]] && cat "$QUEUE_FILE" >> "$tmp"
      mv "$tmp" "$QUEUE_FILE"
    ) 200>"$LOCK_FILE"
    echo ""
    echo "[$NODE_NAME] Interrupted - returned task to queue: $current_task_entry"
  fi
}

trap 'return_in_flight_task; exit 130' INT TERM

while true; do
  current_task_entry=""
  task_entry=$(get_next_task)
  if [[ -z "$task_entry" ]]; then
    echo "[$NODE_NAME] No more tasks in queue, exiting."
    break
  fi
  current_task_entry="$task_entry"

  # Parse format: index|task_name
  task_idx="${task_entry%%|*}"
  task_info="${task_entry#*|}"

  cmd="${task_list[$task_idx]}"
  name="${task_names[$task_idx]}"
  log_file="${LOG_DIR}/${name}.log"
  remaining=$(get_remaining_tasks)

  echo ""
  echo "[$NODE_NAME] ================================================"
  echo "[$NODE_NAME] Starting: $name"
  echo "[$NODE_NAME] Task index: $task_idx | $task_info"
  echo "[$NODE_NAME] Remaining in queue: $remaining"
  echo "[$NODE_NAME] Log: $log_file"
  echo "[$NODE_NAME] ================================================"

  start_time=$(date +%s)

  if bash -c "$cmd" 2>&1 | tee "$log_file"; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "[$NODE_NAME] Finished: $name (took ${duration}s)"
    ((completed++))
  else
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "[$NODE_NAME] FAILED: $name (took ${duration}s)"
    ((failed++))
  fi
  current_task_entry=""
done

echo ""
echo "[$NODE_NAME] ================================================"
echo "[$NODE_NAME] Worker finished"
echo "[$NODE_NAME] Completed: $completed, Failed: $failed"
echo "[$NODE_NAME] ================================================"
