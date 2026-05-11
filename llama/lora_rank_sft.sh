#!/bin/bash
# LLaMA LoRA rank sweep (multi-node queue).
# Sweeps (RANKS × LRS × VARIANTS) over a fixed dataset / seed.
# Each task uses 8 GPUs on one node; multiple nodes pull from a shared queue.
#
# Usage:
#   bash lora_rank_sft.sh                                              # default sweep
#   NODE_NAME=node1 bash lora_rank_sft.sh                              # tag logs with a node name
#   RESET_QUEUE=1 bash lora_rank_sft.sh                                # wipe the shared queue and re-init
#   RANKS="2 4 8 16" LRS="1e-4 2e-4" bash lora_rank_sft.sh             # custom grid
#   DATASET=codefeedback VARIANTS="lora_adamw lora_muon" bash lora_rank_sft.sh

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
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-2-7b-hf"}
MODEL_DTYPE=${MODEL_DTYPE:-"bf16"}
FLASH_ATTENTION=${FLASH_ATTENTION:-"true"}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-"false"}
MAX_LENGTH=${MAX_LENGTH:-1024}

# Sweep grid (override via env to customize)
DATASET=${DATASET:-"meta_math"}                  # fixed dataset
SEED=${SEED:-0}                                  # fixed seed
VARIANTS=${VARIANTS:-"lora_adamw"}               # space-separated: lora_adamw / lora_muon / lora_muon_pe
RANKS=${RANKS:-"512"}                            # space-separated LoRA ranks
LRS=${LRS:-"2e-4 3e-4 5e-4"}                     # space-separated learning rates
LORA_ALPHA_MULTIPLIER=${LORA_ALPHA_MULTIPLIER:-2}  # alpha = multiplier * rank

# Training
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
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

# Muon
NS_DTYPE=${NS_DTYPE:-"bf16"}

# Parallelism: Zero-2 shards optimizer state (helps at high rank).
# Override to "none" for plain DDP (recommended if Muon + Zero-2 misbehaves).
DEEPSPEED=${DEEPSPEED:-"conf/ds_zero2.json"}

# Wandb
WANDB_PROJECT=${WANDB_PROJECT:-"llama_instruction_tuning"}
WANDB_MODE=${WANDB_MODE:-"online"}

# GPU + multi-node queue
NUM_GPUS=${NUM_GPUS:-8}
NODE_NAME=${NODE_NAME:-$(hostname)}
QUEUE_DIR=${QUEUE_DIR:-"$PWD/.queue_llama_rank_sweep"}
QUEUE_FILE=${QUEUE_FILE:-"$QUEUE_DIR/tasks.queue"}
LOCK_FILE=${LOCK_FILE:-"$QUEUE_DIR/lock"}
RESET_QUEUE=${RESET_QUEUE:-0}

# Logging / output
LOG_DIR=${LOG_DIR:-"logs/llama_rank_sweep"}
OUTPUT_DIR=${OUTPUT_DIR:-"./results_llama_rank_sweep"}
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
        # Format: index|task_name (e.g., "0|lora_adamw_meta_math_r512_lr2e-4_seed0_ep1")
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
  local rank="$3"
  local lr="$4"
  local seed="$5"
  local alpha=$((rank * LORA_ALPHA_MULTIPLIER))

  local exp_name="${variant}_${ds}_r${rank}_lr${lr}_seed${seed}_${EXP_SUFFIX}"

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
        training.deepspeed=${DEEPSPEED} \
        evaluation.eval_before_training=${EVAL_BEFORE_TRAINING} \
        evaluation.eval_times=${EVAL_TIMES} \
        experiment.seed=${seed} \
        experiment.output_dir=${OUTPUT_DIR}/${exp_name} \
        experiment.cleanup_outputs=${CLEANUP_OUTPUTS} \
        experiment.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT} \
        training.save_total_limit=${SAVE_TOTAL_LIMIT} \
        wandb.project=${WANDB_PROJECT} \
        wandb.mode=${WANDB_MODE}"

  local lora_args="peft.lora_r=${rank} peft.lora_alpha=${alpha}"

  local launch_cmd="torchrun --nproc_per_node=${NUM_GPUS} run_exp.py"

  case "$variant" in
    lora_adamw)
      echo "${launch_cmd} \
        init_method=lora \
        ${base_args} \
        ${lora_args} \
        optimizer.name=adamw \
        wandb.name=${exp_name}"
      ;;
    lora_muon_pe)
      echo "${launch_cmd} \
        init_method=lora \
        ${base_args} \
        ${lora_args} \
        optimizer.name=muon \
        optimizer.ns_using_pe=true \
        optimizer.ns_dtype=${NS_DTYPE} \
        wandb.name=${exp_name}"
      ;;
    lora_muon)
      echo "${launch_cmd} \
        init_method=lora \
        ${base_args} \
        ${lora_args} \
        optimizer.name=muon \
        optimizer.ns_using_pe=false \
        optimizer.ns_dtype=${NS_DTYPE} \
        wandb.name=${exp_name}"
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
# Sweep: ranks × lrs × variants  (dataset and seed fixed)
# =============================================================================

echo "Generating task list..."
echo ""

add_task() {
  local variant="$1" ds="$2" rank="$3" lr="$4" seed="$5"
  cmd=$(build_cmd "$variant" "$ds" "$rank" "$lr" "$seed")
  task_list+=("$cmd")
  task_names+=("${variant}_${ds}_r${rank}_lr${lr}_seed${seed}_${EXP_SUFFIX}")
  echo "  [${#task_list[@]}] ${variant} | ${ds} | rank=${rank} | lr=${lr} | seed=${seed}"
}

for rank in ${RANKS}; do
  for lr in ${LRS}; do
    for variant in ${VARIANTS}; do
      add_task "$variant" "$DATASET" "$rank" "$lr" "$SEED"
    done
  done
done

echo ""
total_tasks=${#task_list[@]}

# =============================================================================
# Print Configuration Summary
# =============================================================================

echo "=============================================="
echo "LLaMA LoRA Rank Sweep"
echo "=============================================="
echo "Node:       ${NODE_NAME}"
echo "Model:      ${MODEL_NAME} (${MODEL_DTYPE}, flash=${FLASH_ATTENTION}, grad_ckpt=${GRADIENT_CHECKPOINTING})"
echo "Max length: ${MAX_LENGTH}"
echo "Epochs:     ${NUM_EPOCHS}   max_steps=${MAX_STEPS}"
echo "Eval:       before=${EVAL_BEFORE_TRAINING}, times=${EVAL_TIMES}"
echo "Batch:      ${PER_DEVICE_BATCH_SIZE} (per device) / ${REAL_BATCH_SIZE} (effective)"
echo "DeepSpeed:  ${DEEPSPEED}"
echo "LoRA alpha: ${LORA_ALPHA_MULTIPLIER} * rank"
echo "Muon:       ns_dtype=${NS_DTYPE}"
echo "----------------------------------------------"
echo "Dataset:    ${DATASET} (fixed)   Seed: ${SEED} (fixed)"
echo "Variants:   ${VARIANTS}"
echo "Ranks:      ${RANKS}"
echo "LRs:        ${LRS}"
echo "----------------------------------------------"
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
