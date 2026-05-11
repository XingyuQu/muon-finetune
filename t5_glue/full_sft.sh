#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Silence TensorFlow logs pulled in by transformers.
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export TRANSFORMERS_NO_TF=${TRANSFORMERS_NO_TF:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

MODEL_NAME=${MODEL_NAME:-"t5-base"}
EVAL_PROTOCOL=${EVAL_PROTOCOL:-train_val_test}
WANDB_PROJECT=${WANDB_PROJECT:-"T5_SFT_bs64"}
log_dir="logs"
mkdir -p "$log_dir"

MAX_GRAD_NORM=1
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-64}
REAL_BATCH_SIZE=${REAL_BATCH_SIZE:-64}
QUEUE_DIR=${QUEUE_DIR:-"$PWD/.queue_t5_glue_full_sft"}
QUEUE_FILE=${QUEUE_FILE:-"$QUEUE_DIR/tasks.queue"}
LOCK_FILE=${LOCK_FILE:-"$QUEUE_DIR/lock"}
RESET_QUEUE=${RESET_QUEUE:-0}

# Update this list to match available GPUs.
gpus=(0 1 2 3 4 5 6 7)
if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
fi

epochs_for_dataset() {
  case "$1" in
    mrpc|cola) echo 5 ;;
    *) echo 3 ;;
  esac
}

init_queue() {
  mkdir -p "$QUEUE_DIR"
  (
    flock -x 200
    if [[ "$RESET_QUEUE" == "1" ]]; then
      rm -f "$QUEUE_FILE" "$QUEUE_DIR/initialized"
    fi
    if [[ ! -f "$QUEUE_DIR/initialized" ]]; then
      : > "$QUEUE_FILE"
      for i in "${!task_list[@]}"; do
        echo "$i" >> "$QUEUE_FILE"
      done
      echo "initialized" > "$QUEUE_DIR/initialized"
    fi
  ) 200>"$LOCK_FILE"
}

build_cmd() {
  local variant="$1"
  local ds="$2"
  local lr="$3"
  local seed="$4"
  local epochs="$5"
  local max_grad_norm="${MAX_GRAD_NORM}"

  case "$variant" in
    full_muon|full_muon_pe)
      using_pe=False
      if [[ "$variant" == "full_muon_pe" ]]; then
        using_pe=True
      fi
      variant_suffix=$([ "$variant" == "full_muon_pe" ] && echo "_pe" || echo "")
      echo "python run_exp.py \
        +peft=full_ft \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++model.learning_rate=${lr} \
        ++model.bf16=False \
        ++model.name=${MODEL_NAME} \
        ++model.max_grad_norm=${max_grad_norm} \
        ++model.epochs=${epochs} \
        ++model.per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
        ++model.real_batch_size=${REAL_BATCH_SIZE} \
        ++model.optimizer=muon \
        ++seed=${seed} \
        ++using_pe=${using_pe} \
        ++wandb.project=${WANDB_PROJECT} \
        ++wandb.name=full_muon${variant_suffix}_${ds}_lr${lr}_seed${seed}"
      ;;
    full_adamw)
      echo "python run_exp.py \
        +peft=full_ft \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++model.learning_rate=${lr} \
        ++model.bf16=False \
        ++model.name=${MODEL_NAME} \
        ++model.max_grad_norm=${max_grad_norm} \
        ++model.epochs=${epochs} \
        ++model.per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
        ++model.real_batch_size=${REAL_BATCH_SIZE} \
        ++model.optimizer=adamw \
        ++seed=${seed} \
        ++wandb.project=${WANDB_PROJECT} \
        ++wandb.name=full_adamw_${ds}_lr${lr}_seed${seed} \
        ++using_pe=False"
      ;;
    *)
      echo "echo 'Unknown variant: $variant'" >&2
      return 1
      ;;
  esac
}

# =============================================================================
# Generate all tasks
# =============================================================================
# learning rates:
#              CoLA   MNLI   MRPC   QNLI   SST-2
# Full-Adam    1e-4   1e-4   1e-4   5e-5   1e-4
# Full-Muon    5e-4   1e-4   5e-4   1e-4   1e-4
# Full-Muon-PE 1e-3   1e-4   5e-4   1e-4   1e-4

task_list=()
task_names=()

seeds=(0 1 2)

# Helper function to add tasks
add_task() {
  local variant="$1"
  local ds="$2"
  local lr="$3"
  local seed="$4"
  local epochs=$(epochs_for_dataset "$ds")
  local cmd=$(build_cmd "$variant" "$ds" "$lr" "$seed" "$epochs")
  task_list+=("$cmd")
  task_names+=("${variant}_${ds}_lr${lr}_seed${seed}")
}

# All experiments with Table 9 learning rates
for seed in "${seeds[@]}"; do
  # ============ CoLA ============
  add_task "full_adamw" "cola" "1e-4" "$seed"
  add_task "full_muon" "cola" "5e-4" "$seed"
  add_task "full_muon_pe" "cola" "1e-3" "$seed"

  # ============ MNLI ============
  add_task "full_adamw" "mnli" "1e-4" "$seed"
  add_task "full_muon" "mnli" "1e-4" "$seed"
  add_task "full_muon_pe" "mnli" "1e-4" "$seed"

  # ============ MRPC ============
  add_task "full_adamw" "mrpc" "1e-4" "$seed"
  add_task "full_muon" "mrpc" "5e-4" "$seed"
  add_task "full_muon_pe" "mrpc" "5e-4" "$seed"

  # ============ QNLI ============
  add_task "full_adamw" "qnli" "5e-5" "$seed"
  add_task "full_muon" "qnli" "1e-4" "$seed"
  add_task "full_muon_pe" "qnli" "1e-4" "$seed"

  # ============ SST-2 ============
  add_task "full_adamw" "sst2" "1e-4" "$seed"
  add_task "full_muon" "sst2" "1e-4" "$seed"
  add_task "full_muon_pe" "sst2" "1e-4" "$seed"
done

total_tasks=${#task_list[@]}
echo "Total tasks: $total_tasks"

# =============================================================================
# GPU Worker function
# =============================================================================
init_queue

get_next_task() {
  (
    flock -x 200
    if [[ -s "$QUEUE_FILE" ]]; then
      head -n 1 "$QUEUE_FILE"
      tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp"
      mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
    fi
  ) 200>"$LOCK_FILE"
}

gpu_worker() {
  local gpu="$1"
  while true; do
    task_idx=$(get_next_task)
    if [[ -z "$task_idx" ]]; then
      break
    fi
    
    local cmd="${task_list[$task_idx]}"
    local name="${task_names[$task_idx]}"
    local log_file="${log_dir}/${name}.log"
    
    echo "[GPU $gpu] Starting task $((task_idx + 1))/$total_tasks: $name"
    
    CUDA_VISIBLE_DEVICES="$gpu" \
    LOCAL_RANK=-1 RANK=0 WORLD_SIZE=1 \
      bash -c "$cmd" > "$log_file" 2>&1
    
    local status=$?
    if [[ $status -eq 0 ]]; then
      echo "[GPU $gpu] Finished: $name"
    else
      echo "[GPU $gpu] FAILED (exit $status): $name"
    fi
  done
  echo "[GPU $gpu] No more tasks, exiting."
}

# =============================================================================
# Launch all GPU workers
# =============================================================================
echo "Starting ${#gpus[@]} GPU workers..."

for gpu in "${gpus[@]}"; do
  gpu_worker "$gpu" &
done

wait

echo "All tasks completed."
