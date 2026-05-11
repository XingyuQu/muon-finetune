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

LORA_R=8
DEFAULT_LORA_ALPHA=16
MAX_GRAD_NORM=1
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-64}
REAL_BATCH_SIZE=${REAL_BATCH_SIZE:-64}
QUEUE_DIR=${QUEUE_DIR:-"$PWD/.queue_t5_glue_sft"}
QUEUE_FILE=${QUEUE_FILE:-"$QUEUE_DIR/tasks.queue"}
LOCK_FILE=${LOCK_FILE:-"$QUEUE_DIR/lock"}
RESET_QUEUE=${RESET_QUEUE:-0}

# Update this list to match available GPUs.
gpus=(0 1 2 3 4 5 6 7)
if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
fi

stable_gamma_for_dataset() {
  case "$1" in
    mnli) echo 128 ;;
    sst2) echo 16 ;;
    cola) echo 128 ;;
    qnli) echo 128 ;;
    mrpc) echo 64 ;;
    *) echo 64 ;;
  esac
}

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
  local max_grad_norm="$4"
  local seed="$5"
  local lora_alpha="$6"
  local stable_gamma="$7"
  local epochs="$8"

  case "$variant" in
    lora_muon)
      echo "python run_exp.py \
        +peft=all \
        +init=default \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=False \
        ++model.learning_rate=${lr} \
        ++model.bf16=False \
        ++model.name=${MODEL_NAME} \
        ++model.max_grad_norm=${max_grad_norm} \
        ++model.epochs=${epochs} \
        ++model.per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
        ++model.real_batch_size=${REAL_BATCH_SIZE} \
        ++model.optimizer=muon \
        ++using_pe=False \
        ++seed=${seed} \
        ++wandb.project=${WANDB_PROJECT} \
        ++wandb.name=lora_muon_${ds}_lr${lr}_seed${seed}"
      ;;
    lora_muon_pe)
      echo "python run_exp.py \
        +peft=all \
        +init=default \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=False \
        ++model.learning_rate=${lr} \
        ++model.bf16=False \
        ++model.name=${MODEL_NAME} \
        ++model.max_grad_norm=${max_grad_norm} \
        ++model.epochs=${epochs} \
        ++model.per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
        ++model.real_batch_size=${REAL_BATCH_SIZE} \
        ++model.optimizer=muon \
        ++using_pe=True \
        ++seed=${seed} \
        ++wandb.project=${WANDB_PROJECT} \
        ++wandb.name=lora_muon_${ds}_lr${lr}_pe_seed${seed}"
      ;;
    lora_muon_pe_rs)
      echo "python run_exp.py \
        +peft=all \
        +init=default \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=True \
        ++model.learning_rate=${lr} \
        ++model.bf16=False \
        ++model.name=${MODEL_NAME} \
        ++model.max_grad_norm=${max_grad_norm} \
        ++model.epochs=${epochs} \
        ++model.per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
        ++model.real_batch_size=${REAL_BATCH_SIZE} \
        ++model.optimizer=muon \
        ++using_pe=True \
        ++seed=${seed} \
        ++wandb.project=${WANDB_PROJECT} \
        ++wandb.name=lora_muon_${ds}_lr${lr}_pe_rs_seed${seed}"
      ;;
    loraone_adamw_rs)
      echo "python run_exp.py \
        +peft=all \
        +init=lora_one \
        ++init.skip_merge=True \
        ++init.stable_gamma=${stable_gamma} \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=True \
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
        ++wandb.name=loraone_adamw_${ds}_lr${lr}_rs_seed${seed} \
        ++using_pe=False"
      ;;
    rslora_adamw)
      echo "python run_exp.py \
        +peft=all \
        +init=default \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=True \
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
        ++wandb.name=rslora_adamw_${ds}_lr${lr}_seed${seed} \
        ++using_pe=False"
      ;;
    lora_adamw)
      echo "python run_exp.py \
        +peft=all \
        +init=default \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=False \
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
        ++wandb.name=lora_adamw_${ds}_lr${lr}_seed${seed} \
        ++using_pe=False"
      ;;
    lorarite_adamw)
      echo "python run_exp.py \
        +peft=all \
        +init=default \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=False \
        ++model.learning_rate=${lr} \
        ++model.bf16=False \
        ++model.name=${MODEL_NAME} \
        ++model.max_grad_norm=${max_grad_norm} \
        ++model.epochs=${epochs} \
        ++model.per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
        ++model.real_batch_size=${REAL_BATCH_SIZE} \
        ++model.optimizer=lora_rite \
        ++seed=${seed} \
        ++wandb.project=${WANDB_PROJECT} \
        ++wandb.name=lorarite_adamw_${ds}_lr${lr}_seed${seed} \
        ++using_pe=False"
      ;;
    dora_adamw)
      echo "python run_exp.py \
        +peft=dora \
        +init=default \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=False \
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
        ++wandb.name=dora_adamw_${ds}_lr${lr}_seed${seed} \
        ++using_pe=False"
      ;;
    adalora_adamw)
      echo "python run_exp.py \
        +peft=adalora \
        +init=default \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=False \
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
        ++wandb.name=adalora_adamw_${ds}_lr${lr}_seed${seed} \
        ++using_pe=False"
      ;;
    lorapro_adamw_rs)
      echo "python run_exp.py \
        +peft=all \
        +init=default \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=True \
        ++model.learning_rate=${lr} \
        ++model.bf16=False \
        ++model.name=${MODEL_NAME} \
        ++model.max_grad_norm=${max_grad_norm} \
        ++model.epochs=${epochs} \
        ++model.per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
        ++model.real_batch_size=${REAL_BATCH_SIZE} \
        ++model.optimizer=lorapro \
        ++seed=${seed} \
        ++wandb.project=${WANDB_PROJECT} \
        ++wandb.name=lorapro_adamw_${ds}_lr${lr}_rs_seed${seed} \
        ++using_pe=False"
      ;;
    pissa_adamw)
      echo "python run_exp.py \
        +peft=all \
        +init=pissa \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=False \
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
        ++wandb.name=pissa_adamw_${ds}_lr${lr}_seed${seed} \
        ++using_pe=False"
      ;;
    loraone_muon_pe_rs)
      echo "python run_exp.py \
        +peft=all \
        +init=lora_one \
        ++init.skip_merge=True \
        ++init.stable_gamma=${stable_gamma} \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=True \
        ++model.learning_rate=${lr} \
        ++model.bf16=False \
        ++model.name=${MODEL_NAME} \
        ++model.max_grad_norm=${max_grad_norm} \
        ++model.epochs=${epochs} \
        ++model.per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
        ++model.real_batch_size=${REAL_BATCH_SIZE} \
        ++model.optimizer=muon \
        ++using_pe=True \
        ++seed=${seed} \
        ++wandb.project=${WANDB_PROJECT} \
        ++wandb.name=loraone_muon_${ds}_lr${lr}_pe_rs_seed${seed}"
      ;;
    pissa_muon_pe)
      echo "python run_exp.py \
        +peft=all \
        +init=pissa \
        ++dataset_name=${ds} \
        ++eval_protocol=${EVAL_PROTOCOL} \
        ++peft.lora_r=${LORA_R} \
        ++peft.lora_alpha=${lora_alpha} \
        ++peft.use_rslora=False \
        ++model.learning_rate=${lr} \
        ++model.bf16=False \
        ++model.name=${MODEL_NAME} \
        ++model.max_grad_norm=${max_grad_norm} \
        ++model.epochs=${epochs} \
        ++model.per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
        ++model.real_batch_size=${REAL_BATCH_SIZE} \
        ++model.optimizer=muon \
        ++using_pe=True \
        ++seed=${seed} \
        ++wandb.project=${WANDB_PROJECT} \
        ++wandb.name=pissa_muon_${ds}_lr${lr}_pe_seed${seed}"
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
seeds=(0 1 2)

add_task() {
  local variant="$1"
  local ds="$2"
  local lr="$3"
  local lora_alpha="${4:-$DEFAULT_LORA_ALPHA}"

  local stable_gamma
  stable_gamma=$(stable_gamma_for_dataset "$ds")
  local epochs
  epochs=$(epochs_for_dataset "$ds")

  for seed in "${seeds[@]}"; do
    cmd=$(build_cmd "$variant" "$ds" "$lr" \
      "$MAX_GRAD_NORM" "$seed" "$lora_alpha" "$stable_gamma" "$epochs")
    task_list+=("$cmd")
    task_names+=("${variant}_${ds}_lr${lr}_seed${seed}")
  done
}

task_list=()
task_names=()

# =============================================================================
# learning rates
# =============================================================================
#              CoLA   MNLI   MRPC   QNLI   SST-2
# LoRA-Adam    1e-3   1e-3   2e-3   5e-4   5e-4
# LoRA-Muon    2e-3   1e-3   2e-3   5e-4   1e-3
# LoRA-Muon-PE 2e-3   1e-3   2e-3   5e-4   1e-3

# ============ CoLA ============
add_task "lora_adamw" "cola" "1e-3"
add_task "lora_muon" "cola" "2e-3"
add_task "lora_muon_pe" "cola" "2e-3"
# ============ MNLI ============
add_task "lora_adamw" "mnli" "1e-3"
add_task "lora_muon" "mnli" "1e-3"
add_task "lora_muon_pe" "mnli" "1e-3"
# ============ MRPC ============
# add_task "lora_adamw" "mrpc" "2e-3"
add_task "lora_muon" "mrpc" "2e-3"
add_task "lora_muon_pe" "mrpc" "2e-3"
# ============ QNLI ============
add_task "lora_adamw" "qnli" "5e-4"
add_task "lora_muon" "qnli" "5e-4"
add_task "lora_muon_pe" "qnli" "5e-4"
# ============ SST-2 ============
add_task "lora_adamw" "sst2" "5e-4"
add_task "lora_muon" "sst2" "1e-3"
add_task "lora_muon_pe" "sst2" "1e-3"
# =============================================================================
# LoRA variants learning rates
# =============================================================================
# rsLoRA-Adam:      CoLA=5e-4, MNLI=5e-4, MRPC=2e-3, QNLI=1e-3, SST-2=5e-4
# LoRA-One-Adam:    CoLA=5e-4, MNLI=1e-3, MRPC=1e-3, QNLI=5e-4, SST-2=1e-3
# PiSSA-Adam:       CoLA=5e-4, MNLI=5e-4, MRPC=5e-4, QNLI=1e-4, SST-2=5e-4
# rsLoRA-Muon-PE:   CoLA=1e-3, MNLI=5e-4, MRPC=1e-3, QNLI=5e-4, SST-2=5e-4
# LoRA-One-Muon-PE: CoLA=2e-3, MNLI=1e-3, MRPC=2e-3, QNLI=1e-3, SST-2=5e-4
# PiSSA-Muon-PE:    CoLA=5e-4, MNLI=5e-4, MRPC=1e-3, QNLI=5e-4, SST-2=5e-4
# AdaLoRA-Adam:     CoLA=5e-3, MNLI=1e-3, MRPC=5e-3, QNLI=1e-2, SST-2=2e-3
# LoRA-Pro-Adam:    CoLA=5e-4, MNLI=5e-4, MRPC=1e-3, QNLI=1e-4, SST-2=1e-4
# LoRA-RITE-Adam:   CoLA=1e-3, MNLI=2e-3, MRPC=1e-3, QNLI=1e-3, SST-2=1e-3
# DoRA-Adam:        CoLA=1e-3, MNLI=5e-4, MRPC=2e-3, QNLI=5e-4, SST-2=2e-3

# ============ rsLoRA-Adam ============
add_task "rslora_adamw" "cola" "5e-4"
add_task "rslora_adamw" "mnli" "5e-4"
add_task "rslora_adamw" "mrpc" "2e-3"
add_task "rslora_adamw" "qnli" "1e-3"
add_task "rslora_adamw" "sst2" "5e-4"

# ============ LoRA-One-Adam (rs) ============
add_task "loraone_adamw_rs" "cola" "5e-4"
add_task "loraone_adamw_rs" "mnli" "1e-3"
add_task "loraone_adamw_rs" "mrpc" "1e-3"
add_task "loraone_adamw_rs" "qnli" "5e-4"
add_task "loraone_adamw_rs" "sst2" "1e-3"

# ============ PiSSA-Adam ============
add_task "pissa_adamw" "cola" "5e-4"
add_task "pissa_adamw" "mnli" "5e-4"
add_task "pissa_adamw" "mrpc" "5e-4"
add_task "pissa_adamw" "qnli" "1e-4"
add_task "pissa_adamw" "sst2" "5e-4"

# ============ rsLoRA-Muon-PE ============
add_task "lora_muon_pe_rs" "cola" "1e-3"
add_task "lora_muon_pe_rs" "mnli" "5e-4"
add_task "lora_muon_pe_rs" "mrpc" "1e-3"
add_task "lora_muon_pe_rs" "qnli" "5e-4"
add_task "lora_muon_pe_rs" "sst2" "5e-4"

# ============ LoRA-One-Muon-PE (rs) ============
add_task "loraone_muon_pe_rs" "cola" "2e-3"
add_task "loraone_muon_pe_rs" "mnli" "1e-3"
add_task "loraone_muon_pe_rs" "mrpc" "2e-3"
add_task "loraone_muon_pe_rs" "qnli" "1e-3"
add_task "loraone_muon_pe_rs" "sst2" "5e-4"

# ============ PiSSA-Muon-PE ============
add_task "pissa_muon_pe" "cola" "5e-4"
add_task "pissa_muon_pe" "mnli" "5e-4"
add_task "pissa_muon_pe" "mrpc" "1e-3"
add_task "pissa_muon_pe" "qnli" "5e-4"
add_task "pissa_muon_pe" "sst2" "5e-4"
# ============ AdaLoRA-Adam ============
add_task "adalora_adamw" "cola" "5e-3"
add_task "adalora_adamw" "mnli" "1e-3"
add_task "adalora_adamw" "mrpc" "5e-3"
add_task "adalora_adamw" "qnli" "1e-2"
add_task "adalora_adamw" "sst2" "2e-3"

# ============ LoRA-Pro-Adam (rs) ============
add_task "lorapro_adamw_rs" "cola" "5e-4"
add_task "lorapro_adamw_rs" "mnli" "5e-4"
add_task "lorapro_adamw_rs" "mrpc" "1e-3"
add_task "lorapro_adamw_rs" "qnli" "1e-4"
add_task "lorapro_adamw_rs" "sst2" "1e-4"

# ============ LoRA-RITE-Adam ============
add_task "lorarite_adamw" "cola" "1e-3"
add_task "lorarite_adamw" "mnli" "2e-3"
add_task "lorarite_adamw" "mrpc" "1e-3"
add_task "lorarite_adamw" "qnli" "1e-3"
add_task "lorarite_adamw" "sst2" "1e-3"

# ============ DoRA-Adam ============
add_task "dora_adamw" "cola" "1e-3"
add_task "dora_adamw" "mnli" "5e-4"
add_task "dora_adamw" "mrpc" "2e-3"
add_task "dora_adamw" "qnli" "5e-4"
add_task "dora_adamw" "sst2" "2e-3"

total_tasks=${#task_list[@]}
echo "Total tasks: $total_tasks"

# =============================================================================
# GPU Worker Function
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
