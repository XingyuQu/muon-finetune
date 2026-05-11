#!/bin/bash
# CLIP ViT-B/32 best-lr experiments (queue-based, multi-node safe)
#
# Uses "train-test" split mode with pre-determined best learning rates.
# Loop order: dataset -> variant -> seed
#
# Usage:
#   bash scripts/run_bestlr.sh
#   RESET_QUEUE=1 bash scripts/run_bestlr.sh
#   GPU_IDS="0 1 2 3" bash scripts/run_bestlr.sh

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# =============================================================================
# Environment
# =============================================================================
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export TRANSFORMERS_NO_TF=${TRANSFORMERS_NO_TF:-1}

# AMD ROCm: redirect MIOpen's per-user cache out of /tmp (where shared
# clusters often have cross-user-locked files). No-op on NVIDIA.
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-$MIOPEN_USER_DB_PATH}
mkdir -p "$MIOPEN_USER_DB_PATH"

# =============================================================================
# Configurable Parameters
# =============================================================================

# Data
DATASETS=${DATASETS:-"stanford_cars resisc45 sun397 svhn gtsrb dtd"}
DATA_ROOT=${DATA_ROOT:-"./data"}
SAVE_ROOT_BASE=${SAVE_ROOT_BASE:-"./runs_clip_std"}

# Training
SEEDS=${SEEDS:-"0 1 2"}
NUM_EPOCHS=${NUM_EPOCHS:-40}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LOG_INTERVAL=${LOG_INTERVAL:-5}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-""}
AMP=${AMP:-"bf16"}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}

# LoRA (alpha = 2 * r, fixed)
LORA_R=${LORA_R:-8}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj v_proj"}
LORA_VISUAL_PROJECTION=${LORA_VISUAL_PROJECTION:-1}

# Muon optimizer
MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
MUON_BACKEND=${MUON_BACKEND:-"newtonschulz5"}
MUON_BACKEND_STEPS=${MUON_BACKEND_STEPS:-5}
NS_DTYPE=${NS_DTYPE:-"bf16"}

# Experiment variants (6 combinations of init_method x optimizer)
VARIANTS=${VARIANTS:-"lora_adamw lora_muon lora_muon_pe"}  # full_adamw full_muon full_muon_pe

# Best LR per (variant, dataset) — from sweep experiments
# Format: "dataset1=lr1 dataset2=lr2 ..."
BESTLR_FULL_ADAMW=${BESTLR_FULL_ADAMW:-"stanford_cars=1e-5 resisc45=1e-5 sun397=1e-5 svhn=1e-5 gtsrb=1e-5 dtd=1e-5"}
BESTLR_FULL_MUON=${BESTLR_FULL_MUON:-"stanford_cars=5e-5 resisc45=1e-5 sun397=1e-5 svhn=5e-5 gtsrb=1e-5 dtd=1e-5"}
BESTLR_FULL_MUON_PE=${BESTLR_FULL_MUON_PE:-"stanford_cars=5e-5 resisc45=1e-5 sun397=1e-5 svhn=5e-5 gtsrb=1e-5 dtd=1e-5"}
BESTLR_LORA_ADAMW=${BESTLR_LORA_ADAMW:-"stanford_cars=1e-3 resisc45=1e-3 sun397=5e-4 svhn=1e-3 gtsrb=1e-3 dtd=1e-3"}
BESTLR_LORA_MUON=${BESTLR_LORA_MUON:-"stanford_cars=1e-3 resisc45=1e-3 sun397=5e-4 svhn=1e-3 gtsrb=5e-4 dtd=1e-3"}
BESTLR_LORA_MUON_PE=${BESTLR_LORA_MUON_PE:-"stanford_cars=1e-3 resisc45=1e-3 sun397=5e-4 svhn=1e-3 gtsrb=1e-3 dtd=1e-3"}

# Wandb
WANDB_PROJECT=${WANDB_PROJECT:-"clip_vit_bestlr"}
WANDB_MODE=${WANDB_MODE:-"online"}
WANDB_GROUP=${WANDB_GROUP:-"default"}

# =============================================================================
# GPU Scheduling (SLURM / CUDA_VISIBLE_DEVICES / manual)
# =============================================================================

NODE_NAME=${NODE_NAME:-$(hostname)}
NUM_GPUS=${NUM_GPUS:-8}
GPU_IDS_RAW=${GPU_IDS-}
CVD_RAW="${CUDA_VISIBLE_DEVICES:-}"
CVD_ARR=()
if [[ -n "$CVD_RAW" ]]; then
  CVD_RAW="${CVD_RAW//,/ }"
  read -r -a CVD_ARR <<< "$CVD_RAW"
fi
if [[ -z "$GPU_IDS_RAW" && -n "$CVD_RAW" ]]; then
  GPU_IDS_RAW="$CVD_RAW"
fi

# SLURM auto-detection: parse GPU IDs from SLURM env vars
if [[ -z "$GPU_IDS_RAW" ]]; then
  SLURM_GPU_RAW="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
  if [[ -z "$SLURM_GPU_RAW" && -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    SLURM_GPU_RAW=$(seq -s, 0 $((SLURM_GPUS_ON_NODE - 1)))
  fi
  if [[ -n "$SLURM_GPU_RAW" ]]; then
    # Strip prefixes and expand ranges (e.g. "0-3" -> "0 1 2 3")
    SLURM_GPU_RAW="${SLURM_GPU_RAW//gpu:/}"
    SLURM_GPU_RAW="${SLURM_GPU_RAW//GPU:/}"
    SLURM_GPU_RAW="${SLURM_GPU_RAW// /}"
    expanded=()
    IFS=',' read -r -a tokens <<< "$SLURM_GPU_RAW"
    for tok in "${tokens[@]}"; do
      if [[ "$tok" =~ ^[0-9]+-[0-9]+$ ]]; then
        start="${tok%-*}"; end="${tok#*-}"
        for ((i=start; i<=end; i++)); do expanded+=("$i"); done
      elif [[ "$tok" =~ ^[0-9]+$ ]]; then
        expanded+=("$tok")
      fi
    done
    if [[ "${#expanded[@]}" -gt 0 ]]; then
      GPU_IDS_RAW="${expanded[*]}"
    fi
  fi
fi

# Map GPU IDs to physical devices (handles CVD remapping)
if [[ -n "$GPU_IDS_RAW" ]]; then
  GPU_IDS_RAW="${GPU_IDS_RAW//,/ }"
  read -r -a GPU_ID_ARR <<< "$GPU_IDS_RAW"
  if [[ "${#CVD_ARR[@]}" -gt 0 ]]; then
    map_local=true
    for id in "${GPU_ID_ARR[@]}"; do
      if [[ ! "$id" =~ ^[0-9]+$ ]] || (( id >= ${#CVD_ARR[@]} )); then
        map_local=false; break
      fi
    done
    if [[ "$map_local" == "true" ]]; then
      mapped=()
      for id in "${GPU_ID_ARR[@]}"; do mapped+=("${CVD_ARR[$id]}"); done
      GPU_ID_ARR=("${mapped[@]}")
    fi
  fi
else
  # Fallback: sequential GPU IDs 0..NUM_GPUS-1
  GPU_ID_ARR=()
  if [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] && [[ "$NUM_GPUS" -gt 0 ]]; then
    for ((i=0; i<NUM_GPUS; i++)); do GPU_ID_ARR+=("$i"); done
  fi
fi
if [[ "${#GPU_ID_ARR[@]}" -eq 0 ]]; then
  echo "[error] No GPU IDs configured. Set GPU_IDS or NUM_GPUS." >&2
  exit 1
fi

# Auto-tune NUM_WORKERS based on CPU count / GPU count
CPU_TOTAL=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-}}
if [[ -z "${CPU_TOTAL}" ]]; then CPU_TOTAL=$(nproc); fi
GPU_COUNT=${#GPU_ID_ARR[@]}
if [[ -z "${NUM_WORKERS}" ]]; then
  per_proc=$((CPU_TOTAL / GPU_COUNT))
  if [[ $per_proc -lt 2 ]]; then per_proc=2; fi
  if [[ $per_proc -gt 8 ]]; then per_proc=8; fi
  NUM_WORKERS=$per_proc
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# =============================================================================
# Queue (file-based with flock, multi-node safe)
# =============================================================================

QUEUE_DIR=${QUEUE_DIR:-"$PWD/.queue_std_vitb32_bestlr"}
QUEUE_FILE=${QUEUE_FILE:-"$QUEUE_DIR/tasks.queue"}
LOCK_FILE=${LOCK_FILE:-"$QUEUE_DIR/lock"}
RESET_QUEUE=${RESET_QUEUE:-0}
WORKER_STAT_DIR=${WORKER_STAT_DIR:-"$QUEUE_DIR/worker_stats"}

log_dir="logs/std_vitb32_bestlr"
mkdir -p "$log_dir"

# =============================================================================
# LR Lookup
# =============================================================================

is_full_variant() {
  [[ "$1" == full_* ]]
}

# Lookup key in "k1=v1 k2=v2 ..." spec string
_lookup_in_spec() {
  local spec="$1" key="$2"
  for kv in $spec; do
    local k="${kv%%=*}" v="${kv#*=}"
    if [[ "$k" == "$key" ]]; then echo "$v"; return 0; fi
  done
  return 1
}

get_best_lr_for_variant_dataset() {
  local variant="$1" ds="$2" spec=""
  case "$variant" in
    full_adamw)   spec="${BESTLR_FULL_ADAMW}" ;;
    full_muon)    spec="${BESTLR_FULL_MUON}" ;;
    full_muon_pe) spec="${BESTLR_FULL_MUON_PE}" ;;
    lora_adamw)   spec="${BESTLR_LORA_ADAMW}" ;;
    lora_muon)    spec="${BESTLR_LORA_MUON}" ;;
    lora_muon_pe) spec="${BESTLR_LORA_MUON_PE}" ;;
    *) echo "Unknown variant: $variant" >&2; return 1 ;;
  esac

  if [[ -z "${spec// }" ]]; then
    echo "Missing BESTLR table for variant=$variant. Set BESTLR_* env vars." >&2
    return 1
  fi

  _lookup_in_spec "$spec" "$ds" \
    || { echo "Missing best lr for variant=$variant dataset=$ds" >&2; return 1; }
}

# =============================================================================
# Command Builder
# =============================================================================

build_cmd() {
  local variant="$1" ds="$2" lr="$3" seed="$4"
  local save_root="${SAVE_ROOT_BASE}_${ds}"
  local run_name="${variant}_${ds}_lr${lr}_seed${seed}"
  local lora_alpha=$((LORA_R * 2))

  local base_args="--dataset ${ds} \
        --data-root ${DATA_ROOT} \
        --save-root ${save_root} \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --split-mode train-test \
        --num-epochs ${NUM_EPOCHS} \
        --warmup-ratio ${WARMUP_RATIO} \
        --eval-interval 99999 \
        --log-interval ${LOG_INTERVAL} \
        --max-grad-norm ${MAX_GRAD_NORM} \
        --amp ${AMP} \
        --lr ${lr} \
        --wd ${WEIGHT_DECAY} \
        --seed ${seed} \
        --wandb-project ${WANDB_PROJECT} \
        --wandb-mode ${WANDB_MODE} \
        --wandb-group ${WANDB_GROUP} \
        --wandb-name ${run_name}"

  local optimizer_args="--muon-momentum ${MUON_MOMENTUM} \
        --muon-backend ${MUON_BACKEND} \
        --muon-backend-steps ${MUON_BACKEND_STEPS} \
        --ns-dtype ${NS_DTYPE}"

  local lora_args="--lora-r ${LORA_R} \
        --lora-alpha ${lora_alpha} \
        --lora-dropout ${LORA_DROPOUT} \
        --lora-target-modules ${LORA_TARGET_MODULES}"
  if [[ "${LORA_VISUAL_PROJECTION}" == "1" || "${LORA_VISUAL_PROJECTION}" == "true" ]]; then
    lora_args+=" --lora-visual-projection"
  fi

  local init_method="" optimizer="" extra_flags=""
  case "$variant" in
    full_adamw)   init_method="full_ft"; optimizer="adamw" ;;
    full_muon)    init_method="full_ft"; optimizer="muon" ;;
    full_muon_pe) init_method="full_ft"; optimizer="muon"; extra_flags="--ns-using-pe" ;;
    lora_adamw)   init_method="lora";    optimizer="adamw" ;;
    lora_muon)    init_method="lora";    optimizer="muon" ;;
    lora_muon_pe) init_method="lora";    optimizer="muon"; extra_flags="--ns-using-pe" ;;
    *) echo "Unknown variant: $variant" >&2; return 1 ;;
  esac

  local launch_cmd="python -m clip_vit.train.main"

  if is_full_variant "$variant"; then
    echo "${launch_cmd} \
      --init-method ${init_method} \
      --optimizer ${optimizer} \
      ${optimizer_args} \
      ${extra_flags} \
      ${base_args}"
  else
    echo "${launch_cmd} \
      --init-method ${init_method} \
      --optimizer ${optimizer} \
      ${lora_args} \
      ${optimizer_args} \
      ${extra_flags} \
      ${base_args}"
  fi
}

# =============================================================================
# Queue Functions (flock-based, multi-node safe)
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
      # Write task indices (one per line) to queue file
      : > "$QUEUE_FILE"
      for i in "${!task_list[@]}"; do
        echo "$i" >> "$QUEUE_FILE"
      done
      echo "initialized" > "$QUEUE_DIR/initialized"
      echo "[$NODE_NAME] Queue initialized with $total_tasks tasks"
    else
      remaining=$(wc -l < "$QUEUE_FILE" 2>/dev/null || echo "0")
      echo "[$NODE_NAME] Queue already exists, $remaining tasks remaining"
    fi
  ) 200>"$LOCK_FILE"
}

# Atomic pop: read first line, remove it from queue
get_next_task() {
  (
    flock -x 200
    if [[ -s "$QUEUE_FILE" ]]; then
      head -n 1 "$QUEUE_FILE"
      tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp" && mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
    fi
  ) 200>"$LOCK_FILE"
}

# Read-only count (shared lock)
get_remaining_tasks() {
  (
    flock -s 200
    wc -l < "$QUEUE_FILE" 2>/dev/null || echo "0"
  ) 200>"$LOCK_FILE"
}

# =============================================================================
# Generate Task List (dataset -> variant -> seed)
# =============================================================================

task_list=()
task_names=()

for ds in $DATASETS; do
  for variant in $VARIANTS; do
    lr=$(get_best_lr_for_variant_dataset "$variant" "$ds") || exit 1
    for seed in $SEEDS; do
      cmd=$(build_cmd "$variant" "$ds" "$lr" "$seed")
      task_list+=("$cmd")
      task_names+=("${variant}_${ds}_lr${lr}_seed${seed}")
    done
  done
done

total_tasks=${#task_list[@]}

# =============================================================================
# Print Configuration Summary
# =============================================================================

echo "=============================================="
echo "CLIP ViT-B/32 Standard Best-LR Configuration"
echo "=============================================="
echo "Node:       ${NODE_NAME}"
echo "GPUs:       ${GPU_ID_ARR[*]}"
echo "Datasets:   ${DATASETS}"
echo "Seeds:      ${SEEDS}"
echo "Training:   ${NUM_EPOCHS} epochs"
echo "Batch:      ${BATCH_SIZE}"
echo "Workers:    ${NUM_WORKERS} (CPU total=${CPU_TOTAL}, GPUs=${GPU_COUNT})"
echo "Variants:   ${VARIANTS}"
echo "LoRA:       r=${LORA_R}, alpha=$((LORA_R * 2)) (2*r), dropout=${LORA_DROPOUT}"
echo "Muon:       backend=${MUON_BACKEND}, ns_dtype=${NS_DTYPE}"
echo "Split mode: train-test (no validation set)"
echo "Wandb:      project=${WANDB_PROJECT}, mode=${WANDB_MODE}, group=${WANDB_GROUP}"
echo "BestLR FULL_ADAMW:   ${BESTLR_FULL_ADAMW}"
echo "BestLR FULL_MUON:    ${BESTLR_FULL_MUON}"
echo "BestLR FULL_MUON_PE: ${BESTLR_FULL_MUON_PE}"
echo "BestLR LORA_ADAMW:   ${BESTLR_LORA_ADAMW}"
echo "BestLR LORA_MUON:    ${BESTLR_LORA_MUON}"
echo "BestLR LORA_MUON_PE: ${BESTLR_LORA_MUON_PE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM GPU:  step=${SLURM_STEP_GPUS:-<unset>} job=${SLURM_JOB_GPUS:-<unset>} gpus_on_node=${SLURM_GPUS_ON_NODE:-<unset>}"
echo "Queue dir:  ${QUEUE_DIR}"
echo "Total tasks: $total_tasks"
echo "=============================================="

if [[ $total_tasks -eq 0 ]]; then
  echo "No tasks to run. Check your configuration."
  exit 0
fi

# =============================================================================
# Launch Workers (one per GPU)
# =============================================================================

init_queue

mkdir -p "$WORKER_STAT_DIR"
echo "[$NODE_NAME] Starting workers on GPUs: ${GPU_ID_ARR[*]}"

worker() {
  local gpu_id="$1"
  local stat_file="$2"
  local completed=0
  local failed=0
  local worker_tag="${NODE_NAME}/gpu${gpu_id}"

  while true; do
    task_idx=$(get_next_task)
    if [[ -z "$task_idx" ]]; then
      echo "[$worker_tag] No more tasks in queue, exiting."
      break
    fi

    cmd="${task_list[$task_idx]}"
    name="${task_names[$task_idx]}"
    log_file="${log_dir}/${name}.log"
    remaining=$(get_remaining_tasks)

    echo ""
    echo "[$worker_tag] ================================================"
    echo "[$worker_tag] Starting: $name"
    echo "[$worker_tag] Task index: $task_idx, Remaining in queue: $remaining"
    echo "[$worker_tag] Log: $log_file"
    echo "[$worker_tag] ================================================"

    start_time=$(date +%s)

    if CUDA_VISIBLE_DEVICES="$gpu_id" bash -c "$cmd" 2>&1 | tee "$log_file"; then
      end_time=$(date +%s)
      duration=$((end_time - start_time))
      echo "[$worker_tag] Finished: $name (took ${duration}s)"
      ((completed++))
    else
      end_time=$(date +%s)
      duration=$((end_time - start_time))
      echo "[$worker_tag] FAILED: $name (took ${duration}s)"
      ((failed++))
    fi
  done

  # Write stats for aggregation
  echo "$completed $failed" > "$stat_file"
}

# Spawn one worker per GPU
worker_pids=()
for gpu_id in "${GPU_ID_ARR[@]}"; do
  stat_file="${WORKER_STAT_DIR}/stat_${NODE_NAME}_gpu${gpu_id}.txt"
  rm -f "$stat_file"
  worker "$gpu_id" "$stat_file" &
  worker_pids+=("$!")
done

# Wait for all workers to finish
for pid in "${worker_pids[@]}"; do
  wait "$pid"
done

# Aggregate stats across all GPUs
total_completed=0
total_failed=0
for gpu_id in "${GPU_ID_ARR[@]}"; do
  stat_file="${WORKER_STAT_DIR}/stat_${NODE_NAME}_gpu${gpu_id}.txt"
  c=0
  f=0
  if [[ -f "$stat_file" ]]; then
    read -r c f < "$stat_file" || { c=0; f=0; }
  fi
  total_completed=$((total_completed + c))
  total_failed=$((total_failed + f))
done

echo ""
echo "[$NODE_NAME] ================================================"
echo "[$NODE_NAME] Workers finished"
echo "[$NODE_NAME] Completed: $total_completed, Failed: $total_failed"
echo "[$NODE_NAME] ================================================"
