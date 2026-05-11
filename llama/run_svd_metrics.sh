#!/bin/bash
# Compute SVD metrics (svd_entropy, stable_rank) on LoRA checkpoint(s).
#
# Usage:
#   bash run_svd_metrics.sh <path>
#
# <path> is one of:
#   - a single checkpoint dir (with adapter_config.json, no checkpoint-*/ children)
#   - an experiment dir (with one or more checkpoint-<N>/ children); each gets processed
#
# Optional env:
#   NUM_GPUS=N           intra-node GPU parallelism (default 8)
#   OUTPUT_DIR=...       where to write metrics_*.json/.log (default metrics_logs/metrics)
#   METRICS=...          --metrics value     (default svd_entropy,stable_rank)
#   DTYPE=...            --dtype value       (default bf16)
#   SVD_GROUPS=...       --svd-groups value  (default all 6 LoRA groups)

set -u
cd "$(dirname "$0")"

# =============================================================================
# Args / config
# =============================================================================

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $(basename "$0") <checkpoint_or_experiment_dir>" >&2
  exit 1
fi
INPUT_PATH="$1"

METRICS_SCRIPT="metrics/compute_svd_metrics.py"
OUTPUT_DIR=${OUTPUT_DIR:-"metrics_logs/metrics"}
METRICS=${METRICS:-"svd_entropy,stable_rank"}
DTYPE=${DTYPE:-"bf16"}
SVD_GROUPS=${SVD_GROUPS:-"AttnQO_A,AttnQO_B,AttnKV_A,AttnKV_B,Dense_A,Dense_B"}
NUM_GPUS=${NUM_GPUS:-8}

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Collect tasks
# =============================================================================

if [[ ! -d "$INPUT_PATH" ]]; then
  echo "Error: $INPUT_PATH is not a directory" >&2
  exit 1
fi

TASKS=()  # each entry: "<ckpt>|<out_file>|<log_file>"

# Queue one (ckpt, exp_name, step) tuple; skip if its output JSON already exists.
add_task() {
  local ckpt="$1" exp_name="$2" step="$3"
  local out_file="${OUTPUT_DIR}/metrics_${exp_name}_step${step}.json"
  local log_file="${OUTPUT_DIR}/metrics_${exp_name}_step${step}.log"
  if [[ -f "$out_file" ]]; then
    echo "Skip (exists): $(basename "$out_file")"
    return
  fi
  TASKS+=("${ckpt}|${out_file}|${log_file}")
}

# Detect mode: prefer experiment-dir (multiple checkpoint-*/ children) when both present.
has_subdirs=0
for c in "$INPUT_PATH"/checkpoint-*/; do
  [[ -d "$c" ]] && has_subdirs=1 && break
done

if [[ $has_subdirs -eq 1 ]]; then
  exp_name=$(basename "${INPUT_PATH%/}")
  for ckpt in "$INPUT_PATH"/checkpoint-*/; do
    [[ -d "$ckpt" ]] || continue
    step=$(basename "${ckpt%/}" | sed 's/checkpoint-//')
    add_task "${ckpt%/}" "$exp_name" "$step"
  done
elif [[ -f "$INPUT_PATH/adapter_config.json" ]]; then
  ckpt_base=$(basename "${INPUT_PATH%/}")
  if [[ "$ckpt_base" == checkpoint-* ]]; then
    step="${ckpt_base#checkpoint-}"
    exp_name=$(basename "$(dirname "${INPUT_PATH%/}")")
  else
    step="final"
    exp_name="$ckpt_base"
  fi
  add_task "${INPUT_PATH%/}" "$exp_name" "$step"
else
  echo "Error: no adapter_config.json and no checkpoint-*/ found under $INPUT_PATH" >&2
  exit 1
fi

total=${#TASKS[@]}

echo "=============================================="
echo "SVD Metrics"
echo "=============================================="
echo "Input:       $INPUT_PATH"
echo "Tasks:       $total"
echo "GPUs:        $NUM_GPUS"
echo "Metrics:     $METRICS"
echo "SVD groups:  $SVD_GROUPS"
echo "Dtype:       $DTYPE"
echo "Output dir:  $OUTPUT_DIR"
echo "=============================================="

if [[ $total -eq 0 ]]; then
  echo "Nothing to compute."
  exit 0
fi

# =============================================================================
# Run with intra-node GPU parallelism
# =============================================================================

running=0
completed=0
failed=0
pids=()
pid_info=()

# Poll the pid table; if any child has exited, reap it and update counters. Returns 0 on reap.
reap_one_done() {
  for i in "${!pids[@]}"; do
    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
      wait "${pids[$i]}" 2>/dev/null
      local exit_code=$?
      if [[ $exit_code -eq 0 ]]; then
        ((completed++))
      else
        echo "FAILED: ${pid_info[$i]}"
        ((failed++))
      fi
      unset 'pids[i]' 'pid_info[i]'
      pids=("${pids[@]}")
      pid_info=("${pid_info[@]}")
      ((running--))
      return 0
    fi
  done
  return 1
}

# Pick the lowest GPU id in [0, NUM_GPUS) not currently held by a running child.
pick_free_gpu() {
  local used=" "
  for i in "${!pids[@]}"; do
    used="${used}${pid_info[$i]##*GPU=} "
  done
  local g
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    if [[ "$used" != *" $g "* ]]; then
      echo "$g"
      return
    fi
  done
  echo 0
}

for entry in "${TASKS[@]}"; do
  while [[ $running -ge $NUM_GPUS ]]; do
    if ! reap_one_done; then sleep 0.5; fi
  done

  IFS='|' read -r ckpt out_file log_file <<< "$entry"
  gpu=$(pick_free_gpu)
  ckpt_label="$(basename "$(dirname "$ckpt")")/$(basename "$ckpt")"
  echo "GPU=$gpu | $ckpt_label"

  CUDA_VISIBLE_DEVICES=$gpu python3 "$METRICS_SCRIPT" \
    --checkpoint "$ckpt" \
    --metrics "$METRICS" \
    --dtype "$DTYPE" \
    --output "$out_file" \
    --log-file "$log_file" \
    --svd-groups "$SVD_GROUPS" \
    &

  pids+=($!)
  pid_info+=("$ckpt_label GPU=$gpu")
  ((running++))
done

# Wait for the rest
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" 2>/dev/null
  if [[ $? -eq 0 ]]; then
    ((completed++))
  else
    echo "FAILED: ${pid_info[$i]}"
    ((failed++))
  fi
done

echo ""
echo "=============================================="
echo "Done. Completed: $completed | Failed: $failed"
echo "=============================================="
