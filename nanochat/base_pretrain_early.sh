#!/usr/bin/env bash
set -euo pipefail

# Early-step pretraining: run only NUM_ITERATIONS=10 steps and save the
# step-10 checkpoint. The resulting d20_*_early checkpoints feed
# scripts/analysis/compute_early_metrics.py, which adds the leftmost
# "Step 10" point to the SVD figures (paper Figure 3 right).
#
# Usage:
#   bash base_pretrain_early.sh muon                       # -> d20_muon_early
#   bash base_pretrain_early.sh adam                       # -> d20_adam_lr0.001_early
#   MATRIX_LR=0.001 bash base_pretrain_early.sh adam       # explicit Adam matrix LR
#
# The script is self-contained: it reuses the same idempotent setup as
# base_pretrain_muon.sh / base_pretrain_adam_lr_sweep.sh (rustbpe build,
# tokenizer training, FineWeb-Edu shard download) and skips work that is
# already done. Re-running on a finished checkpoint is a no-op.

if [ $# -lt 1 ]; then
    echo "Usage: bash base_pretrain_early.sh <muon|adam>"
    exit 1
fi

OPTIMIZER="$1"
case "$OPTIMIZER" in
    muon|adam) ;;
    *) echo "Unknown optimizer: $OPTIMIZER (expected muon or adam)"; exit 1 ;;
esac

# ------------------------------ Configuration ------------------------------
: "${NPROC_PER_NODE:=8}"
: "${NANOCHAT_BASE_DIR:=$HOME/.cache/nanochat}"
: "${MATRIX_LR:=0.001}"   # only used for the adam branch (matches the paper pretrain)
export NANOCHAT_BASE_DIR
export OMP_NUM_THREADS=1
mkdir -p "$NANOCHAT_BASE_DIR"

DEPTH=20
DEVICE_BATCH_SIZE=16
NUM_ITERATIONS=10
CHECKPOINT_EVERY=10        # save at the very last step (=10)
DATA_SHARDS=240
TOK_MAX_CHARS=2000000000

if [ "$OPTIMIZER" = "muon" ]; then
    MODEL_TAG="d${DEPTH}_muon_early"
    TRAIN_MODULE="scripts.base_train"
    EXTRA_TRAIN_ARGS=()
else
    MODEL_TAG="d${DEPTH}_adam_lr${MATRIX_LR}_early"
    TRAIN_MODULE="scripts.base_adam_train"
    EXTRA_TRAIN_ARGS=(--matrix_lr="$MATRIX_LR")
fi

CHECKPOINT_DIR="${NANOCHAT_BASE_DIR}/base_checkpoints/${MODEL_TAG}"
export REPORT_TAG="$MODEL_TAG"
WANDB_RUN="${WANDB_RUN:-$MODEL_TAG}"

echo ""
echo "############################################################"
echo "# Early-step pretraining: $OPTIMIZER ($NUM_ITERATIONS steps)"
echo "# MODEL_TAG: $MODEL_TAG"
echo "# NPROC_PER_NODE: $NPROC_PER_NODE"
echo "############################################################"

# Skip if the step-10 checkpoint already exists
if [ -f "${CHECKPOINT_DIR}/model_$(printf '%06d' $NUM_ITERATIONS).pt" ]; then
    echo "Checkpoint already exists at ${CHECKPOINT_DIR}/, skipping."
    exit 0
fi

# ------------------------------ Python venv (uv) ------------------------------
if [ "${SKIP_UV:-0}" != "1" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync
    source .venv/bin/activate
else
    echo ">>> SKIP_UV=1: Using current Python environment <<<"
    pip install maturin --quiet
fi

# Reset the report for this MODEL_TAG
python -m nanochat.report reset --tag="$MODEL_TAG"

# ------------------------------ Tokenizer / data bootstrap (idempotent) ------------------------------
command -v cargo &> /dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

if ! python -c "import rustbpe; rustbpe.Tokenizer" 2>/dev/null; then
    echo "Building rustbpe..."
    if [ "${SKIP_UV:-0}" != "1" ]; then
        uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
    else
        # Use `maturin build` + `pip install --no-deps` so we DO NOT trigger pip
        # to resolve the project's pyproject.toml deps (which would reinstall torch
        # from the default CUDA index and clobber a user-managed ROCm/CPU torch).
        maturin build --release --manifest-path rustbpe/Cargo.toml
        pip install --no-deps rustbpe/target/wheels/nanochat-*.whl
    fi
else
    echo "rustbpe already built, skipping."
fi

# Tokenizer (idempotent guard)
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Training tokenizer (one-time setup)..."
    python -m nanochat.dataset -n 8
    python -m scripts.tok_train --max_chars="$TOK_MAX_CHARS"
    python -m scripts.tok_eval
else
    echo "Tokenizer already trained, skipping."
fi

# Dataset shards: nanochat.dataset only fetches missing shards
python -m nanochat.dataset -n "$DATA_SHARDS"

# ------------------------------ Training (10 steps only) ------------------------------
echo "==================== [TRAIN] ${MODEL_TAG} (${NUM_ITERATIONS} steps) ===================="
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m "$TRAIN_MODULE" -- \
    --run="$WANDB_RUN" \
    --model_tag="$MODEL_TAG" \
    --depth="$DEPTH" \
    --device_batch_size="$DEVICE_BATCH_SIZE" \
    --num_iterations="$NUM_ITERATIONS" \
    --checkpoint_every="$CHECKPOINT_EVERY" \
    --eval_every=9999 \
    --core_metric_every=-1 \
    --sample_every=9999 \
    "${EXTRA_TRAIN_ARGS[@]}"

echo ""
echo "############################################################"
echo "# Completed: $MODEL_TAG"
echo "# Checkpoint: $CHECKPOINT_DIR/"
echo "############################################################"
