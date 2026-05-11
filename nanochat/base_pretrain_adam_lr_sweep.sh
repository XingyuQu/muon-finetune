#!/usr/bin/env bash
set -euo pipefail

# Adam pretraining with a configurable matrix learning rate (the paper's
# d20_adam_lr0.001 model is produced by the default MATRIX_LR=0.001).
#
# Self-contained: this script bootstraps the tokenizer, dataset shards,
# and CORE eval_bundle on first run, and skips each step on subsequent runs.
# It can therefore be invoked either standalone or after base_pretrain_muon.sh
# (in which case all the setup steps are no-ops).
#
# Usage:
#   # Single run with the paper-tuned LR
#   SKIP_UV=1 NPROC_PER_NODE=8 bash base_pretrain_adam_lr_sweep.sh
#
#   # Sweep across nodes
#   SKIP_UV=1 NPROC_PER_NODE=8 MATRIX_LR=0.03 bash base_pretrain_adam_lr_sweep.sh
#   SKIP_UV=1 NPROC_PER_NODE=8 MATRIX_LR=0.04 bash base_pretrain_adam_lr_sweep.sh
#
# Outputs:
#   - checkpoint: $NANOCHAT_BASE_DIR/base_checkpoints/d20_adam_lr{MATRIX_LR}/
#   - report:     $NANOCHAT_BASE_DIR/report_d20_adam_lr{MATRIX_LR}/

# ------------------------------ Configuration ------------------------------
: "${NPROC_PER_NODE:=8}"
: "${NANOCHAT_BASE_DIR:=$HOME/.cache/nanochat}"
: "${MATRIX_LR:=0.001}"   # paper default; override per node for an LR sweep
export NANOCHAT_BASE_DIR
export OMP_NUM_THREADS=1
mkdir -p "$NANOCHAT_BASE_DIR"

DEPTH=20
DEVICE_BATCH_SIZE=16
DATA_SHARDS=240
TOK_MAX_CHARS=2000000000

MODEL_TAG="d${DEPTH}_adam_lr${MATRIX_LR}"
export REPORT_TAG="$MODEL_TAG"
WANDB_RUN="${WANDB_RUN:-$MODEL_TAG}"

echo ""
echo "############################################################"
echo "# Pretraining Adam with matrix_lr=${MATRIX_LR}"
echo "# MODEL_TAG: ${MODEL_TAG}"
echo "# NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "############################################################"

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

# ------------------------------ Tokenizer / data bootstrap (idempotent) ------------------------------

# Reset the report for this MODEL_TAG
python -m nanochat.report reset --tag="$MODEL_TAG"

# Rust toolchain (needed only to build rustbpe)
command -v cargo &> /dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

# rustbpe: skip rebuild if the extension already imports
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

# Dataset shards: nanochat.dataset itself is idempotent (only fetches missing shards)
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n "$DATA_SHARDS" &
DATASET_DOWNLOAD_PID=$!

# Tokenizer: skip training if the artifact already exists
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Training tokenizer (one-time setup)..."
    python -m scripts.tok_train --max_chars="$TOK_MAX_CHARS"
    python -m scripts.tok_eval
else
    echo "Tokenizer already trained at $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl, skipping."
fi

# CORE eval bundle (~162MB)
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
    curl -L -o eval_bundle.zip "$EVAL_BUNDLE_URL"
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle "$NANOCHAT_BASE_DIR"
fi

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# ------------------------------ Training + eval ------------------------------

echo "==================== [TRAIN] ${MODEL_TAG} ===================="
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_adam_train -- \
    --run="$WANDB_RUN" \
    --model_tag="$MODEL_TAG" \
    --depth="$DEPTH" \
    --device_batch_size="$DEVICE_BATCH_SIZE" \
    --matrix_lr="$MATRIX_LR"

echo "==================== [LOSS EVAL] ${MODEL_TAG} ===================="
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_loss -- \
    --model_tag="$MODEL_TAG"

echo "==================== [CORE EVAL] ${MODEL_TAG} ===================="
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model_tag="$MODEL_TAG"

echo "==================== [REPORT] ${MODEL_TAG} ===================="
python -m nanochat.report generate --tag="$MODEL_TAG"

echo ""
echo "############################################################"
echo "# Completed: ${MODEL_TAG}"
echo "# Checkpoint: ${NANOCHAT_BASE_DIR}/base_checkpoints/${MODEL_TAG}/"
echo "# Report:     ${NANOCHAT_BASE_DIR}/report_${MODEL_TAG}/"
echo "############################################################"
