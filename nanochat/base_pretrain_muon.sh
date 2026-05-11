#!/bin/bash

# Muon pretraining of the d20 base model (nanochat speedrun-style: tokenizer
# training + FineWeb-Edu shard download + base_train + base_loss + base_eval +
# report). Produces the d20_muon checkpoint used as the Muon-pretrained
# starting point for the WikiText fine-tuning experiments.
#
# 1) Example launch (simplest):
# bash base_pretrain_muon.sh
# 2) Example launch in a screen session (the full run takes a few hours):
# screen -L -Logfile pretrain.log -S pretrain bash base_pretrain_muon.sh
# 3) Example launch with wandb logging (see below for setting up wandb first):
# WANDB_RUN=d20_muon screen -L -Logfile pretrain.log -S pretrain bash base_pretrain_muon.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
# (override by passing NANOCHAT_BASE_DIR=... before this script).
export OMP_NUM_THREADS=1
# export HF_HOME="/volume/hf_cache"  # uncomment / edit to redirect the HuggingFace cache
: "${NANOCHAT_BASE_DIR:=$HOME/.cache/nanochat}"
export NANOCHAT_BASE_DIR
# WANDB_RUN will be set after MODEL_TAG is determined (see below)
mkdir -p "$NANOCHAT_BASE_DIR"

# Number of GPUs (torchrun --nproc_per_node); default 8, override via env var:
#   NPROC_PER_NODE=4 bash base_pretrain_muon.sh
# or pass it as the first positional argument:
#   bash base_pretrain_muon.sh 4
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [ "${1:-}" != "" ]; then
    NPROC_PER_NODE="$1"
fi

# -----------------------------------------------------------------------------
# Debug mode: smoke-test the pipeline with a tiny model and few iterations.
# Usage: DEBUG=1 bash base_pretrain_muon.sh
# Debug config: depth=4, 20 iterations, CORE evaluation skipped.
if [ "${DEBUG:-0}" = "1" ]; then
    echo ">>> DEBUG MODE ENABLED <<<"
    MODEL_TAG="debug_muon"
    TRAIN_ARGS="--depth=4 --max_seq_len=64 --device_batch_size=1 --total_batch_size=512 --num_iterations=20 --eval_tokens=512 --core_metric_every=-1"
    DATA_SHARDS=2
    TOK_MAX_CHARS=10000000
else
    MODEL_TAG="d20_muon"
    TRAIN_ARGS="--depth=20 --device_batch_size=16"
    DATA_SHARDS=240
    TOK_MAX_CHARS=2000000000
fi
# Export REPORT_TAG so get_report() in the training scripts picks it up
export REPORT_TAG=$MODEL_TAG
# Set WANDB_RUN (default to MODEL_TAG when not provided externally)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=$MODEL_TAG
fi

# -----------------------------------------------------------------------------
# Python venv setup with uv.
# If you manage your own environment (e.g. conda), set SKIP_UV=1 to skip this.
# Usage: SKIP_UV=1 bash base_pretrain_muon.sh

if [ "${SKIP_UV:-0}" != "1" ]; then
    # install uv (if not already installed)
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    # create a .venv local virtual environment (if it doesn't exist)
    [ -d ".venv" ] || uv venv
    # install the repo dependencies
    uv sync
    # activate venv so that `python` uses the project's venv instead of system python
    source .venv/bin/activate
else
    echo ">>> SKIP_UV=1: Skipping uv setup, using current Python environment <<<"
    # Make sure maturin is installed (used to build rustbpe)
    pip install maturin --quiet
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset --tag=$MODEL_TAG

# -----------------------------------------------------------------------------
# Tokenizer

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

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m nanochat.dataset -n $DATA_SHARDS &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=$TOK_MAX_CHARS
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# pretrain the model
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- --run=$WANDB_RUN --model_tag=$MODEL_TAG $TRAIN_ARGS
# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_loss -- --model_tag=$MODEL_TAG
# evaluate the model on CORE tasks (skip in debug mode due to short sequence length)
if [ "${DEBUG:-0}" != "1" ]; then
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- --model_tag=$MODEL_TAG
else
    echo ">>> Skipping CORE evaluation in debug mode (sequence too short) <<<"
fi

# -----------------------------------------------------------------------------
# Generate the report (tokenizer + base-model training/loss/CORE sections)
python -m nanochat.report generate --tag=$MODEL_TAG
