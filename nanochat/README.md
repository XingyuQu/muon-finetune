# nanochat (paper code)

This repository is the nanochat fork used for the experiments in our paper.
It covers three things:

- pretraining of two 561M-parameter base models (Muon, Adam) on FineWeb-Edu;
- WikiText-2 fine-tuning of those base models in 4 modes
  (`full-adam`, `full-muon`, `lora-adam`, `lora-muon`), 3 seeds each, with a
  learning-rate sweep;
- spectral analysis of attention QKV weights during pretraining and
  fine-tuning.

The original karpathy/nanochat README is preserved verbatim as
[`NANOCHAT_README.md`](NANOCHAT_README.md) for attribution.

## Setup

Pick **one** path based on your GPU.

**CUDA** — use `uv` (the shell scripts will install it for you on first run if needed):

```bash
uv venv && source .venv/bin/activate && uv sync
```

**AMD ROCm** — do **not** use `uv`. Install torch from your own wheel index in a fresh conda env, then the rest of the deps. The shell scripts will install `cargo` (via `rustup`) and build the `rustbpe` Rust extension on first run, taking care not to clobber your torch install.

```bash
conda create -n nanochat python=3.11 -y && conda activate nanochat
pip install torch==2.9.1 pytorch-triton-rocm==3.5.1 \
            --index-url https://download.pytorch.org/whl/rocm6.4
pip install datasets wandb tiktoken tokenizers regex 'numpy==1.26.4' \
            psutil files-to-prompt fastapi uvicorn \
            pyarrow pandas pyyaml safetensors
```

Then run the three pretrain launchers (`base_pretrain_muon.sh`, `base_pretrain_adam_lr_sweep.sh`, `base_pretrain_early.sh`) with `SKIP_UV=1` so they don't try to bootstrap their own `uv` venv on top of your conda env. Example:

```bash
SKIP_UV=1 bash base_pretrain_muon.sh
```

Artefacts (checkpoints, datasets, eval bundles, reports) land in `$NANOCHAT_BASE_DIR`, default `~/.cache/nanochat`. Overwrite it by `NANOCHAT_BASE_DIR=[YOUR_BASE_DIR] SKIP_UV=1 bash base_pretrain_muon.sh`.

## Experiments

Three groups: pretraining, fine-tuning, analysis. Defaults match the paper. Prepend `SKIP_UV=1` if you used the conda path above.

### Pretraining

- `base_pretrain_muon.sh [NPROC]` — Muon pretraining (default `NPROC=8`, matrix LR `0.02`). Produces the `d20_muon` checkpoint.
- `base_pretrain_adam_lr_sweep.sh` — Adam pretraining; default `MATRIX_LR=0.001` matches the paper. The chosen LR is encoded in the model tag (`d20_adam_lr0.001`).
- `base_pretrain_early.sh muon|adam` — 10-step pretraining; produces the `d20_*_early` checkpoints used by the early-step SVD point.

```bash
bash base_pretrain_muon.sh                                            # Muon pretrain → d20_muon
bash base_pretrain_adam_lr_sweep.sh                                   # Adam pretrain at paper's MATRIX_LR=0.001 → d20_adam_lr0.001
MATRIX_LR=0.03 bash base_pretrain_adam_lr_sweep.sh                    # other Adam LRs in the sweep → d20_adam_lr0.03
bash base_pretrain_early.sh muon && bash base_pretrain_early.sh adam  # 10-step early ckpts (for SVD step-10 point)
```

First run of any pretrain bootstraps the shared setup (rustbpe / FineWeb-Edu shards / CORE eval bundle / tokenizer); subsequent runs reuse it. After training, `scripts.base_loss` and `scripts.base_eval` run automatically and a markdown report is written to `report_<MODEL_TAG>/`.

### Fine-tuning (WikiText-2)

- `wikitext_sweep.sh` — full LR sweep over `MODELS × MODES × SEEDS × LRs`:
  ```
  MODELS = (d20_muon, d20_adam_lr0.001)
  MODES  = (full-adam, full-muon, lora-adam, lora-muon)
  SEEDS  = (0, 1, 2)
  LRs    = ADAM_LRS / MUON_LRS / LORA_ADAM_LRS / LORA_MUON_LRS  (env-overridable)
  ```
  Each cell writes `meta_<step>.json` (init/final/best PPL + per-step `eval_log` trajectory + run config) under `wikitext_checkpoints/<run>/`; model weights `model_<step>.pt` are written only with `--save_weights=True`.
- `run_8_bestlr_experiments.sh` — 8 cells (4 modes × 2 pretrains, `seed=0`) at the paper's best LR each, with `--save_weights=False --eval_every=15` for fine-grained trajectories.
- `scripts/wikitext_finetune.py` — single-config runner; invoke directly via `torchrun` if you only want one cell.

```bash
bash wikitext_sweep.sh                    # full LR sweep
bash run_8_bestlr_experiments.sh          # 8 best-LR trajectory runs (paper headline)

# Single configuration (matched: Muon-pretrained + full-Muon, paper best LR)
torchrun --standalone --nproc_per_node=8 -m scripts.wikitext_finetune \
    --mode=full-muon --model_tag=d20_muon --matrix_lr=0.9 --num_epochs=1 --eval_every=15
```

### Analysis (`scripts/analysis/`)

**SVD / spectral metrics on pretrained models.** Run in order — each appends to `analysis_results/svd_results/summary.json`:

```bash
python scripts/analysis/run_svd_metrics.py        # SVD entropy + stable rank (initialises summary.json)
python scripts/analysis/compute_early_metrics.py  # adds the early-step point (needs d20_*_early ckpts)
python scripts/analysis/compute_max_norm.py       # adds max norm
python scripts/analysis/compute_spectral_norm.py  # adds spectral norm + per-ckpt singular value safetensors
```

For the fine-tuned models, run `python scripts/analysis/run_finetune_svd_metrics.py` (requires the 8 best-LR runs to have been saved with `--save_weights=True`); output → `analysis_results/svd_results_finetune/`.

**Aggregation.** `python scripts/analysis/collect_all_seeds.py > wikitext_results.csv` collects per-seed PPL across the WikiText sweep into a single CSV (consumed by `plot_lr_sweep.py`).

**Plotting.**
- `plot_lr_sweep.py` — LR sweep figures (reads `wikitext_results.csv`).
- `plot_trajectory_matched_vs_mismatched.py`, `plot_trajectory_per_pretrain.py` — fine-tune PPL trajectories (reads `wikitext_checkpoints/<run>/meta_*.json`).
- `plot_svd_by_group.py`, `plot_attn_qkv_separate.py` — SVD entropy / stable rank figures (reads `summary.json`).

## Where everything is saved

After running the full pipeline once with default settings:

```
$NANOCHAT_BASE_DIR/                  # default ~/.cache/nanochat
├── tokenizer/
│   ├── tokenizer.pkl                # rustbpe-trained, vocab=65536
│   └── token_bytes.pt
├── base_data/                       # FineWeb-Edu shards (~22 GB / 240 shards)
├── eval_bundle/                     # CORE eval data (~162 MB, fetched from S3)
├── base_checkpoints/
│   ├── d20_muon/                    # Muon pretrain
│   │   ├── model_021400.pt
│   │   └── meta_021400.json
│   ├── d20_muon_early/              # 10-step Muon pretrain (for the early-step SVD point)
│   ├── d20_adam_lr0.001/            # Adam pretrain (LR-tuned)
│   └── d20_adam_lr0.001_early/      # 10-step Adam pretrain
├── wikitext_checkpoints/            # one directory per fine-tune run
│   └── <model_tag>_<mode>_<lr>_ep<n>_seed<s>/
│       ├── meta_<step>.json         # always written (incl. eval_log trajectory)
│       └── model_<step>.pt          # only when --save_weights=True
├── pretrain_ppl_eval/               # auxiliary WikiText PPL on the pretrained models
│   ├── base_d20_muon.json
│   └── base_d20_adam_lr0.001.json
├── base_eval/                       # CORE evaluation, one CSV per model
│   └── base_eval_021400.csv
└── report_<MODEL_TAG>/              # markdown training report (per pretrain)
    ├── header.md
    ├── tokenizer-training.md
    ├── tokenizer-evaluation.md
    ├── base-model-training.md
    ├── base-model-loss.md
    ├── base-model-evaluation.md
    └── report.md                    # final concatenated summary
```
