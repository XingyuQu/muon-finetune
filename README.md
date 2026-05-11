# Can Muon Fine-tune Adam-Pretrained Models?

Code release for the ICML 2026 paper *Can Muon Fine-tune Adam-Pretrained Models?*

## Repository layout

```
icml_muon/
├── nanochat/        # NanoChat 561M pretraining + WikiText fine-tune; mismatch analysis & SVD metrics
├── t5_glue/         # T5-base on GLUE (NLU) + LoRA-variants comparison
├── llama/           # Llama-2-7B/13B on MetaMath / Code-Feedback / WizardLM
├── clip_vitb32/     # CLIP ViT-B/32 on 6 image-classification benchmarks
├── common/muon/     # shared Muon optimizer (symlinked as llama/muon and clip_vitb32/muon)
├── peft/            # vendored PEFT fork from https://github.com/Outsider565/LoRA-GA
```

`nanochat/nanochat/` is vendored from [Karpathy's nanochat](https://github.com/karpathy/nanochat) (Muon implementation, GPT model, dataloader, checkpoint manager). Paper-specific code lives under `nanochat/scripts/`. See [`nanochat/README.md`](nanochat/README.md) for nanochat-specific setup.

## Environments

Three environments cover the four pillars:

- **`muon_ft`** — training env for T5, Llama, and CLIP.
- **`lm_eval`** — separate env for generative Llama evaluation (lm-eval-harness + vLLM); kept isolated to avoid dependency conflicts with the training stack.
- **nanochat** — has its own `pyproject.toml` + `uv.lock` and a Rust BPE extension (`rustbpe/`); install per [`nanochat/README.md`](nanochat/README.md).

### `muon_ft` (training)

The `transformers` / `accelerate` / `wandb` / `datasets` / `hydra-core` versions below are the ones used in the paper. Letting pip pick the latest currently lands on `transformers 5.x`, which has API changes that silently break the HuggingFace Trainer ↔ wandb integration (`wandb.init()` succeeds but `train/loss`, `eval/accuracy` etc. never appear on the dashboard). Always install `peft` from the **vendored** `peft/` in this repo (a fork of [LoRA-GA's PEFT](https://github.com/Outsider565/LoRA-GA)), not from PyPI — the t5-glue LoRA-Pro / LoRA-RITE wrappers depend on the fork's internal hooks.

Run the commands below from the repo root (so that `./peft` resolves correctly).

**CUDA**:

```bash
conda create -n muon_ft python=3.12 -y && conda activate muon_ft
pip install torch torchvision matplotlib notebook scipy loguru
pip install "transformers[torch]==4.57.3" "accelerate==1.12.0" \
            "wandb==0.23.1" "datasets==4.4.1" "hydra-core==1.3.2" \
            deepspeed safetensors tqdm
pip install -e ./peft                                  # vendored LoRA-GA PEFT fork
pip install clip-benchmark                             # only required for the CLIP-ViT experiments
```

**AMD ROCm** (paper experiments used MI210):

```bash
conda create -n muon_ft python=3.12 -y && conda activate muon_ft
pip install torch==2.9.1 torchvision==0.24.1 pytorch-triton-rocm==3.5.1 \
            --index-url https://download.pytorch.org/whl/rocm6.4
pip install matplotlib notebook scipy loguru
pip install "transformers==4.57.3" "accelerate==1.12.0" \
            "wandb==0.23.1" "datasets==4.4.1" "hydra-core==1.3.2" \
            deepspeed safetensors tqdm
pip install -e ./peft                                  # vendored LoRA-GA PEFT fork
pip install clip-benchmark                             # only required for the CLIP-ViT experiments
```

### `lm_eval` (Llama eval)

```bash
conda create -n lm_eval python=3.11 -y && conda activate lm_eval
pip install vllm   # CUDA wheel. For AMD ROCm follow vLLM's official ROCm install guide: https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e . && cd ..
pip install "lm_eval[hf,vllm,api]" wandb
```

## Experiments

### NanoChat — `nanochat/`

Pretrain a 561M depth-20 transformer on FineWeb-Edu, then fine-tune on WikiText-2 to study optimizer mismatch. Setup, the 5-step reproduction pipeline (pretraining → WikiText LR sweep → best-LR trajectory runs → spectral analysis), output layout, and which artefacts are gitignored vs. regenerable are all documented in [`nanochat/README.md`](nanochat/README.md).

### T5-GLUE — `t5_glue/`

T5-base fine-tuning on GLUE under different optimizers and LoRA variants.

- `full_sft.sh` / `lora_sft.sh` — full-FT and LoRA sweeps over (variant × dataset × LR × seed) with a default 8-GPU file-locked queue.
- `run_exp.py` — single-config Hydra runner.
- `conf/` — Hydra configs: `model/` (T5 settings), `peft/` (`all`/`dora`/`adalora`/`full_ft`), `init/` (`default`/`lora_one`/`pissa`).
- `lora_pro.py`, `lora_rite/` — custom optimizer wrappers wired into the runner.

```bash
cd t5_glue
bash full_sft.sh                              # full-FT sweep across all GLUE tasks
bash lora_sft.sh                              # LoRA + LoRA variants sweep

# Single configuration: full-FT with Muon on MNLI
python run_exp.py +peft=full_ft ++dataset_name=mnli \
    ++model.optimizer=muon ++using_pe=False \
    ++model.learning_rate=1e-4 ++seed=0
```

**GPU pool.** `lora_sft.sh` / `full_sft.sh` default to GPUs `0..7` (one task per GPU, no DDP). Override with `GPU_LIST` (comma-separated):

```bash
GPU_LIST="0,1,2,3" bash lora_sft.sh           # use only 4 GPUs
```

**Task queue.** Tasks are enumerated in advance, written to `.queue_t5_glue_sft/tasks.queue` (or `.queue_t5_glue_full_sft/` for full-FT), and consumed by GPU workers via a `flock`-protected pop. Multiple shells / nodes can launch the same script against the same `QUEUE_DIR` and they will safely cooperate. Knobs:

- `RESET_QUEUE=1 bash lora_sft.sh` — wipe the existing queue and rebuild from the full task list (use after editing the sweep grid in the script).
- `QUEUE_DIR=/some/shared/path bash lora_sft.sh` — override where the queue lives (default is `$PWD/.queue_*`); set this to a network-shared directory for multi-node runs.
- Per-task stdout/stderr lands in `t5_glue/logs/<task_name>.log`. Tail one with `tail -f logs/<task_name>.log` while the sweep runs.

**`HF_HUB_OFFLINE`.** Both sweep scripts `export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}` — i.e. they default to **offline** mode and assume `t5-base` and the GLUE datasets are already in the HuggingFace cache. If you have not pre-downloaded them, run once with `HF_HUB_OFFLINE=0`:

```bash
HF_HUB_OFFLINE=0 bash lora_sft.sh             # one-time: allow Hub access on first run
```

### Llama — `llama/`

Llama-2-7B/13B instruction tuning on MetaMath / Code-Feedback / WizardLM, with separate generative evaluation. Llama-2 weights are gated on HuggingFace Hub — accept the license and run `huggingface-cli login` once.

- `full_sft.sh` / `lora_sft.sh` / `lora_13b_sft.sh` / `lora_rank_sft.sh` — training sweeps (default 8 GPUs per node; full-FT uses DeepSpeed Zero-2 for AdamW, while the rest are using DDP).
- `run_exp.py` — single-config Hydra runner.
- `run_lm_eval.sh <model_path> <tasks> [--merge]` — generative evaluation through lm-eval-harness + vLLM (run inside the `lm_eval` env). `run_all_evals.sh` fans this out across every checkpoint in `results/`.
- `merge_lora.py` — merge LoRA adapters into the base model (called automatically by `--merge`).
- `lm_eval_tasks/` — custom task definitions (`gsm8k_qa`, `humaneval`) auto-included by `run_lm_eval.sh`.
- `run_svd_metrics.sh` + `metrics/` — SVD / spectral metrics on Llama checkpoints.

```bash
cd llama

# Training sweeps (8 GPUs per node by default; multiple nodes can pull from the same shared queue)
bash full_sft.sh                              # full-FT (DeepSpeed Zero-2 for AdamW, DDP for Muon)
bash lora_sft.sh                              # LoRA on Llama-2-7B
bash lora_13b_sft.sh                          # LoRA on Llama-2-13B
bash lora_rank_sft.sh                         # LoRA rank ablation

# Single configuration: LoRA + Muon-PE on MetaMath
torchrun --nproc_per_node=8 run_exp.py \
    init_method=lora dataset.name=meta_math \
    optimizer.name=muon optimizer.ns_using_pe=true \
    training.learning_rate=5e-4 experiment.seed=0

# Evaluation (in the lm_eval env)
bash run_all_evals.sh                                                      # batch-eval every checkpoint in results/
bash run_lm_eval.sh <model_path> gsm8k_qa --merge                          # math   (single ckpt)
HF_ALLOW_CODE_EVAL=1 bash run_lm_eval.sh <model_path> humaneval --merge    # code
bash run_lm_eval.sh <model_path> arc_challenge,arc_easy,hellaswag,piqa,winogrande,boolq,openbookqa --merge   # commonsense
```

The training sweep launchers follow the same design as the [T5-GLUE ones](#t5-glue--t5_glue) — `flock`-protected `tasks.queue` under `.queue_llama_*/`, `RESET_QUEUE=1` to rebuild, `QUEUE_DIR=/shared/path` for multi-node, per-task logs under `llama/logs/llama_*/<task_name>.log`, and `HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}` defaulting to offline (run once with `HF_HUB_OFFLINE=0` to populate the HF cache for the Llama-2 weights and the three instruction-tuning datasets).

### CLIP ViT-B/32 — `clip_vitb32/`

CLIP ViT-B/32 fine-tuning on six image-classification benchmarks (SVHN, GTSRB, DTD, StanfordCars, RESISC45, SUN397).

- `scripts/run_sweep.sh` — full LR sweep over (variant × dataset × LR × seed).
- `scripts/run_bestlr.sh` — best-LR runs from the paper.
- `scripts/run_lora_rank_sweep.sh` / `run_lora_rank_bestlr.sh` — LoRA rank ablation.
- `clip_vit/train/main.py` — single-config training entry point.
- `tools/summarize_logs.py`, `summarize_lora_sweep.py`, `summarize_rank_study.py` — log → CSV aggregation.

```bash
cd clip_vitb32
bash scripts/run_sweep.sh                     # full LR sweep (variant × dataset × LR × seed)
bash scripts/run_bestlr.sh                    # best-LR runs from the paper
bash scripts/run_lora_rank_sweep.sh           # LoRA rank LR sweep
bash scripts/run_lora_rank_bestlr.sh          # LoRA rank best-LR runs

# Single configuration: full-FT with Muon on SVHN
CUDA_VISIBLE_DEVICES=0 python -m clip_vit.train.main \
    --dataset svhn --init-method full_ft --optimizer muon \
    --lr 1e-4 --num-epochs 40 --batch-size 256 \
    --save-root ./runs_clip_svhn --data-root ./data
```

Same `flock`-protected queue design as the [T5-GLUE launchers](#t5-glue--t5_glue) (`RESET_QUEUE=1` to rebuild, `QUEUE_DIR=/shared/path` for multi-node), with two CLIP-specific knobs: GPU pool defaults to `NUM_GPUS=8` but auto-respects an existing `CUDA_VISIBLE_DEVICES` and SLURM env vars (override with `NUM_GPUS=` or `GPU_IDS="0,1,2,3"`); the sweep grid can be subsetted per run with `DATASETS=` / `VARIANTS=`:

```bash
DATASETS="svhn" VARIANTS="lora_muon_pe" bash scripts/run_bestlr.sh
```

CLIP scripts run in online mode by default (no `HF_HUB_OFFLINE`); the first run pulls the model and the six benchmarks from HuggingFace / torchvision.

## Citation

```bibtex
```
