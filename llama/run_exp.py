"""
Unified experiment runner for LoRA fine-tuning using Hydra configuration.

Supports INDEPENDENT selection of initialization method and optimizer:

Initialization methods (init_method):
- lora: Standard LoRA (default initialization)
- full_ft: Full fine-tuning (no LoRA, train all parameters)

Optimizers (optimizer.name):
- adamw: Standard AdamW optimizer
- muon: Muon optimizer

Usage:
    # Default: lora + adamw
    python run_exp.py

    # Full fine-tuning
    python run_exp.py init_method=full_ft

    # Different optimizers
    python run_exp.py optimizer.name=muon

    # Combine initialization and optimizer freely
    python run_exp.py init_method=lora optimizer.name=muon

    # Sweep experiments
    python run_exp.py --multirun init_method=lora,full_ft optimizer.name=adamw,muon
"""
import glob
import logging
import os
import shutil

import hydra
import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model

from data import DATASET_MAP
from utils import (
    find_all_linear_modules,
    initialize_text_to_text_model,
    set_seed,
    train_text_to_text_model,
)

log = logging.getLogger(__name__)

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # ---- Seed, accelerator, resolve init_method & optimizer_name ----
    set_seed(cfg.experiment.seed)
    torch.backends.cudnn.benchmark = True
    accelerator = Accelerator()
    init_method = cfg.init_method.lower()
    optimizer_name = cfg.optimizer.name.lower()
    log.info(f"Using init_method: {init_method}")
    log.info(f"Using optimizer: {optimizer_name}")

    # ---- Build wandb run name & init wandb on rank 0 ----
    config_dict = {
        "init": init_method,
        "opt": optimizer_name,
        "model": cfg.model.name.split("/")[-1],
        "d": cfg.dataset.name,
        "sd": cfg.experiment.seed,
    }
    if init_method != "full_ft":
        config_dict["r"] = cfg.peft.lora_r
        config_dict["a"] = cfg.peft.lora_alpha
    wandb_name = cfg.wandb.name or "_".join([f"{k}={v}" for k, v in config_dict.items()])
    if accelerator.is_local_main_process:
        wandb.init(
            project=cfg.wandb.project,
            name=wandb_name,
            mode=cfg.wandb.mode,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(init_timeout=300),  # 5 minutes timeout
        )

    # ---- Load base model & tokenizer ----
    model, tokenizer = initialize_text_to_text_model(
        model_name=cfg.model.name,
        dtype=cfg.model.dtype,
        flash_attention=cfg.model.flash_attention,
    )
    if accelerator.is_local_main_process:
        log.info(f"Model loaded: {cfg.model.name}")

    # ---- Wrap with PEFT (LoRA) or keep full-FT ----
    if init_method == "full_ft":
        log.info("Full fine-tuning mode: training all model parameters")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        log.info(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.2f}")
    elif init_method == "lora":
        if cfg.peft.target_modules == "all":
            target_modules = find_all_linear_modules(model)
        else:
            target_modules = list(cfg.peft.target_modules)
        log.info(f"Target modules: {target_modules}")
        peft_config = LoraConfig(
            target_modules=target_modules,
            lora_alpha=cfg.peft.lora_alpha,
            r=cfg.peft.lora_r,
            lora_dropout=cfg.peft.lora_dropout,
            bias=cfg.peft.bias,
        )
        log.info(f"PEFT config: {peft_config}")
        model = get_peft_model(model=model, peft_config=peft_config)
        model.print_trainable_parameters()
    else:
        raise ValueError(
            f"Unknown init_method: {init_method!r}. Expected 'lora' or 'full_ft'."
        )

    # ---- Load dataset ----
    if cfg.dataset.name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}. Available: {list(DATASET_MAP.keys())}")
    dataset_func = DATASET_MAP[cfg.dataset.name]
    train_set, val_set, _ = dataset_func(
        max_tokens=cfg.dataset.max_length,
        model_name=cfg.model.name,
    )
    log.info(f"Dataset loaded: {cfg.dataset.name}, train={len(train_set)}, val={len(val_set)}")

    # ---- Train ----
    log.info("Starting training...")
    save_only_model = cfg.training.save_only_model
    model = train_text_to_text_model(
        run_name=wandb_name,
        train_dataset=train_set,
        valid_dataset=val_set,
        model=model,
        tokenizer=tokenizer,
        num_train_epochs=cfg.training.num_epochs,
        max_steps=cfg.training.max_steps,
        per_device_batch_size=cfg.training.per_device_batch_size,
        real_batch_size=cfg.training.real_batch_size,
        bf16=(cfg.model.dtype == "bf16"),
        max_length=cfg.dataset.max_length,
        logging_steps=cfg.training.logging_steps,
        save_total_limit=cfg.training.save_total_limit,
        save_only_model=save_only_model,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        num_process=accelerator.num_processes,
        seed=cfg.experiment.seed,
        output_dir=cfg.experiment.output_dir,
        deepspeed=cfg.training.deepspeed,
        # Optimizer settings
        optimizer_name=optimizer_name,
        # Muon-specific settings
        muon_momentum=cfg.optimizer.muon_momentum,
        muon_backend_steps=cfg.optimizer.muon_backend_steps,
        ns_dtype=cfg.optimizer.ns_dtype,
        ns_using_pe=cfg.optimizer.ns_using_pe,
        # Evaluation settings
        eval_before_training=cfg.evaluation.eval_before_training,
        eval_times=cfg.evaluation.eval_times,
        log_eval_samples=cfg.evaluation.log_eval_samples,
        eval_sample_size=cfg.evaluation.eval_sample_size,
        eval_sample_max_new_tokens=cfg.evaluation.eval_sample_max_new_tokens,
        # Cleanup settings
        cleanup_outputs=cfg.experiment.cleanup_outputs,
        # Resume from checkpoint
        resume_from_checkpoint=cfg.experiment.resume_from_checkpoint,
    )

    # ---- Save final model & verify (rank 0 only) ----
    save_dir = os.path.join(cfg.experiment.output_dir, wandb_name)
    if accelerator.is_local_main_process:
        model.save_pretrained(save_dir)
        log.info(f"Final model saved to {save_dir}")

        # Verify by loading (only for PEFT models, not full_ft)
        if init_method != "full_ft":
            model_verify, _ = initialize_text_to_text_model(
                cfg.model.name, cfg.model.dtype, flash_attention=False
            )
            model_verify = PeftModel.from_pretrained(model_verify, save_dir)
            log.info("Model verification successful")
            del model_verify

        # ---- Remove the checkpoint dir that duplicates the final model ----
        if save_only_model:
            max_steps = cfg.training.max_steps
            if max_steps > 0:
                final_step = max_steps
            else:
                # When using epochs, we don't know exact step count here, so we
                # remove the highest checkpoint (it's the final model).
                final_step = None

            checkpoint_dirs = glob.glob(os.path.join(cfg.experiment.output_dir, "checkpoint-*"))
            if checkpoint_dirs:
                checkpoint_steps = []
                for ckpt_dir in checkpoint_dirs:
                    try:
                        step = int(os.path.basename(ckpt_dir).split("-")[1])
                        checkpoint_steps.append((step, ckpt_dir))
                    except (ValueError, IndexError):
                        continue

                if checkpoint_steps:
                    checkpoint_steps.sort(key=lambda x: x[0], reverse=True)
                    highest_step, highest_ckpt = checkpoint_steps[0]
                    # If max_steps is set, only remove on exact match. Otherwise
                    # always remove the highest checkpoint (it's the final model).
                    should_remove = (final_step is None) or (highest_step == final_step)
                    if should_remove:
                        log.info(f"Removing duplicate checkpoint: {highest_ckpt} (same as final model)")
                        shutil.rmtree(highest_ckpt)
                    else:
                        log.info(f"Keeping checkpoint-{highest_step} (final_step={final_step})")

    # ---- Finish wandb ----
    if accelerator.is_local_main_process:
        wandb.finish()
    log.info("Experiment completed!")


if __name__ == "__main__":
    main()
