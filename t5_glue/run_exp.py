import logging
import math
import os
from typing import Dict, List

import hydra
import torch
import wandb
from datasets import concatenate_datasets
from omegaconf import DictConfig, OmegaConf
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq

from data import DATASET_MAP
from utils import (
    find_all_linear_modules,
    find_hidden_state_size,
    initialize_text_to_text_model,
    set_seed,
    train_text_to_text_model,
    transform_dataset,
)

log = logging.getLogger(__name__)


@torch.no_grad()
def reinit_lora_modules(name, module, init_config, named_grads=None):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    lora_r = min(module.lora_A.default.weight.shape)
    if init_config.mode == "simple":
        match init_config.lora_A:
            case "kaiming":
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
            case _:
                raise ValueError(f"Unknown lora_A initialization: {init_config.lora_A}")
        match init_config.lora_B:
            case "zeros":
                torch.nn.init.zeros_(module.lora_B.default.weight)
            case _:
                raise ValueError(f"Unknown lora_B initialization: {init_config.lora_B}")
    elif init_config.mode == "lora_one":
        grad_name = ".".join(name.split(".")[2:]) + ".weight"
        grads = named_grads[grad_name]
        U, S, V = torch.svd_lowrank(-grads.cuda().float(), q=512, niter=16)  # from lora-one repo
        V = V.T
        B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
        A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0])
        gamma = getattr(init_config, "stable_gamma", 16)
        if gamma and gamma > 0:
            B = B / gamma**0.5
            A = A / gamma**0.5
        module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().cuda())

    if init_config.get("dtype") == "fp32":
        module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(torch.float32)
        module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(torch.float32)

    if init_config.get("skip_merge", False):
        log.info("Init skip_merge=True: not subtracting LoRA offset from base weights")
        return
    # If lora_A@lora_B is not zero, subtract lora_A@lora_B from the original weight matrix
    offset = (module.lora_B.default.weight @ module.lora_A.default.weight).to(
        module.weight.data.device
    )
    scaling_factor = module.scaling["default"]
    offset *= scaling_factor
    module.weight.data -= offset


def reinit_lora(model, init_config, named_grads=None):
    r"""
    Reinitialize the lora model in place with the given configuration.
    """
    for name, module in tqdm(model.named_modules(), desc="Reinitializing Lora"):
        if isinstance(module, LoraLinear):
            reinit_lora_modules(name, module, init_config, named_grads=named_grads)


def get_record_gradient_hook(model, record_dict):
    """Return a backward hook that accumulates each parameter's grad into `record_dict` (CPU) and zeros it."""
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook


def estimate_gradient(
    model,
    dataset,
    batch_size: int = 4,
    collate_fn=None,
) -> Dict[str, List[torch.Tensor]]:
    r"""
    Estimate the gradient of the model on the given dataset
    """
    log.info("Estimating gradient")
    model.train()
    named_grads = {}
    hooks = []
    for name, param in model.named_parameters():
        hook = param.register_hook(get_record_gradient_hook(model, named_grads))
        hooks.append(hook)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    num = 0
    for batch in tqdm(dataloader, desc="Estimating gradient"):
        num += 1
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        get_record_gradient_hook(model, named_grads)(None)  # get gradient of last layer
        # make sure the gradient is cleared
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad = None
    for n, g in named_grads.items():
        named_grads[n] /= num
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    return named_grads


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_exp(cfg: DictConfig):
    """Hydra entry: load GLUE dataset, wrap model with PEFT (or full-FT), then train + eval."""
    log.info(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    # ---- Resolve config values & validate dataset ----
    model_name = cfg.model.name
    dataset_name = cfg.dataset_name
    eval_protocol = cfg.eval_protocol
    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Dataset {dataset_name} not found in data.DATASET_MAP")
    dataset_func = DATASET_MAP[dataset_name]

    use_peft = cfg.peft.use_peft
    if_use_rslora = cfg.peft.use_rslora
    lora_r = cfg.peft.lora_r
    lora_relative_r = cfg.peft.lora_relative_r
    lora_target_modules = cfg.peft.lora_target_modules

    if use_peft:
        assert (lora_r is not None) ^ (
            lora_relative_r is not None
        ), "Please specify lora_r or lora_relative_r"
        assert lora_target_modules is not None, "Please specify lora_target_modules"
    if cfg.dry_run:
        return

    # ---- Build wandb run name & init wandb ----
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "eval_protocol": eval_protocol,
        "use_peft": use_peft,
        "lora_r": lora_r,
        "lora_target_modules": lora_target_modules,
        "lora_relative_r": lora_relative_r,
        "seed": cfg.seed,
    }
    if cfg.wandb.name:
        name = cfg.wandb.name
    else:
        name = "_".join([f"{k}={v}" for k, v in config.items()])
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=name,
        config=config,
        settings=wandb.Settings(init_timeout=300),
    )

    # ---- Load dataset & resolve eval protocol splits ----
    train_set, val_set, test_set = dataset_func()
    if eval_protocol == "train_val":
        # Merge train/val splits for training; use GLUE original validation as eval
        train_set = concatenate_datasets([train_set, val_set])
        val_set = test_set
        log.info("train_val: merged train/val splits for training; using GLUE validation as eval.")
        test_set = None
    elif eval_protocol != "train_val_test":
        raise ValueError(f"Unknown eval_protocol={eval_protocol}")

    # ---- Load base model & tokenizer ----
    model, tokenizer = initialize_text_to_text_model(model_name, cfg.model.bf16)
    model = model.to("cuda")

    # ---- LoRA-One pre-pass: estimate gradient on a small subset ----
    # Skip if dora/adalora — those don't go through reinit_lora so named_grads would be discarded.
    named_grads = None
    is_regular_lora_path = use_peft and not cfg.peft.get("dora", False) and not cfg.peft.get("adalora", False)
    if is_regular_lora_path and cfg.init.mode == "lora_one":
        init_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if cfg.model.bf16 else None,
        )
        if isinstance(train_set, list):
            temp_set = train_set[: cfg.init.bsz * cfg.init.iters]
        else:
            temp_set = train_set.select(range(cfg.init.bsz * cfg.init.iters))
        transform_dataset(
            tokenizer=tokenizer,
            dataset=temp_set,
            max_length=cfg.init.max_length,
        )
        named_grads = estimate_gradient(
            model,
            temp_set,
            cfg.init.bsz,
            collate_fn=init_collator,
        )

    # ---- Resolve LoRA target modules / rank / scaling / pissa init ----
    if lora_target_modules == "all":
        lora_target_modules = find_all_linear_modules(model)
    elif lora_target_modules:
        lora_target_modules = list(lora_target_modules)
    else:
        lora_target_modules = []
    if lora_relative_r is not None:
        hidden_size = find_hidden_state_size(model)
        lora_r = int(hidden_size * lora_relative_r)
        log.info(f"lora_r is set to {hidden_size} * {lora_relative_r} = {lora_r}")
    # Replace the wandb config's raw spec with the actually-resolved values
    wandb.config.update(
        {"resolved_target_modules": lora_target_modules, "resolved_lora_r": lora_r},
        allow_val_change=True,
    )
    lora_scaler = None
    if use_peft and lora_r:
        if if_use_rslora:
            lora_scaler = cfg.peft.lora_alpha / math.sqrt(lora_r)
        else:
            lora_scaler = cfg.peft.lora_alpha / lora_r
    init_lora_weights = None
    if use_peft and cfg.init.mode == "pissa":
        pissa_niter = getattr(cfg.init, "pissa_niter", None)
        if pissa_niter:
            init_lora_weights = f"pissa_niter_{pissa_niter}"
        else:
            init_lora_weights = "pissa"

    # ---- Wrap model with PEFT (dora / adalora / regular LoRA) or keep full-FT ----
    orig_model_params = sum(p.numel() for p in model.parameters())
    if use_peft:
        peft_kwargs = {
            "r": lora_r,
            "lora_alpha": cfg.peft.lora_alpha,
            "target_modules": lora_target_modules,
            "use_rslora": if_use_rslora,
        }
        if init_lora_weights is not None:
            peft_kwargs["init_lora_weights"] = init_lora_weights

        if cfg.peft.get("dora", False):
            log.info("Using Dora")
            peft_kwargs["use_dora"] = True
            peft_config = LoraConfig(**peft_kwargs)
            is_regular_lora = False
        elif cfg.peft.get("adalora", False):
            log.info("Using AdaLora")
            adalora_total_step = math.ceil(len(train_set) / cfg.model.real_batch_size) * cfg.model.epochs
            adalora_tinit = max(10, int(adalora_total_step * 0.05))
            adalora_tfinal = max(20, int(adalora_total_step * 0.15))
            adalora_deltaT = max(1, int((adalora_total_step - adalora_tinit - adalora_tfinal) / 50))
            log.info(f"AdaLoRA params: total_step={adalora_total_step}, tinit={adalora_tinit}, tfinal={adalora_tfinal}, deltaT={adalora_deltaT}")
            log.info(f"AdaLoRA config: init_r={cfg.peft.init_r}, target_r={lora_r}, orth_reg_weight={cfg.peft.orth_reg_weight}")
            peft_config = AdaLoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                init_r=cfg.peft.init_r,
                target_r=lora_r,
                lora_alpha=cfg.peft.lora_alpha,
                lora_dropout=cfg.peft.lora_dropout,
                target_modules=lora_target_modules,
                tinit=adalora_tinit,
                tfinal=adalora_tfinal,
                deltaT=adalora_deltaT,
                total_step=adalora_total_step,
                orth_reg_weight=cfg.peft.orth_reg_weight,
            )
            is_regular_lora = False
        else:
            peft_config = LoraConfig(**peft_kwargs)
            is_regular_lora = True

        model = get_peft_model(model, peft_config)

        # Regular LoRA path: manual reinit (load-bearing for lora_one; no-op for default simple init)
        if is_regular_lora:
            if cfg.init.mode == "pissa":
                log.info("Init mode pissa: skipping manual reinit.")
            else:
                reinit_lora(model, cfg.init, named_grads=named_grads)

        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    else:
        # full finetune
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": orig_model_params,
            "trainable_ratio": trainable_params / orig_model_params,
            "param_ratio": 1,
        }
    log.info(rate)
    wandb.summary.update(rate)

    # ---- Train ----
    optimizer_name = getattr(cfg.model, "optimizer", None)
    cleanup_outputs = not (use_peft and dataset_name in {"mnli", "qnli"})
    model = train_text_to_text_model(
        f"{cfg.wandb.project}/{name}",
        train_set,
        val_set,
        test_dataset=test_set,
        model=model,
        tokenizer=tokenizer,
        num_train_epochs=cfg.model.epochs,
        per_device_batch_size=cfg.model.per_device_batch_size,
        real_batch_size=cfg.model.real_batch_size,
        bf16=cfg.model.bf16,
        early_stopping_patience=cfg.model.early_stopping_patience,
        max_length=cfg.model.max_length,
        logging_steps=cfg.model.logging_steps,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_ratio=cfg.model.warmup_ratio,
        max_grad_norm=cfg.model.max_grad_norm,
        optim=cfg.model.optim,
        load_best_model_at_end=(eval_protocol == "train_val_test"),
        eval_protocol=eval_protocol,
        optimizer_name=optimizer_name,
        cleanup_outputs=cleanup_outputs,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        seed=cfg.seed,
        using_pe=cfg.using_pe,
        ns_dtype=cfg.model.ns_dtype,
        lora_scaler=lora_scaler,
        save_strategy=cfg.model.save_strategy,
        save_total_limit=cfg.model.save_total_limit,
    )

    # ---- Persist final full-FT model & finish wandb ----
    if not use_peft:
        save_dir = os.path.join(
            "results", f"{cfg.wandb.project}/{name}/{cfg.seed}", "final_checkpoint"
        )
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        log.info(f"Saved full-FT model to {save_dir}")
    wandb.finish()


if __name__ == "__main__":
    run_exp()
