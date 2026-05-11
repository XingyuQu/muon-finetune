import logging
import math
import os
import random
import typing as tp

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


log = logging.getLogger(__name__)


# =============================================================================
# EvalSampleOutputCallback - Print sample predictions during evaluation
# =============================================================================

class EvalSampleOutputCallback(TrainerCallback):
    """
    Callback to print sample predictions during evaluation.
    Useful for debugging and monitoring model behavior.
    """
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int = 3,
        max_new_tokens: int = 8,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    @torch.no_grad()
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        eval_dataloader = kwargs.get("eval_dataloader")
        dataset = getattr(eval_dataloader, "dataset", None) if eval_dataloader else None
        if dataset is None:
            return
        dataset_size = len(dataset)
        if dataset_size == 0 or self.num_samples <= 0:
            return
        model = kwargs.get("model")
        if model is None:
            return

        rng = random.Random(self.seed + (state.global_step or 0))
        num_samples = min(self.num_samples, dataset_size)
        indices = rng.sample(range(dataset_size), k=num_samples)
        prefix = "test" if metrics and any(k.startswith("test_") for k in metrics.keys()) else "eval"
        print(f"\n[{prefix}] Sample predictions (step={state.global_step})")

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        for i, idx in enumerate(indices):
            # Row access goes through set_transform; "x"/"y" columns are gone post-transform,
            # so we recover them by splitting input_ids on the labels=-100 mask.
            sample = dataset[idx]
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            x_ids = [t for t, l in zip(input_ids, labels) if l == -100]
            y_ids = [t for t, l in zip(input_ids, labels) if l != -100]
            x_text = self.tokenizer.decode(x_ids, skip_special_tokens=True)
            y_text = self.tokenizer.decode(y_ids, skip_special_tokens=True)
            inputs = self.tokenizer(
                x_text,
                return_tensors="pt",
                truncation=True,
            ).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[{i}] x: {x_text[:100]}...")
            print(f"    y: {y_text}")
            print(f"    pred: {pred_text}")

        if was_training:
            model.train()


class CudaCacheCleanupCallback(TrainerCallback):
    """
    Callback to clear CUDA cache before evaluation to prevent OOM.
    Useful for high-rank LoRA training where memory is tight.
    """

    def on_step_end(self, args, state, control, **kwargs):
        """Clear CUDA cache before evaluation starts (when should_evaluate is True)."""
        if control.should_evaluate and torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Custom Optimizer Creation
# =============================================================================

def create_custom_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.01,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Build a non-AdamW optimizer (currently only "muon"); AdamW is handled by HF Trainer.

    For Muon, splits parameters into 2D matrix params (Muon) and 1D / embedding
    params (AdamW fallback inside the Muon optimizer).

    kwargs (Muon-only):
        momentum (default 0.95), backend_steps (default 5),
        ns_dtype (default "bf16"), ns_using_pe (default False).
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "muon":
        from muon.muon_optimizers import Muon

        # Separate 2D params (for Muon) from 1D/embedding params (for AdamW)
        muon_params = []
        adamw_params = []
        adamw_emb_params = []
        # Embedding layer names for different model architectures
        # Note: Use specific patterns to avoid false positives (e.g., "shared" matching "shared_experts" in MoE)
        embedding_keywords = (
            "lm_head",
            "embed_tokens",
            "wte",
            "wpe",
            "embedding",
            "embeddings",
            "position_embedding",
            "positional_embedding",
            "class_embedding",
            "patch_embedding",
            "visual_projection",
            "text_projection",
        )
        # Exact match patterns (for layer names like "shared" in T5 that shouldn't match "shared_experts")
        embedding_exact_suffixes = (".shared", "shared.weight")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_embedding = (
                any(kw in name for kw in embedding_keywords) or
                any(name.endswith(suffix) for suffix in embedding_exact_suffixes)
            )
            if param.ndim >= 2 and not is_embedding:
                muon_params.append(param)
            else:
                if is_embedding:
                    adamw_emb_params.append(param)
                else:
                    adamw_params.append(param)

        ns_using_pe = kwargs.get("ns_using_pe", False)
        ns_dtype = kwargs.get("ns_dtype", "bf16")

        log.info(
            f"Muon param groups -> muon: {len(muon_params)}, "
            f"adamw: {len(adamw_params)}, emb: {len(adamw_emb_params)}"
        )

        optimizer = Muon(
            muon_params,
            lr=learning_rate,
            momentum=kwargs.get("momentum", 0.95),
            nesterov=True,
            backend="newtonschulz5",
            backend_steps=kwargs.get("backend_steps", 5),
            weight_decay=weight_decay,
            adamw_params=adamw_params + adamw_emb_params if (adamw_params or adamw_emb_params) else None,
            adamw_lr=learning_rate,
            adamw_betas=kwargs.get("adamw_betas", (0.9, 0.95)),
            adamw_eps=kwargs.get("adamw_eps", 1e-8),
            adamw_wd=weight_decay,
            ns_using_pe=ns_using_pe,
            ns_dtype=ns_dtype,
        )

        return optimizer

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def set_seed(seed: int):
    """Seed Python / NumPy / PyTorch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def find_all_linear_modules(model) -> tp.List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            module_names.add(name.split(".")[-1])
    return list(module_names)


def causalLMEncode(example, tokenizer, max_length=-1):
    """
    Encode examples for CausalLM training.
    Tokenizes x and y separately so the prompt/response boundary is exact
    (no need to recover it from a combined tokenization).
    Returns data WITHOUT padding — padding is done by the data collator.
    """
    x_list = example["x"]
    y_list = example["y"]

    # Tokenize x and y separately
    x_encodings = tokenizer(
        x_list,
        add_special_tokens=True,
        truncation=False,
    )
    y_encodings = tokenizer(
        [y + tokenizer.eos_token for y in y_list],
        add_special_tokens=False,
        truncation=False,
    )

    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    # Cache space token ids (same for all examples)
    space_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]

    for i in range(len(x_list)):
        x_ids = x_encodings["input_ids"][i]
        y_ids = y_encodings["input_ids"][i]
        x_len = len(x_ids)

        # Concatenate x + " " + y; labels mask out x part (incl. space) with -100
        combined_ids = x_ids + space_ids + y_ids
        labels = [-100] * (x_len + len(space_ids)) + y_ids

        # Truncate both to max_length if needed (keep input_ids and labels aligned)
        if max_length > 0 and len(combined_ids) > max_length:
            combined_ids = combined_ids[:max_length]
            labels = labels[:max_length]

        all_input_ids.append(combined_ids)
        all_attention_mask.append([1] * len(combined_ids))
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def cleanup_run_outputs(output_dir: str) -> None:
    """Clean up checkpoint directories after training."""
    import shutil
    if not output_dir or not os.path.isdir(output_dir):
        return
    shutil.rmtree(output_dir, ignore_errors=True)


def initialize_text_to_text_model(
    model_name: str,
    dtype: str,
    tokenizer: str = None,
    flash_attention: bool = False
):
    """Load a CausalLM + tokenizer; optionally use SDPA (auto Flash-Attn 2)."""
    model_config = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
    )
    if flash_attention:
        # Use SDPA (Scaled Dot Product Attention) which auto-selects Flash Attention 2 when available
        # SDPA is more compatible with torch.compile than the HF flash_attention_2 wrapper
        log.info("Using SDPA (will auto-select Flash Attention 2 if available)")
        model_config["attn_implementation"] = "sdpa"
    match dtype:
        case "fp32":
            model_config["torch_dtype"] = torch.float32
        case "bf16":
            model_config["torch_dtype"] = torch.bfloat16
        case _:
            raise ValueError("Wrong dtype")
    model = AutoModelForCausalLM.from_pretrained(**model_config)
    if tokenizer:
        log.info(f"Using custom tokenizer {tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def transform_dataset(tokenizer, dataset, max_length):
    """Attach an on-the-fly tokenization transform to an HF Dataset."""
    dataset.set_transform(lambda x: causalLMEncode(x, tokenizer, max_length))
    return dataset


def train_text_to_text_model(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    per_device_batch_size: int = 1,
    real_batch_size: int = 32,
    max_length: int = None,
    **kwargs,
) -> torch.nn.Module:
    """Run HF Trainer with optional custom optimizer (Muon) and DeepSpeed config."""
    # ---- Validate batch sizing & tokenize datasets on the fly ----
    num_process = kwargs["num_process"]
    effective_batch_per_step = per_device_batch_size * num_process
    assert (
        real_batch_size % effective_batch_per_step == 0
    ), (
        f"real_batch_size ({real_batch_size}) must be divisible by "
        f"per_device_batch_size * num_process ({per_device_batch_size} * {num_process} = {effective_batch_per_step})"
    )
    accu_step = real_batch_size // effective_batch_per_step
    assert accu_step >= 1, (
        f"accu_step is {accu_step}, which means real_batch_size ({real_batch_size}) is smaller than "
        f"per_device_batch_size * num_process ({effective_batch_per_step}). "
        f"Either increase real_batch_size or decrease per_device_batch_size."
    )
    train_dataset, valid_dataset = transform_dataset(
        tokenizer, train_dataset, max_length
    ), transform_dataset(tokenizer, valid_dataset, max_length)

    # ---- Compute total / eval / warmup steps (honors max_steps if set, else num_epochs) ----
    num_train_epochs = kwargs["num_train_epochs"]
    max_steps = kwargs["max_steps"]  # -1 means use num_epochs
    steps_per_epoch = math.ceil(len(train_dataset) / real_batch_size)
    if max_steps > 0:
        max_train_steps = max_steps
        log.info(f"Using max_steps={max_steps} (equivalent to {max_steps / steps_per_epoch:.2f} epochs)")
    else:
        max_train_steps = steps_per_epoch * num_train_epochs
    eval_times = kwargs["eval_times"]
    eval_steps = max(1, max_train_steps // eval_times)
    log.info(f"Evaluation: every {eval_steps} steps ({eval_times} times during training)")
    # math.ceil to match HuggingFace Trainer's get_warmup_steps behavior
    warmup_steps = math.ceil(kwargs["warmup_ratio"] * max_train_steps)

    # ---- Resolve optimizer / output dir / deepspeed config ----
    optimizer_name = kwargs["optimizer_name"].lower()
    is_custom_optimizer = optimizer_name != "adamw"
    log.info(f"Using optimizer: {optimizer_name}")

    output_dir = kwargs["output_dir"]
    log.info(f"Checkpoint output directory: {output_dir}")

    deepspeed_config = kwargs["deepspeed"]
    if deepspeed_config == "zero2":
        deepspeed_config = os.path.join(os.path.dirname(__file__), "conf", "ds_zero2.json")
    elif deepspeed_config in ("none", "None", ""):
        deepspeed_config = None  # Disable DeepSpeed, use HF DDP

    gradient_checkpointing = kwargs["gradient_checkpointing"]

    # ---- Build TrainingArguments ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir="./logs",
        logging_steps=kwargs["logging_steps"],
        seed=kwargs["seed"],
        num_train_epochs=num_train_epochs if max_steps <= 0 else 100,  # Use large value when max_steps is set
        max_steps=max_steps if max_steps > 0 else -1,  # -1 means use num_epochs
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=kwargs["save_total_limit"],
        save_only_model=kwargs["save_only_model"],
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accu_step,
        per_device_eval_batch_size=per_device_batch_size,
        eval_accumulation_steps=real_batch_size,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        bf16=kwargs["bf16"],
        optim="adamw_torch",
        torch_compile=False,
        learning_rate=kwargs["learning_rate"],
        lr_scheduler_type=kwargs["lr_scheduler_type"],
        warmup_ratio=kwargs["warmup_ratio"],
        weight_decay=kwargs["weight_decay"],
        max_grad_norm=kwargs["max_grad_norm"],
        remove_unused_columns=False,  # tokenize the dataset on the fly
        deepspeed=deepspeed_config,
    )

    # ---- Data collator (skip decoder_input_ids; CausalLM doesn't need it) ----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if kwargs["bf16"] else None,
    )

    # ---- Callbacks (sample preview + CUDA cache cleanup before eval) ----
    callbacks = []
    if kwargs["log_eval_samples"]:
        callbacks.append(
            EvalSampleOutputCallback(
                tokenizer=tokenizer,
                num_samples=kwargs["eval_sample_size"],
                max_new_tokens=kwargs["eval_sample_max_new_tokens"],
                seed=kwargs["seed"],
            )
        )
    callbacks.append(CudaCacheCleanupCallback())

    # ---- Build custom optimizer + lr scheduler (only when not adamw) ----
    custom_optimizer = None
    lr_scheduler = None
    if is_custom_optimizer:
        from transformers.optimization import get_scheduler

        custom_optimizer = create_custom_optimizer(
            model=model,
            optimizer_name=optimizer_name,
            learning_rate=kwargs["learning_rate"],
            weight_decay=kwargs["weight_decay"],
            # Muon-specific kwargs
            momentum=kwargs["muon_momentum"],
            backend_steps=kwargs["muon_backend_steps"],
            ns_dtype=kwargs["ns_dtype"],
            ns_using_pe=kwargs["ns_using_pe"],
        )
        log.info(f"Created custom optimizer: {optimizer_name}")

        lr_scheduler_type = kwargs["lr_scheduler_type"]
        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=custom_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
        log.info(f"Created lr_scheduler: {lr_scheduler_type}, "
                 f"warmup_steps={warmup_steps}, max_train_steps={max_train_steps}")

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=callbacks,
        optimizers=(custom_optimizer, lr_scheduler) if custom_optimizer else (None, None),
    )

    # ---- Resolve resume_from_checkpoint (supports "auto") ----
    resume_from_checkpoint = kwargs["resume_from_checkpoint"]
    if resume_from_checkpoint == "auto":
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            log.info(f"Auto-detected checkpoint: {last_checkpoint}")
            resume_from_checkpoint = last_checkpoint
        else:
            log.info("No checkpoint found in output_dir, starting fresh")
            resume_from_checkpoint = None

    # ---- Pre-training eval (skipped when resuming or when DeepSpeed) ----
    eval_before_training = kwargs["eval_before_training"]
    if resume_from_checkpoint:
        log.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    elif eval_before_training:
        if deepspeed_config is None:
            log.info("Evaluating model before training...")
            trainer.evaluate()
        else:
            log.info("Skipping pre-training eval (DeepSpeed mode) - first eval at step %d", eval_steps)
    else:
        log.info("Skipping pre-training eval (eval_before_training=False) - first eval at step %d", eval_steps)

    # ---- Train, then cleanup checkpoint dir if requested ----
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if kwargs["cleanup_outputs"]:
        cleanup_run_outputs(output_dir)

    return model
