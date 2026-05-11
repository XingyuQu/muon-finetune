import json
import logging
import math
import os
import random
import shutil
from typing import List

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.optimization import get_scheduler
from transformers.trainer_utils import PredictionOutput

from lora_pro import LoRAProAdamW
from lora_rite import LoRARite
from muon.muon_optimizers import Muon

log = logging.getLogger(__name__)


def set_seed(seed: int):
    """Seed Python / NumPy / PyTorch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def find_all_linear_modules(model) -> List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    output_layer_names = ["lm_head"]
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not any(
            output_layer in name for output_layer in output_layer_names
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)


def find_hidden_state_size(model):
    """Probe the model for its hidden size by inspecting the first Linear layer."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            return min(module.weight.shape)
    return None


def SeqToSeqEncode(example, tokenizer, max_length=None):
    """Tokenize (x -> input_ids) and (y -> labels) for seq2seq training; no padding here."""
    inputs = tokenizer(
        example["x"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    outputs = tokenizer(
        example["y"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )

    results = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"],
    }

    return results


def initialize_text_to_text_model(
    model_name: str,
    bf16: bool,
    tokenizer: str = None,
):
    """Load a Seq2SeqLM (T5-style) and its tokenizer; require both eos and pad tokens."""
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token is None:
        raise ValueError("No eos token")
    if tokenizer.pad_token is None:
        raise ValueError("No padding token")
    return model, tokenizer


def build_compute_metrics(tokenizer: AutoTokenizer):
    """Return a Trainer-compatible compute_metrics fn that decodes preds/labels and reports exact-match accuracy."""
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    max_token_id = len(tokenizer) - 1

    def compute_metrics(p: PredictionOutput):
        predictions = p.predictions
        label_ids = p.label_ids

        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]
        if predictions is None or label_ids is None:
            return {"accuracy": 0.0}
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        predictions = predictions.astype(np.int64)
        label_ids = label_ids.astype(np.int64)

        if pad_token_id is not None:
            label_ids = np.where(label_ids == -100, pad_token_id, label_ids)
            invalid_pred = (predictions < 0) | (predictions > max_token_id)
            if np.any(invalid_pred):
                predictions = predictions.copy()
                predictions[invalid_pred] = pad_token_id
            invalid_label = (label_ids < 0) | (label_ids > max_token_id)
            if np.any(invalid_label):
                label_ids = label_ids.copy()
                label_ids[invalid_label] = pad_token_id

        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        num_correct = sum(
            pred_texts[i] == label_texts[i] for i in range(len(label_texts))
        )
        accuracy = num_correct / len(label_texts) if label_texts else 0.0
        return {"accuracy": accuracy}

    return compute_metrics


def cleanup_run_outputs(output_dir: str) -> None:
    """Recursively remove `output_dir` if it exists; quiet no-op otherwise."""
    if not output_dir or not os.path.isdir(output_dir):
        return
    shutil.rmtree(output_dir, ignore_errors=True)


class AdaLoraCallback(TrainerCallback):
    """Triggers AdaLoRA's adaptive rank reallocation after each optimizer step.

    No-op for non-AdaLoRA models (the base_model.update_and_allocate hasattr check fails).
    Replaces the old LogTrainer.optimizer_step override, which is dead code under
    HF Transformers >= 4.40 (Trainer no longer exposes an optimizer_step wrapper).
    """

    def on_step_end(self, args, state, control, model=None, **kwargs):
        base = getattr(model, "base_model", None) if model is not None else None
        update_fn = getattr(base, "update_and_allocate", None) if base is not None else None
        if update_fn is None:
            return
        try:
            update_fn(state.global_step)
        except Exception as e:  # noqa: BLE001
            if not getattr(self, "_warned", False):
                log.warning(f"AdaLoRA update_and_allocate failed: {e}")
                self._warned = True


class EvalSampleOutputCallback(TrainerCallback):
    """Print a few greedy-decoded prediction samples on each evaluation (rank 0 only)."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int = 3,
        max_new_tokens: int = 8,
        seed: int = 0,
        max_length: int = None,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.max_length = max_length
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
            # Row access goes through set_transform; the column "x"/"y" is gone post-transform,
            # so we decode input_ids/labels back to text instead.
            sample = dataset[idx]
            x_text = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            y_text = self.tokenizer.decode(sample["labels"], skip_special_tokens=True)
            inputs = self.tokenizer(
                x_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[{i}] x: {x_text}")
            print(f"    y: {y_text}")
            print(f"    pred: {pred_text}")
        if was_training:
            model.train()


def transform_dataset(tokenizer, dataset, max_length):
    """Attach an on-the-fly seq2seq tokenization transform to an HF Dataset."""
    dataset.set_transform(lambda x: SeqToSeqEncode(x, tokenizer, max_length))
    return dataset


def train_text_to_text_model(
    run_name: str,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    test_dataset: Dataset = None,
    per_device_batch_size: int = 1,
    real_batch_size: int = 32,
    max_length: int = None,
    using_pe: bool = False,
    eval_protocol: str = "train_val",
    **kwargs,
) -> torch.nn.Module:
    """Run Seq2SeqTrainer with optional custom optimizer (Muon / LoRARite / LoRAPro) and persist eval results."""
    # ---- Validate batch sizing & tokenize datasets on the fly ----
    assert (
        real_batch_size % per_device_batch_size == 0
    ), "real_batch_size must be divisible by per_device_batch_size"
    accu_step = real_batch_size // per_device_batch_size

    transform_dataset(tokenizer, train_dataset, max_length)
    transform_dataset(tokenizer, valid_dataset, max_length)
    if test_dataset is not None:
        transform_dataset(tokenizer, test_dataset, max_length)

    # ---- Compute eval cadence (~20/40/60/80/100% of training) ----
    num_train_epochs = kwargs["num_train_epochs"]
    total_steps = math.ceil(
        len(train_dataset)
        / (per_device_batch_size * accu_step)
    ) * num_train_epochs
    eval_steps = max(1, total_steps // 5)

    optimizer_name = kwargs.get("optimizer_name", None)
    load_best_model_at_end = kwargs.get(
        "load_best_model_at_end", eval_protocol == "train_val_test"
    )

    # ---- Build Seq2SeqTrainingArguments ----
    output_dir = f"./results/{run_name}/{kwargs['seed']}"
    save_strategy = kwargs["save_strategy"]
    save_total_limit = kwargs["save_total_limit"] if save_strategy != "no" else None
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=kwargs["num_train_epochs"],
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accu_step,
        logging_dir="./logs",
        logging_steps=kwargs["logging_steps"],
        bf16=kwargs["bf16"],
        gradient_checkpointing=kwargs["gradient_checkpointing"],
        optim=kwargs["optim"],
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps if save_strategy != "no" else None,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        learning_rate=kwargs["learning_rate"],
        remove_unused_columns=False,  # tokenized on the fly via set_transform
        eval_accumulation_steps=kwargs.get("eval_accumulation_steps", real_batch_size),
        weight_decay=kwargs["weight_decay"],
        warmup_ratio=kwargs["warmup_ratio"],
        lr_scheduler_type="cosine",
        max_grad_norm=kwargs["max_grad_norm"],
        seed=kwargs["seed"],
        predict_with_generate=True,
        generation_max_length=max_length,
        generation_num_beams=1,
    )

    # ---- Build custom optimizer (muon / lora_rite / lora_pro) + lr scheduler ----
    optimizer = None
    lr_scheduler = None
    if optimizer_name is not None:
        opt_name = optimizer_name.lower()

        if opt_name in {"adamw", "adamw_torch"}:
            # No custom optimizer; let Trainer build its default from training_args.optim.
            pass
        elif opt_name == "muon":
            muon_params = []
            adamw_params = []
            adamw_emb_params = []
            adamw_names = []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                is_embedding = ("shared" in name) or ("lm_head" in name)
                if p.ndim >= 2 and not is_embedding:
                    muon_params.append(p)
                else:
                    adamw_names.append(name)
                    if is_embedding:
                        adamw_emb_params.append(p)
                    else:
                        adamw_params.append(p)

            log.info(
                f"Muon param groups -> muon: {len(muon_params)}, "
                f"adamw: {len(adamw_params)}, emb: {len(adamw_emb_params)}"
            )
            if adamw_names:
                log.info("AdamW params:")
                for n in adamw_names:
                    log.info(f"  {n}")
            else:
                log.info("AdamW params: <none>")

            optimizer = Muon(
                muon_params,
                lr=training_args.learning_rate,
                weight_decay=training_args.weight_decay,
                adamw_params=adamw_params + adamw_emb_params,
                adamw_lr=training_args.learning_rate,
                adamw_wd=training_args.weight_decay,
                ns_using_pe=using_pe,
                ns_dtype=kwargs["ns_dtype"]
            )
        elif opt_name == "lora_rite":
            # Pair LoRA A/B parameters; LoRARite expects them ordered (A, B)
            pair_map = {}
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if "lora_A" in name:
                    key = name.split("lora_A")[0]
                    pair_map.setdefault(key, {})["A"] = p
                elif "lora_B" in name:
                    key = name.split("lora_B")[0]
                    pair_map.setdefault(key, {})["B"] = p

            lora_pairs = []
            missing_pairs = 0
            for key, parts in pair_map.items():
                if "A" in parts and "B" in parts:
                    lora_pairs.extend([parts["A"], parts["B"]])
                else:
                    missing_pairs += 1
            if missing_pairs > 0:
                log.warning(
                    f"LoRARite found {missing_pairs} LoRA submodules without complete A/B pairs; they are skipped."
                )

            if len(lora_pairs) == 0:
                log.warning(
                    "LoRARite selected but no LoRA parameters were found. Falling back to Trainer default optimizer."
                )
            else:
                optimizer = LoRARite(
                    lora_pairs,
                    betas=(training_args.adam_beta1, training_args.adam_beta2),
                    eps=training_args.adam_epsilon,
                    lr=training_args.learning_rate,
                    weight_decay=training_args.weight_decay,
                )
                log.info(f"LoRARite param groups -> {len(lora_pairs)//2} LoRA adapter pairs.")
        elif opt_name in {"lorapro", "lora_pro", "lora-pro"}:
            named_params = {
                "params": [
                    (name, p)
                    for name, p in model.named_parameters()
                    if p.requires_grad
                ]
            }
            lora_scaler = kwargs.get("lora_scaler") or 1.0
            if len(named_params["params"]) == 0:
                log.warning(
                    "LoRAPro selected but no trainable parameters were found. Falling back to Trainer default optimizer."
                )
            else:
                optimizer = LoRAProAdamW(
                    named_params,
                    lora_scaler=lora_scaler,
                    betas=(training_args.adam_beta1, training_args.adam_beta2),
                    eps=training_args.adam_epsilon,
                    lr=training_args.learning_rate,
                    weight_decay=training_args.weight_decay,
                )
                log.info(f"LoRAPro param groups -> {len(named_params['params'])} params.")
        else:
            log.warning(f"Unknown optimizer_name={optimizer_name}, falling back to Trainer default.")
            optimizer = None

        if optimizer is not None:
            # Match HF Trainer's effective dataloader length under DDP.
            world_size = max(1, training_args.world_size)
            num_update_steps_per_epoch = math.ceil(
                len(train_dataset)
                / (training_args.per_device_train_batch_size * world_size * training_args.gradient_accumulation_steps)
            )
            max_train_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
            warmup_steps = (
                training_args.warmup_steps
                if training_args.warmup_steps > 0
                else int(training_args.warmup_ratio * max_train_steps)
            )
            lr_scheduler = get_scheduler(
                name=training_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_train_steps,
            )

    # ---- Callbacks, metric fn, data collator ----
    # EarlyStoppingCallback requires load_best_model_at_end=True, which we only
    # set under train_val_test. Skip it under train_val to avoid an assert at train start.
    # AdaLoraCallback is no-op for non-AdaLoRA models (hasattr guard inside).
    callbacks = [
        EvalSampleOutputCallback(tokenizer=tokenizer, max_length=max_length),
        AdaLoraCallback(),
    ]
    if eval_protocol == "train_val_test":
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=kwargs["early_stopping_patience"]
            )
        )

    compute_metrics_fn = build_compute_metrics(tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if kwargs["bf16"] else None,
    )

    # ---- Trainer ----
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
        optimizers=(optimizer, lr_scheduler)
        if optimizer is not None
        else (None, None),
    )

    # ---- Run pretrained eval, train, then post-train eval (+ test if applicable) ----
    trainer.evaluate()
    trainer.train()
    final_eval_results = trainer.evaluate()
    if test_dataset is not None:
        test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        final_eval_results.update(test_results)

    # ---- Persist eval results outside output_dir so they survive cleanup ----
    eval_results_dir = os.path.dirname(output_dir)  # ./results/{run_name}
    eval_results_path = os.path.join(eval_results_dir, "eval_results.json")
    os.makedirs(eval_results_dir, exist_ok=True)
    with open(eval_results_path, "w") as f:
        json.dump(final_eval_results, f, indent=2)
    log.info(f"Saved eval results to {eval_results_path}")

    if kwargs.get("cleanup_outputs", True):
        cleanup_run_outputs(training_args.output_dir)

    return model
