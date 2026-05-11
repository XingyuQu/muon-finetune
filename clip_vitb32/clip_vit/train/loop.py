import os
from contextlib import nullcontext
from typing import Optional, Callable, Tuple, Dict, List

import torch
import torch.nn as nn

from ..optim.build import apply_grad_clipping, _as_list


def fmt_loss(loss_val: Optional[float]) -> str:
    """Format a possibly-None loss value to 4 decimal places (or the string ``None``)."""
    return "None" if loss_val is None else f"{loss_val:.4f}"


def train_epochs(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    num_epochs: int = 3,
    log_interval: int = 200,
    eval_interval: int = 1,
    device: str = "cuda",
    loss_fn: Optional[torch.nn.Module] = None,
    optimizer=None,
    scheduler=None,
    max_grad_norm: float = 0.0,
    amp_dtype: Optional[torch.dtype] = torch.bfloat16,
    on_step: Optional[Callable[[int, float], None]] = None,
    on_eval: Optional[Callable[[int, float, float], None]] = None,
    save_best_checkpoint: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_prefix: Optional[str] = None,
    model_state_fn: Optional[Callable[[], Dict]] = None,
    eval_fn: Optional[Callable[[nn.Module], Tuple[float, float]]] = None,
) -> Tuple[Dict[str, List[float]], Optional[str]]:
    """Run the standard train/eval loop and (optionally) save the best-val checkpoint."""
    assert optimizer is not None
    optimizers = _as_list(optimizer)
    schedulers = _as_list(scheduler)
    if val_loader is not None:
        assert eval_fn is not None

    model = model.to(device)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_ckpt_path = None
    total_steps = 0

    steps_per_epoch = len(train_loader)
    total_steps_all = num_epochs * steps_per_epoch

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            pixel_values = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device, dtype=amp_dtype) if amp_dtype else nullcontext():
                logits = model(pixel_values)
                loss = loss_fn(logits, labels)

            loss.backward()
            apply_grad_clipping(model, max_grad_norm)
            for opt in optimizers:
                opt.step()
            for sch in schedulers:
                sch.step()

            total_steps += 1

            if total_steps % log_interval == 0 or total_steps == 1:
                all_lrs = [g["lr"] for opt in optimizers for g in opt.param_groups]
                print(f"[train] epoch {epoch+1}/{num_epochs} step {total_steps}/{total_steps_all} loss={loss.item():.4f} lr={all_lrs[0]:.2e}")
                history["train_loss"].append(float(loss.item()))
                if on_step is not None:
                    on_step(total_steps, float(loss.item()))

        if val_loader is not None and (epoch + 1) % eval_interval == 0:
            val_loss, val_acc = eval_fn(model)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"[eval ] epoch {epoch+1}/{num_epochs} val_loss={fmt_loss(val_loss)} val_acc={val_acc*100:.2f}%")
            if on_eval is not None:
                on_eval(total_steps, val_loss, val_acc)

            if save_best_checkpoint and val_acc > best_val_acc:
                best_val_acc = val_acc
                if checkpoint_dir and checkpoint_prefix:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    ckpt_name = f"{checkpoint_prefix}_best_epoch{epoch+1}_valacc{val_acc*100:.2f}.pt"
                    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

                    if model_state_fn:
                        state_dict = model_state_fn()
                    else:
                        state_dict = {"model_state": model.state_dict()}

                    torch.save(state_dict, ckpt_path)
                    print(f"[save] Best checkpoint saved: {ckpt_path} (val_acc={val_acc*100:.2f}%)")

                    if best_ckpt_path and best_ckpt_path != ckpt_path and os.path.exists(best_ckpt_path):
                        os.remove(best_ckpt_path)
                    best_ckpt_path = ckpt_path

    return history, best_ckpt_path
