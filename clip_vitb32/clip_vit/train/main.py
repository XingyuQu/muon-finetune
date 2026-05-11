import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import math
import time

import torch

from clip_vit.config import Paths, set_seed, get_device, MODEL_NAME
from clip_vit.data.entry import get_loaders_and_classnames
from clip_vit.data.manifest import split_manifest_path
from clip_vit.models.clip_loader import load_clip_base
from clip_vit.models.lora import apply_lora_to_vision
from clip_vit.models.aligner import ClipImageTextAligner
from clip_vit.models.text_features import (
    compute_text_features_with_templates,
    textfeat_config_hash,
    save_text_features,
    load_text_features,
)
from clip_vit.optim.build import build_optimizer, optimizer_tag, _as_list
from clip_vit.train.loop import train_epochs, fmt_loss
from clip_vit.eval import clip_benchmark_eval
from clip_vit.tasks import get_task

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False


def main():
    """Entry point for the CLIP ViT-B/32 multi-dataset fine-tuning CLI."""
    parser = argparse.ArgumentParser("CLIP(ViT-B/32) finetune (multi-dataset)")
    parser.add_argument("--save-root", type=str, default="./runs_clip_mnist")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument(
        "--dataset", type=str, default="svhn",
        choices=["svhn", "gtsrb", "dtd", "stanford_cars", "resisc45", "sun397"],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--split-mode", type=str, default="train-val-test",
        choices=["train-val-test", "train-test"],
    )

    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"])

    parser.add_argument("--init-method", type=str, default="full_ft", choices=["full_ft", "lora"])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", type=str, nargs="*", default=["q_proj", "v_proj"])
    parser.add_argument("--use-rslora", action="store_true")
    parser.add_argument("--lora-visual-projection", action="store_true")

    parser.add_argument(
        "--optimizer", "--opt", dest="optimizer", type=str, default="adamw",
        choices=["adamw", "muon"],
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--log-param-groups", action="store_true")

    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-backend", type=str, default="newtonschulz5")
    parser.add_argument("--muon-backend-steps", type=int, default=5)
    parser.add_argument("--ns-dtype", type=str, default="bf16")
    parser.add_argument("--ns-using-pe", action="store_true")

    parser.add_argument("--wandb-project", type=str, default="clip_vit")
    parser.add_argument("--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="default")

    args = parser.parse_args()

    device = get_device(args.device)
    paths = Paths(root=args.save_root)
    paths.ensure()
    seed = set_seed(args.seed)
    args.seed = seed
    print("[device]", device)

    wandb_run = None
    run_name = args.wandb_name
    if run_name is None:
        opt_tag = optimizer_tag(args)
        run_name = (
            f"clip_vitb32_{args.dataset}_init-{args.init_method}_"
            f"opt-{opt_tag}_lr{args.lr}_seed{seed}"
        )
    if args.wandb_mode != "disabled":
        if not _HAS_WANDB:
            print("[warn] wandb is not installed; disabling wandb logging.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                mode=args.wandb_mode,
                group=args.wandb_group,
                config=vars(args),
                settings=wandb.Settings(init_timeout=300),
            )

    task = get_task(args.dataset)
    loaders_result = get_loaders_and_classnames(
        dataset=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=seed or 0,
        split_save_dir=os.path.join(paths.root, paths.splits),
        split_mode=args.split_mode,
    )
    train_loader, val_loader, test_loader, classnames = loaders_result
    has_val_split = (args.split_mode == "train-val-test")

    classnames, cb_templates, cb_dataset_key = task.cb_templates_and_classnames(classnames)
    if args.split_mode == "train-val-test":
        args.split_file = split_manifest_path(
            os.path.join(paths.root, paths.splits), cb_dataset_key,
        )
    else:
        args.split_file = None

    tf_dir = os.path.join(paths.root, paths.textfeats)
    config_hash = textfeat_config_hash(MODEL_NAME, cb_templates, classnames)
    tf_name = f"cache_{cb_dataset_key.replace('/', '_')}_{config_hash}"
    tf_path = os.path.join(tf_dir, f"{tf_name}.pt")

    text_feats = None
    logit_scale = None
    if os.path.exists(tf_path):
        print(f"[skip] Found existing text features at {tf_path}")
        try:
            text_feats, logit_scale = load_text_features(
                tf_dir, tf_name, device=device, expected_classnames=classnames,
            )
        except ValueError:
            text_feats = None
            logit_scale = None
    if text_feats is None or logit_scale is None:
        print(f"[compute] Building TEMPLATE-ENSEMBLED text features and saving to {tf_path}")
        clip_ref = load_clip_base(MODEL_NAME).to(device).eval()
        text_feats, logit_scale = compute_text_features_with_templates(
            classnames=classnames, templates=cb_templates, model=clip_ref, device=device,
        )
        save_text_features(tf_dir, tf_name, text_feats, logit_scale, classnames, config_hash)

    text_feats = text_feats.to(device)
    base_model = load_clip_base(MODEL_NAME).to(device)
    lora_in_use = args.init_method.lower() == "lora"
    if lora_in_use:
        apply_lora_to_vision(
            base_model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            use_rslora=args.use_rslora,
            lora_visual_projection=args.lora_visual_projection,
        )
    del base_model.text_model
    del base_model.text_projection
    torch.cuda.empty_cache()

    clf = ClipImageTextAligner(base_model, text_feats, logit_scale, normalize_img_embed=True)

    optimizer = build_optimizer(clf, args)
    optimizers = _as_list(optimizer)
    if args.log_param_groups:
        name_map = {id(p): n for n, p in clf.named_parameters() if p.requires_grad}
        opt_name = args.optimizer.lower()
        for opt in optimizers:
            for group in opt.param_groups:
                is_muon = group.get("is_muon")
                group_opt = "adamw" if opt_name == "muon" and is_muon is False else opt_name
                for p in group["params"]:
                    name = name_map.get(id(p), "<unnamed>")
                    print(f"[param-groups] optimizer={group_opt} name={name} shape={tuple(p.shape)}")

    steps_per_epoch = len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio) if args.warmup_ratio is not None else args.warmup_steps

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: (
        step / float(max(1, warmup_steps)) if step < warmup_steps
        else 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / float(max(1, total_steps - warmup_steps))))
    )) for opt in optimizers]

    if args.amp == "bf16":
        amp_dtype = torch.bfloat16 if device == "cuda" else None
    elif args.amp == "fp16":
        amp_dtype = torch.float16 if device == "cuda" else None
    else:
        amp_dtype = None

    pre_train_loss, pre_train_acc = clip_benchmark_eval(clf, args, "train", "pre_train")
    eval_split = "val" if has_val_split else "test"
    pre_val_loss, pre_val_acc = clip_benchmark_eval(clf, args, eval_split, "pre_val")
    print(
        f"[pre-eval] train_loss={fmt_loss(pre_train_loss)} train_acc={pre_train_acc*100:.2f}% "
        f"| {'val' if has_val_split else 'test'}_loss={fmt_loss(pre_val_loss)} "
        f"{'val' if has_val_split else 'test'}_acc={pre_val_acc*100:.2f}%"
    )

    history = {
        "pre_train_loss": pre_train_loss,
        "pre_train_acc": pre_train_acc,
        "pre_val_loss": pre_val_loss,
        "pre_val_acc": pre_val_acc,
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    def on_step(step, loss_value):
        if wandb_run is None:
            return
        lr_values = [g["lr"] for opt in optimizers for g in opt.param_groups]
        wandb_run.log({"train/loss": loss_value, "train/lr": lr_values[0], "train/step": step}, step=step)

    def on_eval(step, val_loss, val_acc):
        if wandb_run is None:
            return
        payload = {"eval/acc": val_acc, "eval/step": step}
        if val_loss is not None:
            payload["eval/loss"] = val_loss
        wandb_run.log(payload, step=step)

    if wandb_run is not None:
        wandb_run.log({
            "pre/train_loss": pre_train_loss,
            "pre/train_acc": pre_train_acc,
            "pre/val_loss": pre_val_loss,
            "pre/val_acc": pre_val_acc,
        })

    def model_state_fn():
        return {
            "model_state": clf.state_dict(),
            "meta": {
                "model_name": MODEL_NAME,
                "dataset": args.dataset,
                "classnames": classnames,
                "template_set": cb_dataset_key,
                "textfeat_name": tf_name,
                "init_method": args.init_method,
                "opt": args.optimizer,
                "lr": args.lr,
                "wd": args.wd,
                "lora": {
                    "enabled": lora_in_use,
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": args.lora_target_modules,
                    "use_rslora": args.use_rslora,
                    "lora_visual_projection": args.lora_visual_projection,
                },
                "optimizer": {
                    "muon_momentum": args.muon_momentum,
                    "muon_backend": args.muon_backend,
                    "muon_backend_steps": args.muon_backend_steps,
                    "ns_dtype": args.ns_dtype,
                    "ns_using_pe": args.ns_using_pe,
                },
                "seed": seed,
                "amp": args.amp,
                "num_epochs": args.num_epochs,
                "warmup_steps": warmup_steps,
                "device": device,
            },
        }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    opt_tag = optimizer_tag(args)
    ckpt_prefix = (
        f"vitb32_{args.dataset}_init-{args.init_method}_opt-{opt_tag}_"
        f"lr-{args.lr}_wd-{args.wd}_seed-{seed}_{stamp}"
    )

    eval_fn = lambda m: clip_benchmark_eval(m, args, eval_split, "val")

    if has_val_split:
        train_history, best_ckpt_path = train_epochs(
            model=clf,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            device=device,
            optimizer=optimizers,
            scheduler=schedulers,
            max_grad_norm=args.max_grad_norm,
            amp_dtype=amp_dtype,
            on_step=on_step,
            on_eval=on_eval,
            save_best_checkpoint=True,
            checkpoint_dir=os.path.join(paths.root, paths.ckpts),
            checkpoint_prefix=ckpt_prefix,
            model_state_fn=model_state_fn,
            eval_fn=eval_fn,
        )
        history["train_loss"].extend(train_history.get("train_loss", []))
        history["val_loss"].extend(train_history.get("val_loss", []))
        history["val_acc"].extend(train_history.get("val_acc", []))

        if best_ckpt_path and os.path.exists(best_ckpt_path):
            print(f"\n[test-eval] Loading best checkpoint: {best_ckpt_path}")
            ckpt_data = torch.load(best_ckpt_path, map_location=device)
            clf.load_state_dict(ckpt_data["model_state"])
            test_loss, test_acc = clip_benchmark_eval(clf, args, "test", "test")
            print(f"[test-eval] test_loss={fmt_loss(test_loss)} test_acc={test_acc*100:.2f}%")
            history["best_test_loss"] = test_loss
            history["best_test_acc"] = test_acc
            history["best_ckpt_path"] = best_ckpt_path
            if wandb_run is not None:
                payload = {"test/acc": test_acc, "test/step": total_steps}
                if test_loss is not None:
                    payload["test/loss"] = test_loss
                wandb_run.log(payload, step=total_steps)

        final_ckpt_path = best_ckpt_path
    else:
        train_history, _ = train_epochs(
            model=clf,
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=args.num_epochs,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            device=device,
            optimizer=optimizers,
            scheduler=schedulers,
            max_grad_norm=args.max_grad_norm,
            amp_dtype=amp_dtype,
            on_step=on_step,
            on_eval=on_eval,
            eval_fn=eval_fn,
        )
        history["train_loss"].extend(train_history.get("train_loss", []))
        history["val_loss"].extend(train_history.get("val_loss", []))
        history["val_acc"].extend(train_history.get("val_acc", []))

        ckpt_path = os.path.join(paths.root, paths.ckpts, f"{ckpt_prefix}.pt")
        torch.save(model_state_fn(), ckpt_path)
        print(f"[save] checkpoint -> {ckpt_path}")
        history["final_ckpt_path"] = ckpt_path
        final_ckpt_path = ckpt_path

        print(f"\n[test-eval] Evaluating final model on test set")
        test_loss, test_acc = clip_benchmark_eval(clf, args, "test", "final_test")
        print(f"[test-eval] test_loss={fmt_loss(test_loss)} test_acc={test_acc*100:.2f}%")
        history["final_test_loss"] = test_loss
        history["final_test_acc"] = test_acc
        if wandb_run is not None:
            payload = {"test/acc": test_acc, "test/step": total_steps}
            if test_loss is not None:
                payload["test/loss"] = test_loss
            wandb_run.log(payload, step=total_steps)

    if final_ckpt_path:
        hist_path = final_ckpt_path.replace(".pt", ".history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"[save] history -> {hist_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
