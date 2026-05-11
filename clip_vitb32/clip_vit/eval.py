import json
import os
import subprocess
import sys
import tempfile
from typing import Tuple, Optional

import torch
import torch.nn as nn

from .config import MODEL_NAME
from .data.splitting import DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO
from .optim.build import optimizer_tag


def clip_benchmark_eval(
    model: nn.Module, args, split: str, tag: str,
) -> Tuple[Optional[float], float]:
    """Run zero-shot eval via the run_clip_benchmark.py subprocess and return (loss, acc)."""
    out_dir = os.path.join(args.save_root, "clip_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    opt_tag = optimizer_tag(args)
    run_sig = f"{args.dataset}_init-{args.init_method}_opt-{opt_tag}_lr-{args.lr}_seed-{args.seed}_{tag}"
    out_path = os.path.join(out_dir, f"{run_sig.replace('/', '_')}.json")
    print(f"[clip_benchmark] split={split} tag={tag} output={out_path}")

    fd, ckpt_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    meta = {
        "lora": {
            "enabled": args.init_method.lower() == "lora",
            "r": args.lora_r,
            "alpha": args.lora_alpha,
        }
    }
    torch.save({"model_state": model.state_dict(), "meta": meta}, ckpt_path)

    script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "run_clip_benchmark.py")
    )
    cmd = [
        sys.executable, script,
        "eval",
        "--model_type", "clip_vit_ft",
        "--model", MODEL_NAME,
        "--pretrained", ckpt_path,
        "--dataset", args.dataset,
        "--task", "zeroshot_classification",
        "--language", "en",
        "--dataset_root", args.data_root,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--output", out_path,
        "--split", split,
        "--quiet",
    ]

    env = os.environ.copy()
    env["CB_SPLIT_SEED"] = str(args.seed or 0)
    env["CB_VAL_RATIO"] = str(DEFAULT_VAL_RATIO)
    env["CB_TEST_RATIO"] = str(DEFAULT_TEST_RATIO)
    if args.split_file:
        if not os.path.isfile(args.split_file):
            raise FileNotFoundError(f"Split manifest not found: {args.split_file}")
        env["CB_SPLIT_FILE"] = args.split_file

    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    acc = metrics.get("acc1") or metrics.get("acc") or metrics.get("mean_per_class_recall")
    if acc is None:
        raise ValueError(f"clip_benchmark metrics missing acc in {out_path}")

    loss = metrics.get("loss")
    loss_val = float(loss) if loss is not None else None
    return loss_val, float(acc)
