import os
import random
from dataclasses import dataclass

import numpy as np
import torch

MODEL_NAME = "openai/clip-vit-base-patch32"


@dataclass
class Paths:
    """Filesystem layout for run artifacts (ckpts, heads, text feats, splits)."""
    root: str = ""
    ckpts: str = "ckpts"
    heads: str = "heads"
    textfeats: str = "text_feats"
    splits: str = "splits"

    def ensure(self):
        """Create the configured subdirectories under ``root``."""
        for sub in (self.ckpts, self.heads, self.textfeats, self.splits):
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)


def set_seed(seed: int):
    """Seed Python / NumPy / PyTorch RNGs and return the seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[seed] {seed}")
    return seed


def get_device(arg_device: str) -> str:
    """Return the device string passed by the user (passthrough)."""
    return arg_device
