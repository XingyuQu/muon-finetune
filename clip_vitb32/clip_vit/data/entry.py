from typing import Tuple, List, Optional, Literal

from torch.utils.data import DataLoader

from ..tasks import get_task
from ..tasks.base import SplitMode


def get_loaders_and_classnames(
    dataset: str,
    root="./data",
    batch_size=128,
    num_workers=4,
    device="cpu",
    seed: int = 0,
    split_save_dir: Optional[str] = None,
    split_mode: SplitMode = "train-val-test",
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build (train, val, test) DataLoaders plus class names for the given dataset."""
    task = get_task(dataset)
    return task.build_loaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
        split_save_dir=split_save_dir,
        split_mode=split_mode,
    )
