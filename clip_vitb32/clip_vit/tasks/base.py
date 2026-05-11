from typing import Tuple, List, Optional, Literal
from torch.utils.data import DataLoader

_CB_DATASET_ALIASES = {
    "stanford_cars": "cars",
    "cars": "cars",
    "svhn": "svhn",
}

SplitMode = Literal["train-val-test", "train-test"]


class Task:
    """Base class describing a CLIP fine-tuning dataset (loaders + classnames + templates)."""

    name: str = ""

    def cb_dataset_key(self) -> str:
        """Return the clip_benchmark key for this task (with aliases applied)."""
        key = self.name.strip().lower()
        return _CB_DATASET_ALIASES.get(key, key)

    def cb_templates_and_classnames(
        self,
        fallback_classnames: Optional[List[str]],
        language: str = "en",
    ) -> Tuple[List[str], List[str], str]:
        """Return (classnames, templates, dataset_key) from clip_benchmark assets."""
        from .cb_base import cb_templates_and_classnames
        return cb_templates_and_classnames(self.cb_dataset_key(), fallback_classnames, language=language)

    def build_loaders(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        device: str,
        seed: int,
        split_save_dir: str,
        split_mode: SplitMode = "train-val-test",
    ) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
        """Subclasses build (train, val, test) DataLoaders plus class names."""
        raise NotImplementedError
