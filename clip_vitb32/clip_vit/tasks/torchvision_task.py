from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import DataLoader

from .base import Task, SplitMode
from ..transforms import clip_image_tf
from ..data.loaders import build_loaders
from ..data.splitting import split_train_val, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO
from ..data.manifest import load_split_manifest, save_split_manifest, build_subsets_from_manifest


class TorchvisionTask(Task):
    """Base task for datasets backed by ``torchvision.datasets`` (split-keyed constructors)."""

    dataset_cls = None
    train_split: str = "train"
    val_split: Optional[str] = None
    test_split: str = "test"

    def make_dataset(self, split: str, root: str, tf):
        """Instantiate the underlying torchvision dataset for ``split`` (override if signature differs)."""
        return self.dataset_cls(root=root, split=split, download=True, transform=tf)

    def get_classnames(self, base_ds) -> List[str]:
        """Subclasses return the human-readable class names."""
        raise NotImplementedError

    def _build_sources(self, root: str, tf) -> Dict[str, torch.utils.data.Dataset]:
        sources: Dict[str, torch.utils.data.Dataset] = {}
        sources[self.train_split] = self.make_dataset(self.train_split, root, tf)
        if self.val_split:
            sources[self.val_split] = self.make_dataset(self.val_split, root, tf)
        sources[self.test_split] = self.make_dataset(self.test_split, root, tf)
        return sources

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
        """Build loaders from the torchvision dataset, reusing or creating a split manifest."""
        tf = clip_image_tf()
        dataset_key = self.cb_dataset_key()
        sources = self._build_sources(root, tf)

        if split_mode == "train-test":
            ds_train = sources[self.train_split]
            ds_test = sources[self.test_split]
            ds_val = ds_test
            classnames = self.get_classnames(ds_train)
            return build_loaders(ds_train, ds_val, ds_test, batch_size, num_workers, device) + (classnames,)

        manifest = load_split_manifest(
            split_save_dir, dataset_key, seed=seed,
            val_ratio=DEFAULT_VAL_RATIO, test_ratio=DEFAULT_TEST_RATIO,
        )
        if manifest:
            ds_train, ds_val, ds_test = build_subsets_from_manifest(manifest, sources)
            classnames = manifest["classnames"]
            return build_loaders(ds_train, ds_val, ds_test, batch_size, num_workers, device) + (classnames,)

        ds_train = sources[self.train_split]
        ds_val = sources.get(self.val_split) if self.val_split else None
        ds_test = sources[self.test_split]

        if ds_val is None:
            ds_train, ds_val = split_train_val(ds_train, val_ratio=DEFAULT_VAL_RATIO, seed=seed)
            source_map = {"train": self.train_split, "val": self.train_split, "test": self.test_split}
        else:
            source_map = {"train": self.train_split, "val": self.val_split, "test": self.test_split}

        base_ds = ds_train.dataset if isinstance(ds_train, torch.utils.data.Subset) else ds_train
        classnames = self.get_classnames(base_ds)

        save_split_manifest(
            split_save_dir, dataset_name=self.name, dataset_key=dataset_key, kind="tv",
            sources=source_map, splits={"train": ds_train, "val": ds_val, "test": ds_test},
            classnames=classnames, seed=seed,
            val_ratio=DEFAULT_VAL_RATIO, test_ratio=DEFAULT_TEST_RATIO,
        )
        return build_loaders(ds_train, ds_val, ds_test, batch_size, num_workers, device) + (classnames,)
