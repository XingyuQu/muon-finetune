from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from .base import Task, SplitMode
from ..transforms import clip_image_tf
from ..data.loaders import build_loaders
from ..data.splitting import split_train_val, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO
from ..data.manifest import load_split_manifest, save_split_manifest, build_subsets_from_manifest


class HFImageDataset(torch.utils.data.Dataset):
    """Adapter that wraps a HF dataset row dict into ``(image_tensor, label_int)`` pairs."""

    def __init__(self, ds, image_key: str, label_key: str, transform, label_transform):
        self.ds = ds
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item[self.image_key].convert("RGB")
        img = self.transform(img)
        label = self.label_transform(item[self.label_key])
        return img, label


def _hf_unwrap(ds):
    if isinstance(ds, torch.utils.data.Subset):
        return ds.dataset
    return ds


def _hf_classnames(ds, label_key: str) -> List[str]:
    base = _hf_unwrap(ds)
    feat = base.features[label_key]
    return [str(n).replace("_", " ") for n in feat.names]


def _select_split(ds_dict, split: str) -> Optional[str]:
    return split if split in ds_dict else None


def _extract_labels(ds, label_key: str):
    base = _hf_unwrap(ds)
    labels = list(base[label_key])
    if isinstance(ds, torch.utils.data.Subset):
        return [labels[i] for i in ds.indices]
    return labels


class HFTask(Task):
    """Base task for datasets backed by Hugging Face ``datasets`` (image + label columns)."""

    repo_id: str = ""
    image_key: str = "image"
    label_key: str = "label"
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO

    def infer_label_offset(self, labels, num_classes: Optional[int]) -> int:
        """Return an integer offset added to raw labels (override for 1-based label spaces)."""
        return 0

    def post_process_classnames(self, classnames: List[str]) -> List[str]:
        """Hook to clean up class names from the HF features (default: identity)."""
        return classnames

    def _dataset_fingerprints(self, ds_dict, splits: List[str]) -> Dict[str, Optional[str]]:
        return {name: getattr(ds_dict[name], "_fingerprint", None) for name in splits}

    def _make_label_transform(self, classnames, label_offset, dataset_key):
        def _label_transform(val):
            return int(val) + label_offset
        return _label_transform

    def _wrap_datasets(self, ds_train_raw, ds_val_raw, ds_test_raw, classnames, dataset_key, tf):
        labels = _extract_labels(ds_train_raw, self.label_key)
        label_offset = self.infer_label_offset(labels, len(classnames))
        label_transform = self._make_label_transform(classnames, label_offset, dataset_key)

        def _wrap(ds):
            return HFImageDataset(ds, self.image_key, self.label_key, tf, label_transform=label_transform)

        return _wrap(ds_train_raw), _wrap(ds_val_raw), _wrap(ds_test_raw)

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
        """Build loaders from the HF repo, reusing or creating a stratified split manifest."""
        tf = clip_image_tf()
        ds_dict = load_dataset(self.repo_id, cache_dir=root)
        dataset_key = self.cb_dataset_key()
        val_ratio = self.val_ratio
        test_ratio = self.test_ratio

        manifest = load_split_manifest(
            split_save_dir, dataset_key, seed=seed,
            val_ratio=val_ratio, test_ratio=test_ratio,
        )
        if manifest and manifest.get("fingerprints"):
            current_fp = self._dataset_fingerprints(ds_dict, list(manifest["sources"].values()))
            saved_fp = manifest["fingerprints"]
            if any(current_fp[k] != saved_fp[k] for k in saved_fp):
                manifest = None

        if manifest:
            sources = manifest["sources"]
            source_datasets = {k: ds_dict[k] for k in set(sources.values())}
            ds_train_raw, ds_val_raw, ds_test_raw = build_subsets_from_manifest(manifest, source_datasets)
            classnames = manifest["classnames"]
            classnames = self.post_process_classnames(classnames)
            ds_train, ds_val, ds_test = self._wrap_datasets(
                ds_train_raw, ds_val_raw, ds_test_raw, classnames, dataset_key, tf,
            )
            return build_loaders(ds_train, ds_val, ds_test, batch_size, num_workers, device) + (classnames,)

        train_split = _select_split(ds_dict, "train")
        ds_train_raw = ds_dict[train_split]
        test_split = _select_split(ds_dict, "test")
        ds_test_raw = ds_dict[test_split]
        ds_val_raw = None

        if split_mode == "train-test":
            ds_val_raw = ds_test_raw
        else:
            ds_train_raw, ds_val_raw = split_train_val(
                ds_train_raw, val_ratio=val_ratio, seed=seed, label_key=self.label_key,
            )

        classnames = _hf_classnames(ds_train_raw, self.label_key)
        classnames = self.post_process_classnames(classnames)

        if split_mode != "train-test":
            sources = {
                "train": train_split,
                "val": train_split,
                "test": test_split,
            }
            fingerprints = self._dataset_fingerprints(ds_dict, list(sources.values()))
            save_split_manifest(
                split_save_dir, dataset_name=self.name, dataset_key=dataset_key, kind="hf",
                sources=sources, splits={"train": ds_train_raw, "val": ds_val_raw, "test": ds_test_raw},
                classnames=classnames, seed=seed,
                val_ratio=val_ratio, test_ratio=test_ratio,
                extra={"fingerprints": fingerprints},
            )

        ds_train, ds_val, ds_test = self._wrap_datasets(
            ds_train_raw, ds_val_raw, ds_test_raw, classnames, dataset_key, tf,
        )
        return build_loaders(ds_train, ds_val, ds_test, batch_size, num_workers, device) + (classnames,)
