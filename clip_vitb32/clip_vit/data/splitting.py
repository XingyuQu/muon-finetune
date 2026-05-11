import random
from typing import List, Tuple, Dict

import torch

DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


def _dataset_labels(ds, label_key=None):
    base = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    indices = ds.indices if isinstance(ds, torch.utils.data.Subset) else None

    labels = None
    if label_key and hasattr(base, "column_names") and label_key in base.column_names:
        labels = list(base[label_key])
    if labels is None and hasattr(base, "labels"):
        labels = list(base.labels)
    if labels is None:
        labels = [base[i][1] for i in range(len(base))]
    if indices is not None:
        labels = [labels[i] for i in indices]
    return labels


def _stratified_split_indices(labels: List, split_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    n = len(labels)
    rng = random.Random(seed)
    idx_by_label: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        idx_by_label.setdefault(str(label), []).append(idx)

    main_idx: List[int] = []
    split_idx: List[int] = []
    for idxs in idx_by_label.values():
        rng.shuffle(idxs)
        n_class = len(idxs)
        n_split = int(round(n_class * split_ratio))
        if split_ratio > 0 and n_class > 1 and n_split == 0:
            n_split = 1
        if n_split >= n_class:
            n_split = n_class - 1
        split_idx.extend(idxs[:n_split])
        main_idx.extend(idxs[n_split:])

    rng.shuffle(main_idx)
    rng.shuffle(split_idx)
    return main_idx, split_idx


def split_train_val(ds_train, val_ratio: float, seed: int, label_key=None):
    """Stratified split of ``ds_train`` into (train, val) Subsets by ``val_ratio``."""
    labels = _dataset_labels(ds_train, label_key=label_key)
    train_idx, val_idx = _stratified_split_indices(labels, val_ratio, seed)
    return torch.utils.data.Subset(ds_train, train_idx), torch.utils.data.Subset(ds_train, val_idx)
