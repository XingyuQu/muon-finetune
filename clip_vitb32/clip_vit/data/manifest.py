import hashlib
import json
import os
from typing import Dict, List, Optional

import torch


def _extract_indices(ds) -> List[int]:
    if isinstance(ds, torch.utils.data.Subset):
        return [int(i) for i in ds.indices]
    return list(range(len(ds)))


def split_manifest_path(split_dir: str, dataset_key: str) -> str:
    """Return the JSON manifest path for ``dataset_key`` under ``split_dir``."""
    slug = dataset_key.replace("/", "_")
    return os.path.join(split_dir, f"splits_{slug}.json")


def _split_manifest_hash(seed: int, val_ratio: float, test_ratio: float) -> str:
    payload = {
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.md5(data).hexdigest()


def load_split_manifest(split_dir: str, dataset_key: str, seed: Optional[int] = None,
                        val_ratio: Optional[float] = None, test_ratio: Optional[float] = None) -> Optional[Dict]:
    """Load a split manifest if seed / ratios match, otherwise return None."""
    path = split_manifest_path(split_dir, dataset_key)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest.get("dataset_key") == dataset_key, f"dataset_key mismatch in {path}"
    if seed is not None and manifest.get("seed") is not None:
        if int(manifest.get("seed")) != int(seed):
            return None
    if val_ratio is not None and test_ratio is not None:
        expect = _split_manifest_hash(seed or 0, val_ratio, test_ratio)
        if manifest.get("split_hash") and manifest.get("split_hash") != expect:
            return None
    return manifest


def build_subsets_from_manifest(manifest: Dict, sources: Dict[str, torch.utils.data.Dataset]):
    """Rehydrate (train, val, test) subsets from a manifest plus source dataset map."""
    indices = manifest.get("indices") or {}
    src_map = manifest.get("sources") or {}

    def _subset(name: str):
        src = src_map.get(name, name)
        base = sources.get(src)
        if base is None:
            raise ValueError(f"Missing source split '{src}' for {name}")
        idx = indices.get(name)
        if idx is None:
            return base
        return torch.utils.data.Subset(base, idx)

    return _subset("train"), _subset("val"), _subset("test")


def save_split_manifest(
    split_dir: str,
    dataset_name: str,
    dataset_key: str,
    kind: str,
    sources: Dict[str, str],
    splits: Dict[str, torch.utils.data.Dataset],
    classnames: List[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    extra: Optional[Dict] = None,
) -> str:
    """Persist a split manifest (indices + classnames + hash) to JSON and return its path."""
    os.makedirs(split_dir, exist_ok=True)
    indices: Dict[str, List[int]] = {}
    for name, ds in splits.items():
        indices[name] = _extract_indices(ds)
    manifest = {
        "dataset": dataset_name,
        "dataset_key": dataset_key,
        "kind": kind,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "split_hash": _split_manifest_hash(seed, val_ratio, test_ratio),
        "sources": sources,
        "indices": indices,
        "classnames": list(classnames),
    }
    if extra:
        manifest.update(extra)
    path = split_manifest_path(split_dir, dataset_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return path
