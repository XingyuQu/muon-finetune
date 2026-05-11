from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

HF_CHECKPOINT_MARKERS = {
    "config.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "model.safetensors.index.json",
}


def _resolve_existing_path(path_value: str, base_dir: Path) -> Path:
    """Resolve a path against (absolute, cwd, base_dir); return original if none exist."""
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = base_dir / path
    if repo_candidate.exists():
        return repo_candidate
    return path


def _find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Return the highest-step checkpoint-<N>/ child (fallback: most recently modified)."""
    if not output_dir.exists():
        return None
    candidates: list[tuple[Optional[int], float, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not (path.is_dir() or path.is_file()):
            continue
        match = re.search(r"checkpoint-(\d+)", path.name)
        step = int(match.group(1)) if match else None
        candidates.append((step, path.stat().st_mtime, path))
    if not candidates:
        return None
    with_steps = [item for item in candidates if item[0] is not None]
    if with_steps:
        return max(with_steps, key=lambda item: item[0])[2]
    return max(candidates, key=lambda item: item[1])[2]


def _looks_like_hf_checkpoint(path: Path) -> bool:
    """True if `path` contains a recognizable HF checkpoint marker file."""
    return any((path / marker).exists() for marker in HF_CHECKPOINT_MARKERS)


def resolve_checkpoint(checkpoint_arg: str) -> str:
    """Resolve a checkpoint path. If a parent dir is given without HF markers,
    pick the latest `checkpoint-<step>` subdir."""
    path = _resolve_existing_path(checkpoint_arg, REPO_ROOT)
    if path.exists() and path.is_dir() and not _looks_like_hf_checkpoint(path):
        latest = _find_latest_checkpoint(path)
        if latest is not None:
            return str(latest)
    return str(path) if path.exists() else checkpoint_arg
