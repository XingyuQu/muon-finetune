import os
import json
from typing import Tuple, List, Optional

import clip_benchmark.datasets.builder as cb_builder


def _cb_base_dir() -> str:
    return os.path.dirname(cb_builder.__file__)


def cb_classnames_fallback(dataset_key: str, language: str = "en") -> Optional[List[str]]:
    """Return the clip_benchmark fallback class names for ``dataset_key`` (or None)."""
    path = os.path.join(_cb_base_dir(), f"{language}_classnames.json")
    with open(path, "r", encoding="utf-8") as f:
        classnames_map = json.load(f)
    return classnames_map.get(dataset_key)


def cb_templates_and_classnames(
    dataset_key: str,
    fallback_classnames: Optional[List[str]],
    language: str = "en",
) -> Tuple[List[str], List[str], str]:
    """Look up clip_benchmark zero-shot templates and classnames for ``dataset_key``."""
    folder = _cb_base_dir()
    with open(os.path.join(folder, f"{language}_zeroshot_classification_templates.json"), "r", encoding="utf-8") as f:
        templates_map = json.load(f)
    with open(os.path.join(folder, f"{language}_classnames.json"), "r", encoding="utf-8") as f:
        classnames_map = json.load(f)

    classnames = fallback_classnames or classnames_map.get(dataset_key)
    templates = templates_map.get(dataset_key) or templates_map.get("default")
    return classnames, templates, dataset_key
