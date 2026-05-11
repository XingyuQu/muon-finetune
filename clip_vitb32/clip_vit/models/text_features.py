import os
import json
import hashlib
from typing import List, Tuple, Optional

import torch
from transformers import CLIPProcessor, CLIPModel


def render_prompt(template: str, classname: str) -> str:
    """Format a prompt template with the given classname (positional or ``{c}``)."""
    return template.format(classname, c=classname)


@torch.no_grad()
def compute_text_features_with_templates(
    classnames: List[str],
    templates: List[str],
    model: CLIPModel,
    device: str = "cpu",
) -> Tuple[torch.Tensor, float]:
    """Build per-class template-ensembled normalized text features and return them with logit_scale."""
    model = model.to(device).eval()
    processor = CLIPProcessor.from_pretrained(model.name_or_path)

    feats = []
    for cname in classnames:
        prompts = [render_prompt(t, cname) for t in templates]
        enc = processor(text=prompts, padding=True, return_tensors="pt").to(device)
        txt = model.get_text_features(**enc)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        mean = txt.mean(dim=0, keepdim=True)
        mean = mean / mean.norm(dim=-1, keepdim=True)
        feats.append(mean)

    text_feats = torch.cat(feats, dim=0)
    logit_scale = float(model.logit_scale.exp().item())
    return text_feats, logit_scale


def textfeat_config_hash(model_name: str, templates: List[str], classnames: List[str]) -> str:
    """Stable MD5 hash of (model, templates, classnames) used as a text-features cache key."""
    payload = {
        "model": model_name,
        "templates": [str(t) for t in templates],
        "classnames": list(classnames),
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.md5(data).hexdigest()


def save_text_features(
    path_dir: str,
    name: str,
    text_feats: torch.Tensor,
    logit_scale: float,
    classnames: List[str],
    config_hash: str,
):
    """Cache text features (with logit_scale, classnames, config hash) to ``{path_dir}/{name}.pt``."""
    os.makedirs(path_dir, exist_ok=True)
    torch.save(
        {
            "text_feats": text_feats.cpu(),
            "logit_scale": logit_scale,
            "classnames": list(classnames),
            "config_hash": config_hash,
        },
        os.path.join(path_dir, f"{name}.pt"),
    )


def load_text_features(
    path_dir: str,
    name: str,
    device: str = "cpu",
    expected_classnames: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, float]:
    """Load cached text features and ``logit_scale``; raise if classnames mismatch."""
    data = torch.load(os.path.join(path_dir, f"{name}.pt"), map_location=device)
    cached_names = data.get("classnames")
    if expected_classnames is not None:
        if not cached_names or cached_names != list(expected_classnames):
            raise ValueError("Cached classnames mismatch")
    return data["text_feats"].to(device), float(data["logit_scale"])
