from typing import Dict, Optional

import torch
from transformers import CLIPConfig, CLIPModel
from transformers.utils import WEIGHTS_NAME
from transformers.utils.hub import cached_file

POSITION_IDS_KEYS = {
    "text_model.embeddings.position_ids",
    "vision_model.embeddings.position_ids",
}


def load_pretrained_state_dict(model_name: str, cache_dir: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Fetch the HF CLIP weights for ``model_name`` and drop legacy position_ids keys."""
    resolved = cached_file(
        model_name,
        WEIGHTS_NAME,
        cache_dir=cache_dir,
        _raise_exceptions_for_missing_entries=False,
    )
    if not resolved:
        raise FileNotFoundError(f"Could not find weights for {model_name}")
    state_dict = torch.load(resolved, map_location="cpu")
    for key in POSITION_IDS_KEYS:
        state_dict.pop(key, None)
    return state_dict


def load_clip_base(model_name: str) -> CLIPModel:
    """Load a CLIP model with the text tower frozen and the vision tower trainable."""
    state_dict = load_pretrained_state_dict(model_name)
    config = CLIPConfig.from_pretrained(model_name)
    model = CLIPModel(config)
    model.load_state_dict(state_dict, strict=True)
    for p in model.text_model.parameters():
        p.requires_grad = False
    for p in model.text_projection.parameters():
        p.requires_grad = False
    model.logit_scale.requires_grad = False
    for p in model.vision_model.parameters():
        p.requires_grad = True
    for p in model.visual_projection.parameters():
        p.requires_grad = True
    return model
