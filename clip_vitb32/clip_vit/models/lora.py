from typing import List

import torch
import torch.nn as nn
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model


class _VisionWithProjection(nn.Module):
    def __init__(self, vision_model, visual_projection):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, pixel_values: torch.Tensor):
        out = self.vision_model(pixel_values=pixel_values)
        pooled = out.pooler_output
        return self.visual_projection(pooled)


def apply_lora_to_vision(
    clip_model: CLIPModel,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: List[str],
    use_rslora: bool = False,
    lora_visual_projection: bool = False,
) -> torch.nn.Module:
    """Inject LoRA adapters into the CLIP vision tower (and optionally the visual projection)."""
    for param in clip_model.vision_model.parameters():
        param.requires_grad = False
    for param in clip_model.visual_projection.parameters():
        param.requires_grad = False

    if lora_visual_projection:
        all_target_modules = list(target_modules) + ["visual_projection"]
        lora_config = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            bias="none", target_modules=all_target_modules, use_rslora=use_rslora,
        )
        wrapper = _VisionWithProjection(clip_model.vision_model, clip_model.visual_projection)
        peft_wrapper = get_peft_model(wrapper, lora_config)
        for name, param in peft_wrapper.named_parameters():
            param.requires_grad = "lora_" in name
        clip_model.vision_model = peft_wrapper.base_model.model.vision_model
        clip_model.visual_projection = peft_wrapper.base_model.model.visual_projection
        return peft_wrapper
    else:
        lora_config = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            bias="none", target_modules=target_modules, use_rslora=use_rslora,
        )
        peft_vision = get_peft_model(clip_model.vision_model, lora_config)
        for name, param in peft_vision.named_parameters():
            param.requires_grad = "lora_" in name
        clip_model.vision_model = peft_vision
        return peft_vision
