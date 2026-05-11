import torch
import torch.nn as nn
from transformers import CLIPModel


class ClipImageTextAligner(nn.Module):
    """CLIP image encoder + frozen text features as a classifier head."""

    def __init__(
        self,
        clip_model: CLIPModel,
        text_feats: torch.Tensor,
        logit_scale: float,
        normalize_img_embed: bool = True,
    ):
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.normalize = normalize_img_embed
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        self.register_buffer("text_feats", text_feats)
        self.register_buffer(
            "logit_scale", torch.tensor(logit_scale, dtype=text_feats.dtype)
        )

    def image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images and (optionally) L2-normalize the projected features."""
        out = self.vision_model(pixel_values=pixel_values)
        pooled = out.pooler_output
        feats = self.visual_projection(pooled)
        if self.normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return scaled image-to-text-feature logits over the class set."""
        feats = self.image_features(pixel_values)
        return self.logit_scale * feats @ self.text_feats.t()
