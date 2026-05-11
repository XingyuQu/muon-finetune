from typing import List

import torchvision.datasets as tvds

from .torchvision_task import TorchvisionTask
from .huggingface_task import HFTask
from .cb_base import cb_classnames_fallback


class DTDTask(TorchvisionTask):
    """Describable Textures Dataset (torchvision)."""
    name = "dtd"
    dataset_cls = tvds.DTD
    val_split = "val"

    def get_classnames(self, base_ds) -> List[str]:
        return [str(c).replace("_", " ") for c in base_ds.classes]


class GTSRBTask(TorchvisionTask):
    """German Traffic Sign Recognition Benchmark (torchvision)."""
    name = "gtsrb"
    dataset_cls = tvds.GTSRB

    def get_classnames(self, base_ds) -> List[str]:
        return cb_classnames_fallback(self.cb_dataset_key())


class SVHNTask(TorchvisionTask):
    """Street View House Numbers (torchvision); remaps label 10 -> 0 to match digit semantics."""
    name = "svhn"
    dataset_cls = tvds.SVHN

    def make_dataset(self, split: str, root: str, tf):
        def _svhn_target(y):
            return 0 if int(y) == 10 else int(y)
        return self.dataset_cls(
            root=root, split=split, download=True,
            transform=tf, target_transform=_svhn_target,
        )

    def get_classnames(self, base_ds) -> List[str]:
        return [str(i) for i in range(10)]


class RESISC45Task(HFTask):
    """RESISC45 remote-sensing scene classification (HF: tanganke/resisc45)."""
    name = "resisc45"
    repo_id = "tanganke/resisc45"
    image_key = "image"
    label_key = "label"
    val_ratio = 0.1


class SUN397Task(HFTask):
    """SUN397 scene recognition (HF: tanganke/sun397)."""
    name = "sun397"
    repo_id = "tanganke/sun397"
    image_key = "image"
    label_key = "label"

    def post_process_classnames(self, classnames):
        """Normalize SUN397 names: drop super-category prefix and convert underscores to spaces."""
        cleaned = []
        for cname in classnames:
            name = str(cname).strip()
            if "/" in name:
                name = name.split("/")[-1]
            name = name.replace("_", " ").strip()
            cleaned.append(name)
        return cleaned


class StanfordCarsTask(HFTask):
    """Stanford Cars (HF: tanganke/stanford_cars); auto-detects 1-based labelling and offsets to 0-based."""
    name = "stanford_cars"
    repo_id = "tanganke/stanford_cars"
    image_key = "image"
    label_key = "label"

    def infer_label_offset(self, labels, num_classes: int) -> int:
        min_label = min(labels)
        max_label = max(labels)
        if min_label == 1 and max_label == num_classes:
            return -1
        return 0
