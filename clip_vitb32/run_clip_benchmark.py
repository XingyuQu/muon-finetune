"""clip_benchmark CLI extension: registers a fine-tuned CLIP-ViT loader and runs zero-shot eval."""
import json
import logging
import os
import random
import sys
from contextlib import nullcontext
from itertools import product
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn.functional as F
import torchvision.datasets as tvds
from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPTokenizer
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file

log = logging.getLogger(__name__)

HF_DATASET_MAP = {
    "sun397": "tanganke/sun397",
    "resisc45": "tanganke/resisc45",
    "cars": "tanganke/stanford_cars",
    "stanford_cars": "tanganke/stanford_cars",
}
DATASET_ALIASES = {
}
TV_DATASET_MAP = {
    "svhn": "svhn",
}
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1

POSITION_IDS_KEYS = {
    "text_model.embeddings.position_ids",
    "vision_model.embeddings.position_ids",
}

try:
    from safetensors.torch import load_file as safe_load_file
    _HAS_SAFETENSORS = True
except Exception:
    safe_load_file = None
    _HAS_SAFETENSORS = False


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_split_seed(args) -> int:
    value = os.getenv("CB_SPLIT_SEED")
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return int(getattr(args, "seed", 0) or 0)


def _load_split_manifest() -> Optional[Dict]:
    path = os.getenv("CB_SPLIT_FILE")
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split manifest not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load split manifest: {path}") from exc


def _resolve_base_model(model_name: str) -> str:
    name = (model_name or "").strip()
    if name in {"ViT-B-32", "ViT-B-32-quickgelu"}:
        return "openai/clip-vit-base-patch32"
    if not name:
        return "openai/clip-vit-base-patch32"
    return name


class HFCLIPTokenizerWrapper:
    """Adapter giving a HF CLIPTokenizer the simple ``tokenizer(list_of_strs)`` calling convention."""

    def __init__(self, tokenizer: CLIPTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts):
        return self.tokenizer(texts, padding=True, return_tensors="pt")


class HFCLIPWrapper(torch.nn.Module):
    """Wrap a HF CLIPModel to expose ``encode_text`` / ``encode_image`` like open_clip."""

    def __init__(self, model: CLIPModel):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def encode_text(self, tokens):
        if hasattr(tokens, "input_ids"):
            input_ids = tokens.input_ids
            attention_mask = getattr(tokens, "attention_mask", None)
        elif isinstance(tokens, dict):
            input_ids = tokens.get("input_ids")
            attention_mask = tokens.get("attention_mask")
        else:
            input_ids = tokens
            attention_mask = None
        return self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    def encode_image(self, images):
        return self.model.get_image_features(pixel_values=images)


def _load_clip_vit_state(ckpt_path: str) -> Tuple[Optional[dict], Dict]:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None, {}
    state = torch.load(ckpt_path, map_location="cpu")
    meta = {}
    if isinstance(state, dict) and "model_state" in state:
        meta = state.get("meta", {}) or {}
        state = state["model_state"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    if any("lora_" in k for k in state.keys()):
        log.info("LoRA adapters detected in %s; merging for clip_benchmark.", ckpt_path)
    return state, meta

def _load_pretrained_state_dict(model_name: str, cache_dir: Optional[str] = None) -> Dict[str, torch.Tensor]:
    resolved = None
    if _HAS_SAFETENSORS:
        try:
            resolved = cached_file(
                model_name,
                SAFE_WEIGHTS_NAME,
                cache_dir=cache_dir,
                _raise_exceptions_for_missing_entries=False,
            )
        except Exception:
            resolved = None
        if resolved and resolved.endswith(".safetensors"):
            state_dict = safe_load_file(resolved)
            for key in POSITION_IDS_KEYS:
                state_dict.pop(key, None)
            return state_dict

    try:
        resolved = cached_file(
            model_name,
            WEIGHTS_NAME,
            cache_dir=cache_dir,
            _raise_exceptions_for_missing_entries=False,
        )
    except Exception:
        resolved = None
    if not resolved:
        raise FileNotFoundError(f"Could not find weights for {model_name}")
    state_dict = torch.load(resolved, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    for key in POSITION_IDS_KEYS:
        state_dict.pop(key, None)
    return state_dict


def _strip_peft_prefix(name: str) -> str:
    if name.startswith("vision_model.base_model.model."):
        return "vision_model." + name[len("vision_model.base_model.model."):]
    if name.startswith("vision_model.model."):
        return "vision_model." + name[len("vision_model.model."):]
    return name


def _merge_lora_state(state: dict, meta: Dict) -> dict:
    if not any(".lora_" in k for k in state.keys()):
        return state
    alpha = None
    if isinstance(meta, dict):
        lora_meta = meta.get("lora") or {}
        if isinstance(lora_meta, dict):
            alpha = lora_meta.get("alpha") or lora_meta.get("lora_alpha")
    merged = {}
    lora_a = {}
    lora_b = {}
    for name, tensor in state.items():
        name = _strip_peft_prefix(name)
        if ".lora_A." in name:
            base = (
                name.replace(".lora_A.default.weight", ".weight")
                    .replace(".lora_A.weight", ".weight")
            )
            lora_a[base] = tensor
            continue
        if ".lora_B." in name:
            base = (
                name.replace(".lora_B.default.weight", ".weight")
                    .replace(".lora_B.weight", ".weight")
            )
            lora_b[base] = tensor
            continue
        if ".base_layer.weight" in name:
            base = name.replace(".base_layer.weight", ".weight")
            merged[base] = tensor
            continue
        if ".base_layer.bias" in name:
            base = name.replace(".base_layer.bias", ".bias")
            merged[base] = tensor
            continue
        merged[name] = tensor

    for base, A in lora_a.items():
        B = lora_b.get(base)
        if B is None:
            continue
        r = A.shape[0]
        scale = float(alpha) / float(r) if alpha is not None else 1.0
        delta = (B @ A) * scale
        if base in merged:
            merged[base] = merged[base] + delta.to(merged[base].dtype)
        else:
            merged[base] = delta
    return merged


def _apply_clip_vit_weights(model: CLIPModel, state: dict) -> None:
    vision_state = {
        k: v for k, v in state.items()
        if k.startswith("vision_model.") or k.startswith("visual_projection.")
    }
    missing, unexpected = model.load_state_dict(vision_state, strict=False)
    if missing:
        missing = [k for k in missing if k.startswith(("vision_model.", "visual_projection."))]
        if missing:
            log.warning("Missing keys when loading vision weights: %s", missing)
    if unexpected:
        log.warning("Unexpected keys when loading vision weights: %s", unexpected)


def load_clip_vit_ft(model_name: str, pretrained: str, cache_dir: str, device: str):
    """clip_benchmark loader for fine-tuned CLIP-ViT checkpoints (handles LoRA-merged weights)."""
    base_model_name = _resolve_base_model(model_name)
    ckpt_state, ckpt_meta = _load_clip_vit_state(pretrained)

    if ckpt_state is None and pretrained:
        state_dict = _load_pretrained_state_dict(pretrained, cache_dir=cache_dir)
        config = CLIPConfig.from_pretrained(pretrained, cache_dir=cache_dir)
        model = CLIPModel(config)
        model.load_state_dict(state_dict, strict=True)
        processor_source = pretrained
    else:
        state_dict = _load_pretrained_state_dict(base_model_name, cache_dir=cache_dir)
        config = CLIPConfig.from_pretrained(base_model_name, cache_dir=cache_dir)
        model = CLIPModel(config)
        model.load_state_dict(state_dict, strict=True)
        processor_source = base_model_name
        if ckpt_state is not None:
            ckpt_state = _merge_lora_state(ckpt_state, ckpt_meta)
            _apply_clip_vit_weights(model, ckpt_state)

    model = HFCLIPWrapper(model).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(processor_source, cache_dir=cache_dir)
    tokenizer = CLIPTokenizer.from_pretrained(processor_source, cache_dir=cache_dir)

    def transform(image):
        return image_processor(images=image, return_tensors="pt")["pixel_values"][0]

    return model, transform, HFCLIPTokenizerWrapper(tokenizer)

def _as_list(x):
    if not x:
        return []
    return x if isinstance(x, list) else [x]

def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _select_split(ds_dict, split):
    if split in ds_dict:
        return split
    if split == "val":
        return "validation" if "validation" in ds_dict else ("val" if "val" in ds_dict else None)
    if split == "test":
        return "test" if "test" in ds_dict else None
    if split == "train":
        return "train" if "train" in ds_dict else None
    return None


def _label_key(value) -> str:
    try:
        return str(int(value))
    except Exception:
        return str(value)

def _dataset_labels(ds, label_key: Optional[str] = None):
    base = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    indices = ds.indices if isinstance(ds, torch.utils.data.Subset) else None

    labels = None
    if label_key and hasattr(base, "column_names") and label_key in base.column_names:
        try:
            labels = list(base[label_key])
        except Exception:
            labels = None
    if labels is None and hasattr(base, "targets"):
        labels = list(base.targets)
    if labels is None and hasattr(base, "labels"):
        labels = list(base.labels)
    if labels is None and hasattr(base, "y"):
        labels = list(base.y)
    if labels is None and hasattr(base, "samples"):
        try:
            labels = [s[1] for s in base.samples]
        except Exception:
            labels = None
    if labels is None:
        try:
            labels = [base[i][1] for i in range(len(base))]
        except Exception:
            labels = None
    if labels is None:
        return None
    if indices is not None:
        labels = [labels[i] for i in indices]
    return labels

def _stratified_split_indices(labels, split_ratio: float, seed: int):
    n = len(labels)
    if split_ratio <= 0:
        return list(range(n)), []
    if split_ratio >= 1:
        return [], list(range(n))

    rng = random.Random(seed)
    idx_by_label = {}
    for idx, label in enumerate(labels):
        idx_by_label.setdefault(_label_key(label), []).append(idx)

    main_idx = []
    split_idx = []
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

def _split_train_val(ds_train, val_ratio: float, seed: int, label_key: Optional[str] = None):
    labels = _dataset_labels(ds_train, label_key=label_key)
    if labels is None or len(labels) == 0:
        raise ValueError("Stratified split required but labels are unavailable for this dataset.")
    train_idx, val_idx = _stratified_split_indices(labels, val_ratio, seed)
    return torch.utils.data.Subset(ds_train, train_idx), torch.utils.data.Subset(ds_train, val_idx)


def _split_full_train_val_test(ds_full, train_ratio: float, val_ratio: float, seed: int,
                               label_key: Optional[str] = None):
    labels = _dataset_labels(ds_full, label_key=label_key)
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    if labels is None or len(labels) == 0:
        raise ValueError("Stratified split required but labels are unavailable for this dataset.")
    train_idx, temp_idx = _stratified_split_indices(labels, val_ratio + test_ratio, seed)
    if not temp_idx:
        return (
            torch.utils.data.Subset(ds_full, train_idx),
            torch.utils.data.Subset(ds_full, []),
            torch.utils.data.Subset(ds_full, []),
        )
    temp_labels = [labels[i] for i in temp_idx]
    _, val_idx_rel = _stratified_split_indices(
        temp_labels,
        val_ratio / max(val_ratio + test_ratio, 1e-8),
        seed + 1,
    )
    val_idx = [temp_idx[i] for i in val_idx_rel]
    test_idx = [i for i in temp_idx if i not in set(val_idx)]
    return (
        torch.utils.data.Subset(ds_full, train_idx),
        torch.utils.data.Subset(ds_full, val_idx),
        torch.utils.data.Subset(ds_full, test_idx),
    )


def _prepare_hf_splits(
    ds_dict,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    manifest: Optional[Dict] = None,
    dataset_name: Optional[str] = None,
):
    if manifest:
        if dataset_name and manifest.get("dataset") not in {dataset_name, manifest.get("dataset_key")}:
            raise ValueError(f"Split manifest dataset mismatch: {manifest.get('dataset')} vs {dataset_name}")
        sources = manifest.get("sources", {})
        indices = manifest.get("indices", {})
        def _select(split: str):
            source = sources.get(split, "train")
            if source not in ds_dict:
                raise ValueError(f"Split '{source}' missing in HF dataset")
            base = ds_dict[source]
            idx = indices.get(split)
            return torch.utils.data.Subset(base, idx) if idx is not None else base
        return _select("train"), _select("val"), _select("test"), manifest.get("classnames")
    train_split = _select_split(ds_dict, "train")
    if train_split is None:
        raise ValueError("HF dataset missing train split")
    train = ds_dict[train_split]
    _, label_key = _infer_keys(train)
    val_split = _select_split(ds_dict, "val")
    test_split = _select_split(ds_dict, "test")
    val = ds_dict[val_split] if val_split else None
    test = ds_dict[test_split] if test_split else None
    if val is None and test is None:
        train, val, test = _split_full_train_val_test(
            train,
            train_ratio=1.0 - val_ratio - test_ratio,
            val_ratio=val_ratio,
            seed=seed,
            label_key=label_key,
        )
    elif val is None:
        train, val = _split_train_val(train, val_ratio=val_ratio, seed=seed, label_key=label_key)
    elif test is None:
        train, test = _split_train_val(train, val_ratio=test_ratio, seed=seed, label_key=label_key)
    return train, val, test, None


def _build_cb_dataset(dataset_key: str, root: str, split: str, transform):
    from clip_benchmark.datasets.builder import build_dataset
    return build_dataset(dataset_key, root=root, split=split, transform=transform, download=True)

def _ensure_vtab_deps(dataset_key: str) -> None:
    if not dataset_key.startswith("vtab/"):
        return
    try:
        import task_adaptation  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "vtab datasets require task_adaptation. Install with: "
            "pip install git+https://github.com/google-research/task_adaptation.git"
        ) from exc
    try:
        import tensorflow_datasets  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "vtab datasets require tensorflow_datasets (and tensorflow). Install with: "
            "pip install tensorflow-datasets tensorflow"
        ) from exc


def _prepare_cb_splits(
    dataset_key: str,
    root: str,
    transform,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    manifest: Optional[Dict] = None,
):
    _ensure_vtab_deps(dataset_key)
    if manifest:
        if manifest.get("dataset_key") not in {dataset_key, manifest.get("dataset")}:
            raise ValueError(f"Split manifest dataset mismatch: {manifest.get('dataset_key')} vs {dataset_key}")
        sources = manifest.get("sources", {})
        indices = manifest.get("indices", {})
        base_cache = {}
        def _get_base(source: str):
            if source not in base_cache:
                base_cache[source] = _build_cb_dataset(dataset_key, root, source, transform)
            return base_cache[source]
        def _select(split: str):
            source = sources.get(split, "train")
            base = _get_base(source)
            idx = indices.get(split)
            if isinstance(base, torch.utils.data.IterableDataset):
                return base
            return torch.utils.data.Subset(base, idx) if idx is not None else base
        return _select("train"), _select("val"), _select("test"), manifest.get("classnames")
    splits = {}
    for split in ("train", "val", "test"):
        try:
            splits[split] = _build_cb_dataset(dataset_key, root, split, transform)
        except Exception:
            splits[split] = None
    train = splits.get("train")
    if train is None:
        try:
            train = _build_cb_dataset(dataset_key, root, "trainval", transform)
        except Exception:
            train = None
    if train is None:
        raise ValueError(f"clip_benchmark dataset '{dataset_key}' has no train or trainval split")
    val = splits.get("val")
    test = splits.get("test")
    if val is None and test is None:
        train, val, test = _split_full_train_val_test(
            train,
            train_ratio=1.0 - val_ratio - test_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    elif val is None:
        train, val = _split_train_val(train, val_ratio=val_ratio, seed=seed)
    elif test is None:
        train, test = _split_train_val(train, val_ratio=test_ratio, seed=seed)
    return train, val, test, None

def _prepare_tv_splits(
    dataset_key: str,
    ds_train,
    ds_test,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    manifest: Optional[Dict] = None,
):
    if manifest:
        if manifest.get("dataset_key") not in {dataset_key, manifest.get("dataset")}:
            raise ValueError(f"Split manifest dataset mismatch: {manifest.get('dataset_key')} vs {dataset_key}")
        sources = manifest.get("sources", {})
        indices = manifest.get("indices", {})
        source_map = {"train": ds_train, "test": ds_test}
        def _select(split: str):
            src = sources.get(split, "train")
            base = source_map.get(src)
            if base is None:
                raise ValueError(f"Missing source split '{src}' for {split}")
            idx = indices.get(split)
            return torch.utils.data.Subset(base, idx) if idx is not None else base
        return _select("train"), _select("val"), _select("test")

    if ds_test is None:
        ds_train, ds_val, ds_test = _split_full_train_val_test(
            ds_train,
            train_ratio=1.0 - val_ratio - test_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    else:
        ds_train, ds_val = _split_train_val(ds_train, val_ratio=val_ratio, seed=seed)
    return ds_train, ds_val, ds_test

def _unwrap_subset(ds):
    if isinstance(ds, torch.utils.data.Subset):
        return ds.dataset
    return ds

def _infer_keys(ds):
    image_key = "image"
    label_key = "label"
    base = _unwrap_subset(ds)
    if hasattr(base, "features"):
        for k, v in base.features.items():
            if image_key == "image" and type(v).__name__ == "Image":
                image_key = k
            if label_key == "label" and hasattr(v, "names"):
                label_key = k
    return image_key, label_key

def _label_names(ds, label_key):
    base = _unwrap_subset(ds)
    if hasattr(base, "features") and label_key in base.features:
        feat = base.features[label_key]
        if hasattr(feat, "names") and feat.names:
            return [str(n).replace("_", " ") for n in feat.names]
        num_classes = getattr(feat, "num_classes", 0) or 0
        if num_classes:
            return [str(i) for i in range(int(num_classes))]
    return None

def _extract_classnames_from_dataset(ds) -> Optional[List[str]]:
    base = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    classes = getattr(base, "classes", None)
    if classes:
        return [str(c).replace("_", " ") for c in classes]
    class_to_idx = getattr(base, "class_to_idx", None)
    if isinstance(class_to_idx, dict) and class_to_idx:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        return [str(idx_to_class[i]).replace("_", " ") for i in range(len(idx_to_class))]
    sign_names = getattr(base, "sign_names", None) or getattr(base, "signnames", None)
    if sign_names:
        return [str(c).replace("_", " ") for c in sign_names]
    targets = getattr(base, "targets", None)
    if targets is not None and len(targets) > 0:
        try:
            num_classes = int(max(targets)) + 1
            return [str(i) for i in range(num_classes)]
        except Exception:
            return None
    return None

def _fix_svhn_labels(ds):
    if not hasattr(ds, "labels"):
        return ds
    try:
        import numpy as np
        labels = np.array(ds.labels)
        labels[labels == 10] = 0
        ds.labels = labels
    except Exception:
        ds.labels = [0 if int(x) == 10 else int(x) for x in ds.labels]
    return ds

def _infer_num_classes(ds) -> int:
    classnames = _extract_classnames_from_dataset(ds)
    if classnames:
        return len(classnames)
    return 0

def _build_output_path(args, dataset_name, task):
    pretrained_slug = os.path.basename(args.pretrained) if os.path.isfile(args.pretrained) else args.pretrained
    pretrained_slug_full_path = args.pretrained.replace("/", "_") if os.path.isfile(args.pretrained) else args.pretrained
    dataset_slug = dataset_name.replace("/", "_")
    return args.output.format(
        model=args.model,
        pretrained=pretrained_slug,
        pretrained_full_path=pretrained_slug_full_path,
        task=task,
        dataset=dataset_slug,
        language=args.language,
    )


def _to_device_tokens(tokens, device: str):
    if isinstance(tokens, dict):
        return {k: v.to(device) for k, v in tokens.items()}
    if hasattr(tokens, "to"):
        return tokens.to(device)
    return tokens


def _get_logit_scale(model) -> float:
    base = model
    if hasattr(model, "model"):
        base = model.model
    logit_scale = getattr(base, "logit_scale", None)
    if logit_scale is None:
        return 1.0
    if isinstance(logit_scale, torch.Tensor):
        return float(logit_scale.exp().item())
    return float(logit_scale)


def _render_prompt(template, cname: str) -> str:
    if callable(template):
        return template(cname)
    try:
        return template.format(cname)
    except (KeyError, IndexError, ValueError):
        try:
            return template.format(c=cname, classname=cname)
        except Exception:
            return (
                template.replace("{c}", cname)
                .replace("{classname}", cname)
                .replace("{}", cname)
            )


def _accuracy_topk(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.item() / batch_size)
    return res


def _build_zeroshot_classifier(model, tokenizer, classnames, templates, device: str, amp_dtype):
    model.eval()
    text_features = []
    with torch.no_grad():
        for cname in classnames:
            prompts = [_render_prompt(t, cname) for t in templates]
            tokens = _to_device_tokens(tokenizer(prompts), device)
            ctx = torch.amp.autocast(device_type=device, dtype=amp_dtype) if amp_dtype else nullcontext()
            with ctx:
                txt = model.encode_text(tokens)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            mean = txt.mean(dim=0, keepdim=True)
            mean = mean / mean.norm(dim=-1, keepdim=True)
            text_features.append(mean)
    classifier = torch.cat(text_features, dim=0)
    return classifier, _get_logit_scale(model)


def _zeroshot_eval(model, dataloader, tokenizer, classnames, templates, device: str, amp: Optional[str], return_preds: bool = False):
    amp_dtype = None
    if amp == "bf16":
        amp_dtype = torch.bfloat16
    elif amp == "fp16":
        amp_dtype = torch.float16

    classifier, logit_scale = _build_zeroshot_classifier(
        model, tokenizer, classnames, templates, device, amp_dtype
    )
    classifier = classifier.to(device)
    num_classes = len(classnames)
    topk = (1, 5) if num_classes >= 5 else (1,)
    total = 0
    loss_sum = 0.0
    acc1_sum = 0.0
    acc5_sum = 0.0
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            ctx = torch.amp.autocast(device_type=device, dtype=amp_dtype) if amp_dtype else nullcontext()
            with ctx:
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * image_features @ classifier.t()
                loss = F.cross_entropy(logits, target)
            batch_size = target.size(0)
            total += batch_size
            loss_sum += float(loss.item()) * batch_size
            accs = _accuracy_topk(logits, target, topk=topk)
            acc1_sum += accs[0] * batch_size
            if len(accs) > 1:
                acc5_sum += accs[1] * batch_size
            if return_preds:
                preds_all.append(logits.argmax(dim=1).cpu())
                targets_all.append(target.cpu())
    metrics = {"acc1": acc1_sum / total, "loss": loss_sum / total}
    if num_classes >= 5:
        metrics["acc5"] = acc5_sum / total
    if return_preds:
        preds = torch.cat(preds_all) if preds_all else torch.empty(0, dtype=torch.long)
        targets = torch.cat(targets_all) if targets_all else torch.empty(0, dtype=torch.long)
        return metrics, preds, targets
    return metrics


def _macro_f1_recall(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[float, float]:
    if preds.numel() == 0 or targets.numel() == 0 or num_classes <= 0:
        return 0.0, 0.0
    preds = preds.to(torch.long)
    targets = targets.to(torch.long)
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(targets, preds):
        conf[int(t), int(p)] += 1
    tp = conf.diag().float()
    fn = conf.sum(dim=1).float() - tp
    fp = conf.sum(dim=0).float() - tp
    denom_recall = tp + fn
    denom_prec = tp + fp
    recall = torch.where(denom_recall > 0, tp / denom_recall, torch.zeros_like(tp))
    precision = torch.where(denom_prec > 0, tp / denom_prec, torch.zeros_like(tp))
    f1 = torch.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    valid = denom_recall > 0
    mean_recall = float(recall[valid].mean().item()) if valid.any() else 0.0
    macro_f1 = float(f1[valid].mean().item()) if valid.any() else 0.0
    return mean_recall, macro_f1

def _safe_balanced_accuracy_score(y_true, y_pred) -> float:
    import numpy as np
    from sklearn.metrics import confusion_matrix

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    if y_true_np.size == 0:
        return 0.0
    labels = np.unique(y_true_np)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(cm) / cm.sum(axis=1)
    per_class = per_class[~np.isnan(per_class)]
    return float(per_class.mean()) if per_class.size else 0.0


def _standard_zeroshot_metrics(model, dataloader, tokenizer, classnames, templates, device: str, amp: Optional[str]) -> Dict:
    try:
        from clip_benchmark.metrics import zeroshot_classification as zsc
    except Exception:
        return {}
    orig_balanced = getattr(zsc, "balanced_accuracy_score", None)
    zsc.balanced_accuracy_score = _safe_balanced_accuracy_score
    amp_flag = False
    if isinstance(amp, str):
        amp_flag = amp in {"bf16", "fp16"}
    elif isinstance(amp, bool):
        amp_flag = amp
    try:
        return zsc.evaluate(
            model,
            dataloader,
            tokenizer,
            classnames,
            templates,
            device=device,
            amp=amp,
        )
    except TypeError:
        try:
            return zsc.evaluate(
                model,
                dataloader,
                tokenizer,
                classnames,
                templates,
                device,
                amp_flag,
            )
        except Exception:
            return {}
    finally:
        if orig_balanced is not None:
            zsc.balanced_accuracy_score = orig_balanced

def _floatify_metrics(metrics: Dict) -> Dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif hasattr(v, "item"):
            try:
                out[k] = float(v.item())
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _resolve_classnames_templates(
    dataset_key: str,
    args,
    ds_raw=None,
    label_key: Optional[str] = None,
    classnames_override: Optional[List[str]] = None,
):
    import clip_benchmark.datasets.builder as cb_builder
    def _lookup(mapping: Dict, key: str):
        if key in mapping:
            return mapping[key]
        alt = key.replace("/", "_")
        return mapping.get(alt)
    short_key = dataset_key.split("/", 1)[1] if dataset_key.startswith("vtab/") else None
    folder = os.path.dirname(cb_builder.__file__)
    templates_map = _load_json(args.custom_template_file) if args.custom_template_file else _load_json(
        os.path.join(folder, f"{args.language}_zeroshot_classification_templates.json")
    )
    classnames_map = _load_json(args.custom_classname_file) if args.custom_classname_file else _load_json(
        os.path.join(folder, f"{args.language}_classnames.json")
    )
    classnames = list(classnames_override) if classnames_override else None
    if classnames is None and ds_raw is not None and label_key is not None:
        classnames = _label_names(ds_raw, label_key)
    if classnames is None and ds_raw is not None:
        classnames = _extract_classnames_from_dataset(ds_raw)
    if classnames is None:
        classnames = _lookup(classnames_map, dataset_key)
    templates = _lookup(templates_map, dataset_key) or templates_map.get("default")
    if short_key:
        if classnames is None:
            classnames = _lookup(classnames_map, short_key)
        if templates is None:
            templates = _lookup(templates_map, short_key)
    if classnames is None:
        raise ValueError(f"Classes not specified for {dataset_key}")
    if templates is None:
        raise ValueError(f"Templates not specified for {dataset_key}")
    return classnames, templates

def _run_hf_eval(args):
    from datasets import load_dataset
    from clip_benchmark.datasets.builder import get_dataset_collate_fn, get_dataset_default_task
    from clip_benchmark.models import load_clip

    dataset_name = args.dataset
    dataset_key = "cars" if dataset_name in {"cars", "stanford_cars"} else dataset_name
    task = get_dataset_default_task(dataset_key) if args.task == "auto" else args.task
    if task != "zeroshot_classification":
        raise ValueError(f"HF datasets only support zeroshot_classification (got {task})")

    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    torch.manual_seed(getattr(args, "seed", 0) or 0)
    split_seed = _get_split_seed(args)
    val_ratio = _env_float("CB_VAL_RATIO", DEFAULT_VAL_RATIO)
    test_ratio = _env_float("CB_TEST_RATIO", DEFAULT_TEST_RATIO)

    dataset_root = args.dataset_root.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace("/", "-"))
    model, transform, tokenizer = load_clip(
        model_type=args.model_type,
        model_name=args.model,
        pretrained=args.pretrained,
        cache_dir=args.model_cache_dir,
        device=args.device,
    )
    model.eval()

    ds_dict = load_dataset(HF_DATASET_MAP[dataset_name], cache_dir=dataset_root)
    manifest = _load_split_manifest()
    ds_train, ds_val, ds_test, classnames_override = _prepare_hf_splits(
        ds_dict, split_seed, val_ratio, test_ratio, manifest=manifest, dataset_name=dataset_name
    )
    split = args.split.lower()
    if split in {"val", "validation"}:
        ds_raw = ds_val
    elif split == "test":
        ds_raw = ds_test
    else:
        ds_raw = ds_train
    if ds_raw is None:
        raise ValueError(f"Split '{args.split}' is unavailable for {dataset_name}")
    image_key, label_key = _infer_keys(ds_raw)

    class _HFDataset(torch.utils.data.Dataset):
        def __init__(self, ds, image_key, label_key, transform):
            self.ds = ds
            self.image_key = image_key
            self.label_key = label_key
            self.transform = transform
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            item = self.ds[idx]
            img = item[self.image_key]
            if hasattr(img, "convert"):
                img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, int(item[self.label_key])

    dataset = _HFDataset(ds_raw, image_key, label_key, transform)
    collate_fn = get_dataset_collate_fn(dataset_key)
    effective_workers = args.num_workers
    if isinstance(dataset, torch.utils.data.IterableDataset) and effective_workers > 0:
        log.warning("IterableDataset detected; forcing num_workers=0 to avoid TF worker crashes.")
        effective_workers = 0
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": effective_workers,
        "collate_fn": collate_fn,
    }
    if not isinstance(dataset, torch.utils.data.IterableDataset):
        loader_kwargs["shuffle"] = False
    dataloader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    classnames, templates = _resolve_classnames_templates(
        dataset_key, args, ds_raw, label_key, classnames_override=classnames_override
    )
    dataloader.dataset.classes = classnames
    custom_metrics, preds, targets = _zeroshot_eval(
        model,
        dataloader,
        tokenizer,
        classnames,
        templates,
        device=args.device,
        amp=args.amp,
        return_preds=True,
    )
    standard_metrics = _standard_zeroshot_metrics(
        model,
        dataloader,
        tokenizer,
        classnames,
        templates,
        device=args.device,
        amp=args.amp,
    )
    custom_metrics = _floatify_metrics(custom_metrics)
    standard_metrics = _floatify_metrics(standard_metrics)
    num_classes = len(classnames)
    mean_recall, macro_f1 = _macro_f1_recall(preds, targets, num_classes)
    metrics = {}
    sources = {}
    std_acc = standard_metrics.get("acc1") or standard_metrics.get("acc")
    if std_acc is not None:
        metrics["acc1"] = float(std_acc)
        sources["acc1"] = "standard_benchmark"
    else:
        metrics["acc1"] = float(custom_metrics.get("acc1", 0.0))
        sources["acc1"] = "custom_logic"
    std_acc5 = standard_metrics.get("acc5")
    if std_acc5 is not None:
        metrics["acc5"] = float(std_acc5)
        sources["acc5"] = "standard_benchmark"
    elif "acc5" in custom_metrics:
        metrics["acc5"] = float(custom_metrics["acc5"])
        sources["acc5"] = "custom_logic"
    std_mpr = standard_metrics.get("mean_per_class_recall")
    if std_mpr is not None:
        metrics["mean_per_class_recall"] = float(std_mpr)
        sources["mean_per_class_recall"] = "standard_benchmark"
    else:
        metrics["mean_per_class_recall"] = float(mean_recall)
        sources["mean_per_class_recall"] = "custom_logic"
    std_f1 = standard_metrics.get("f1_score") or standard_metrics.get("f1")
    if std_f1 is not None:
        metrics["f1_score"] = float(std_f1)
        sources["f1_score"] = "standard_benchmark"
    else:
        metrics["f1_score"] = float(macro_f1)
        sources["f1_score"] = "custom_logic"
    metrics["loss"] = float(custom_metrics.get("loss", 0.0))
    sources["loss"] = "custom_logic"
    if standard_metrics:
        metrics["standard"] = standard_metrics
    metrics["custom"] = custom_metrics
    metrics["sources"] = sources

    output = _build_output_path(args, dataset_name, task)
    dump = {
        "dataset": dataset_name,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": task,
        "metrics": metrics,
        "language": args.language,
    }
    if args.verbose:
        print(f"Dump results to: {output}")
    with open(output, "w") as f:
        json.dump(dump, f)
    return 0


def _run_cb_eval(args):
    from clip_benchmark.datasets.builder import get_dataset_collate_fn, get_dataset_default_task
    from clip_benchmark.models import load_clip

    dataset_name = args.dataset
    dataset_key = dataset_name
    task = get_dataset_default_task(dataset_key) if args.task == "auto" else args.task
    if task != "zeroshot_classification":
        raise ValueError(f"Only zeroshot_classification is supported (got {task})")

    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    torch.manual_seed(getattr(args, "seed", 0) or 0)
    split_seed = _get_split_seed(args)
    val_ratio = _env_float("CB_VAL_RATIO", DEFAULT_VAL_RATIO)
    test_ratio = _env_float("CB_TEST_RATIO", DEFAULT_TEST_RATIO)

    dataset_root = args.dataset_root.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace("/", "-"))
    model, transform, tokenizer = load_clip(
        model_type=args.model_type,
        model_name=args.model,
        pretrained=args.pretrained,
        cache_dir=args.model_cache_dir,
        device=args.device,
    )
    model.eval()

    manifest = _load_split_manifest()
    ds_train, ds_val, ds_test, classnames_override = _prepare_cb_splits(
        dataset_key, dataset_root, transform, split_seed, val_ratio, test_ratio, manifest=manifest
    )
    split = args.split.lower()
    if split in {"val", "validation"}:
        ds_raw = ds_val
    elif split == "test":
        ds_raw = ds_test
    else:
        ds_raw = ds_train
    if ds_raw is None:
        raise ValueError(f"Split '{args.split}' is unavailable for {dataset_name}")

    collate_fn = get_dataset_collate_fn(dataset_key)
    effective_workers = args.num_workers
    if isinstance(ds_raw, torch.utils.data.IterableDataset) and effective_workers > 0:
        log.warning("IterableDataset detected; forcing num_workers=0 to avoid TF worker crashes.")
        effective_workers = 0
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": effective_workers,
        "collate_fn": collate_fn,
    }
    if not isinstance(ds_raw, torch.utils.data.IterableDataset):
        loader_kwargs["shuffle"] = False
    dataloader = torch.utils.data.DataLoader(ds_raw, **loader_kwargs)

    classnames, templates = _resolve_classnames_templates(
        dataset_key, args, ds_raw, classnames_override=classnames_override
    )
    dataloader.dataset.classes = classnames
    custom_metrics, preds, targets = _zeroshot_eval(
        model,
        dataloader,
        tokenizer,
        classnames,
        templates,
        device=args.device,
        amp=args.amp,
        return_preds=True,
    )
    standard_metrics = _standard_zeroshot_metrics(
        model,
        dataloader,
        tokenizer,
        classnames,
        templates,
        device=args.device,
        amp=args.amp,
    )
    custom_metrics = _floatify_metrics(custom_metrics)
    standard_metrics = _floatify_metrics(standard_metrics)
    num_classes = len(classnames)
    mean_recall, macro_f1 = _macro_f1_recall(preds, targets, num_classes)
    metrics = {}
    sources = {}
    std_acc = standard_metrics.get("acc1") or standard_metrics.get("acc")
    if std_acc is not None:
        metrics["acc1"] = float(std_acc)
        sources["acc1"] = "standard_benchmark"
    else:
        metrics["acc1"] = float(custom_metrics.get("acc1", 0.0))
        sources["acc1"] = "custom_logic"
    std_acc5 = standard_metrics.get("acc5")
    if std_acc5 is not None:
        metrics["acc5"] = float(std_acc5)
        sources["acc5"] = "standard_benchmark"
    elif "acc5" in custom_metrics:
        metrics["acc5"] = float(custom_metrics["acc5"])
        sources["acc5"] = "custom_logic"
    std_mpr = standard_metrics.get("mean_per_class_recall")
    if std_mpr is not None:
        metrics["mean_per_class_recall"] = float(std_mpr)
        sources["mean_per_class_recall"] = "standard_benchmark"
    else:
        metrics["mean_per_class_recall"] = float(mean_recall)
        sources["mean_per_class_recall"] = "custom_logic"
    std_f1 = standard_metrics.get("f1_score") or standard_metrics.get("f1")
    if std_f1 is not None:
        metrics["f1_score"] = float(std_f1)
        sources["f1_score"] = "standard_benchmark"
    else:
        metrics["f1_score"] = float(macro_f1)
        sources["f1_score"] = "custom_logic"
    metrics["loss"] = float(custom_metrics.get("loss", 0.0))
    sources["loss"] = "custom_logic"
    if standard_metrics:
        metrics["standard"] = standard_metrics
    metrics["custom"] = custom_metrics
    metrics["sources"] = sources

    output = _build_output_path(args, dataset_name, task)
    dump = {
        "dataset": dataset_name,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": task,
        "metrics": metrics,
        "language": args.language,
    }
    if args.verbose:
        print(f"Dump results to: {output}")
    with open(output, "w") as f:
        json.dump(dump, f)
    return 0


def _run_tv_eval(args):
    from clip_benchmark.datasets.builder import get_dataset_collate_fn, get_dataset_default_task
    from clip_benchmark.models import load_clip

    dataset_name = args.dataset
    dataset_key = dataset_name
    task = get_dataset_default_task(dataset_key) if args.task == "auto" else args.task
    if task != "zeroshot_classification":
        raise ValueError(f"Only zeroshot_classification is supported (got {task})")

    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    torch.manual_seed(getattr(args, "seed", 0) or 0)
    split_seed = _get_split_seed(args)
    val_ratio = _env_float("CB_VAL_RATIO", DEFAULT_VAL_RATIO)
    test_ratio = _env_float("CB_TEST_RATIO", DEFAULT_TEST_RATIO)

    dataset_root = args.dataset_root.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace("/", "-"))
    model, transform, tokenizer = load_clip(
        model_type=args.model_type,
        model_name=args.model,
        pretrained=args.pretrained,
        cache_dir=args.model_cache_dir,
        device=args.device,
    )
    model.eval()

    if dataset_key == "svhn":
        ds_train = tvds.SVHN(root=dataset_root, split="train", download=True, transform=transform)
        ds_test = tvds.SVHN(root=dataset_root, split="test", download=True, transform=transform)
        _fix_svhn_labels(ds_train)
        _fix_svhn_labels(ds_test)
        classnames_override = [str(i) for i in range(10)]
    else:
        raise ValueError(f"Unsupported torchvision dataset: {dataset_key}")

    manifest = _load_split_manifest()
    ds_train, ds_val, ds_test = _prepare_tv_splits(
        dataset_key, ds_train, ds_test, split_seed, val_ratio, test_ratio, manifest=manifest
    )

    split = args.split.lower()
    if split in {"val", "validation"}:
        ds_raw = ds_val
    elif split == "test":
        ds_raw = ds_test
    else:
        ds_raw = ds_train
    if ds_raw is None:
        raise ValueError(f"Split '{args.split}' is unavailable for {dataset_name}")

    collate_fn = get_dataset_collate_fn(dataset_key)
    dataloader = torch.utils.data.DataLoader(
        ds_raw,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    classnames, templates = _resolve_classnames_templates(
        dataset_key, args, ds_raw, classnames_override=classnames_override
    )
    dataloader.dataset.classes = classnames
    custom_metrics, preds, targets = _zeroshot_eval(
        model,
        dataloader,
        tokenizer,
        classnames,
        templates,
        device=args.device,
        amp=args.amp,
        return_preds=True,
    )
    standard_metrics = _standard_zeroshot_metrics(
        model,
        dataloader,
        tokenizer,
        classnames,
        templates,
        device=args.device,
        amp=args.amp,
    )
    custom_metrics = _floatify_metrics(custom_metrics)
    standard_metrics = _floatify_metrics(standard_metrics)
    num_classes = len(classnames)
    mean_recall, macro_f1 = _macro_f1_recall(preds, targets, num_classes)
    metrics = {}
    sources = {}
    std_acc = standard_metrics.get("acc1") or standard_metrics.get("acc")
    if std_acc is not None:
        metrics["acc1"] = float(std_acc)
        sources["acc1"] = "standard_benchmark"
    else:
        metrics["acc1"] = float(custom_metrics.get("acc1", 0.0))
        sources["acc1"] = "custom_logic"
    std_acc5 = standard_metrics.get("acc5")
    if std_acc5 is not None:
        metrics["acc5"] = float(std_acc5)
        sources["acc5"] = "standard_benchmark"
    elif "acc5" in custom_metrics:
        metrics["acc5"] = float(custom_metrics["acc5"])
        sources["acc5"] = "custom_logic"
    std_mpr = standard_metrics.get("mean_per_class_recall")
    if std_mpr is not None:
        metrics["mean_per_class_recall"] = float(std_mpr)
        sources["mean_per_class_recall"] = "standard_benchmark"
    else:
        metrics["mean_per_class_recall"] = float(mean_recall)
        sources["mean_per_class_recall"] = "custom_logic"
    std_f1 = standard_metrics.get("f1_score") or standard_metrics.get("f1")
    if std_f1 is not None:
        metrics["f1_score"] = float(std_f1)
        sources["f1_score"] = "standard_benchmark"
    else:
        metrics["f1_score"] = float(macro_f1)
        sources["f1_score"] = "custom_logic"
    metrics["loss"] = float(custom_metrics.get("loss", 0.0))
    sources["loss"] = "custom_logic"
    if standard_metrics:
        metrics["standard"] = standard_metrics
    metrics["custom"] = custom_metrics
    metrics["sources"] = sources

    output = _build_output_path(args, dataset_name, task)
    dump = {
        "dataset": dataset_name,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": task,
        "metrics": metrics,
        "language": args.language,
    }
    if args.verbose:
        print(f"Dump results to: {output}")
    with open(output, "w") as f:
        json.dump(dump, f)
    return 0


def main() -> int:
    """CLI entry point: register ``clip_vit_ft`` and dispatch to HF/torchvision/clip_benchmark eval."""
    import clip_benchmark.models as cb_models

    if "clip_vit_ft" not in cb_models.TYPE2FUNC:
        cb_models.TYPE2FUNC["clip_vit_ft"] = load_clip_vit_ft
        if "clip_vit_ft" not in cb_models.MODEL_TYPES:
            cb_models.MODEL_TYPES.append("clip_vit_ft")

    import clip_benchmark.cli as cb_cli
    from clip_benchmark.datasets.builder import dataset_collection, get_dataset_collection_from_file
    from clip_benchmark.model_collection import get_model_collection_from_file, model_collection

    parser, base = cb_cli.get_parser_args()
    if not hasattr(base, "which"):
        parser.print_help()
        return 0
    if base.which == "build":
        cb_cli.main_build(base)
        return 0
    if base.which != "eval":
        return 0

    datasets = []
    for name in _as_list(base.dataset):
        if os.path.isfile(name):
            datasets.extend(get_dataset_collection_from_file(name))
        elif name in dataset_collection:
            datasets.extend(dataset_collection[name])
        else:
            datasets.append(name)
    datasets = [DATASET_ALIASES.get(d, d) for d in datasets]
    if any(d == "vtab/svhn" for d in datasets):
        raise ValueError("vtab/svhn is not supported; use torchvision svhn instead.")

    if base.pretrained_model:
        models = []
        for name in _as_list(base.pretrained_model):
            if os.path.isfile(name):
                models.extend(get_model_collection_from_file(name))
            elif name in model_collection:
                models.extend(model_collection[name])
            else:
                model, pretrained = name.split(",", 1)
                models.append((model, pretrained))
    else:
        models = list(product(_as_list(base.model), _as_list(base.pretrained)))

    languages = _as_list(base.language)
    for (model, pretrained) in models:
        for dataset in datasets:
            for language in languages:
                args = cb_cli.copy(base)
                args.model = model
                args.pretrained = pretrained
                args.dataset = dataset
                args.language = language
                if dataset in HF_DATASET_MAP:
                    _run_hf_eval(args)
                elif dataset in TV_DATASET_MAP:
                    _run_tv_eval(args)
                else:
                    _run_cb_eval(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
