import torch


def _build_muon_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    from muon import Muon

    muon_params = []
    adamw_params = []
    adamw_emb_params = []

    embedding_keywords = (
        "embedding", "embeddings",
        "position_embedding", "positional_embedding",
        "class_embedding", "patch_embedding",
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_lora_adapter = "lora_A" in name or "lora_B" in name
        is_embedding = any(kw in name for kw in embedding_keywords) and not is_lora_adapter

        if param.ndim >= 2 and not is_embedding:
            muon_params.append(param)
        elif is_embedding:
            adamw_emb_params.append(param)
        else:
            adamw_params.append(param)

    print(
        f"[Muon] param groups -> muon: {len(muon_params)}, "
        f"adamw: {len(adamw_params)}, emb: {len(adamw_emb_params)}"
    )

    return Muon(
        muon_params,
        lr=args.lr,
        momentum=args.muon_momentum,
        nesterov=True,
        backend=args.muon_backend,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.wd,
        adamw_params=adamw_params + adamw_emb_params if (adamw_params or adamw_emb_params) else None,
        adamw_lr=args.lr,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_wd=args.wd,
        ns_using_pe=args.ns_using_pe,
        ns_dtype=args.ns_dtype,
    )
