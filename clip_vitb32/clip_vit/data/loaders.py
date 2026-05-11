import torch
from torch.utils.data import DataLoader


def build_loaders(ds_train, ds_val, ds_test, batch_size: int, num_workers: int, device: str):
    """Wrap (train, val, test) datasets in DataLoaders with shared worker / pin-memory settings."""
    pin = device == "cuda"

    def _loader_kwargs(shuffle: bool):
        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin,
            "persistent_workers": num_workers > 0,
            "shuffle": shuffle,
        }

    train_loader = DataLoader(ds_train, **_loader_kwargs(True))
    val_loader = DataLoader(ds_val, **_loader_kwargs(False))
    test_loader = DataLoader(ds_test, **_loader_kwargs(False))
    return train_loader, val_loader, test_loader
