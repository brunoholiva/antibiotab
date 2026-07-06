"""Shared utilities for TabPFN model device management."""

from __future__ import annotations

from tabpfn import TabPFNClassifier


def move_model_to_device(model: TabPFNClassifier, device: str) -> TabPFNClassifier:
    """Transfer model parameters and KV-caches to *device*.

    ``model.to(device)`` does **not** move the KV-cache
    (``executor_.kv_caches``), which must be transferred separately.

    Parameters
    ----------
    model : TabPFNClassifier
        Fitted classifier.
    device : str
        Target device (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    TabPFNClassifier
        The same model instance, for chaining.
    """
    model.to(device)
    model.device = device
    if hasattr(model, "executor_") and hasattr(model.executor_, "kv_caches"):
        model.executor_.kv_caches = [
            cache.to(device) for cache in model.executor_.kv_caches
        ]
    return model
