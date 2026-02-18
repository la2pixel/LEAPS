"""Training utilities for stride-level EMG dimensionality reduction models.

Mirrors trainer.py but for 3D stride data (n_strides, 101, 11).
Provides build_stride_model factory, train functions, and model name sets.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from leaps.evaluation.metrics import compute_eval_metrics
from leaps.models.base import EMGModel

LogFn = Callable[[dict[str, float], int], None] | None

# Model name sets — used by train_strides.py to pick the right train function
STRIDE_SKLEARN_MODELS = {"StridePCA", "StrideNMF", "StrideCNMF"}
STRIDE_PYTORCH_MODELS = {"StrideAE", "StrideVAE", "StrideMAE", "StrideWAE", "StrideFlatAE"}


def train_stride_sklearn_model(
    model: EMGModel,
    x_train: np.ndarray,
    x_val: np.ndarray | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, float]:
    """Train a sklearn-based stride model (StridePCA or StrideNMF).

    Args:
        model:    StridePCAModel or StrideNMFModel instance.
        x_train:  Training strides, shape (n_strides, 101, 11).
        x_val:    Validation strides, same shape convention.
        save_dir: Directory to save checkpoint (.pkl file).

    Returns:
        Metrics dict with train and val reconstruction quality.
    """
    metrics = model.fit(x_train, x_val)
    if save_dir is not None:
        save_path = Path(save_dir) / f"{model.name}_d{model.latent_dim}.pkl"
        model.save(save_path)
    return metrics


def train_stride_pytorch_model(
    model: EMGModel,
    x_train: np.ndarray,
    x_val: np.ndarray | None = None,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_dir: str | Path | None = None,
    log_fn: LogFn = None,
) -> dict[str, float]:
    """Train a PyTorch-based stride model (StrideAE, StrideVAE, StrideMAE).

    Args:
        model:      StrideAutoencoder, StrideVAE, or StrideMaskedAutoencoder.
        x_train:    Training strides, shape (n_strides, 101, 11).
        x_val:      Validation strides.
        epochs:     Number of training epochs.
        batch_size: Mini-batch size (default: 64, smaller than snapshot due to memory).
        lr:         Learning rate for Adam.
        save_dir:   Directory to save checkpoint (.pt file).
        log_fn:     Callback for logging: log_fn(metrics_dict, epoch).

    Returns:
        Metrics dict with final train and val metrics.
    """
    metrics = model.fit(
        x_train,
        x_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        log_fn=log_fn,
    )
    if save_dir is not None:
        save_path = Path(save_dir) / f"{model.name}_d{model.latent_dim}.pt"
        model.save(save_path)
    return metrics


def build_stride_model(
    model_name: str, latent_dim: int, n_channels: int = 11, stride_len: int = 101, **kwargs
) -> EMGModel:
    """Factory function to create a stride-level model by name.

    Args:
        model_name: One of "StridePCA", "StrideNMF", "StrideAE", "StrideVAE", "StrideMAE".
        latent_dim: Dimensionality of the latent space.
        n_channels: Number of EMG channels (default: 11).
        stride_len: Timepoints per stride (default: 101).
        **kwargs:   Model-specific args (beta for StrideVAE, mask_ratio for StrideMAE).

    Returns:
        An EMGModel instance ready for training.

    Raises:
        ValueError: If model_name is not recognized.
    """
    from leaps.models.stride_models import (
        StrideAutoencoder,
        StrideCNMFModel,
        StrideFlatAE,
        StrideMaskedAutoencoder,
        StrideNMFModel,
        StridePCAModel,
        StrideVAE,
        StrideWAE_MMD,
    )

    models = {
        "StridePCA": StridePCAModel,
        "StrideNMF": StrideNMFModel,
        "StrideCNMF": StrideCNMFModel,
        "StrideAE": StrideAutoencoder,
        "StrideVAE": StrideVAE,
        "StrideMAE": StrideMaskedAutoencoder,
        "StrideWAE": StrideWAE_MMD,
        "StrideFlatAE": StrideFlatAE,
    }
    if model_name not in models:
        raise ValueError(f"Unknown stride model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name](
        latent_dim=latent_dim, n_channels=n_channels, stride_len=stride_len, **kwargs
    )
