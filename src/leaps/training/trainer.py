"""Training utilities for EMG dimensionality reduction models.

This module provides:
  - build_model():          Factory to create any model by name string
  - train_sklearn_model():  One-shot fit for PCA and NMF
  - train_pytorch_model():  Epoch-based training for AE, VAE, MAE

The train functions accept an optional `log_fn` callback that is called
with (metrics_dict, step) each epoch. This is how wandb logging works:
the train.py script creates a log_fn that calls wandb.log(), and passes
it here. The training code itself has no wandb dependency.

## Architecture

    train.py (CLI)
        │
        ├── build_model("AE", latent_dim=4)  →  Autoencoder instance
        │
        ├── train_sklearn_model(model, x_train, x_val)
        │       → model.fit(x_train, x_val)
        │       → saves checkpoint to {save_dir}/{ModelName}_d{dim}.pkl
        │       → returns metrics dict
        │
        └── train_pytorch_model(model, x_train, x_val, log_fn=...)
                → model.fit(x_train, x_val, epochs=..., log_fn=log_fn)
                → the model's fit() calls log_fn(metrics, epoch) each epoch
                → saves checkpoint to {save_dir}/{ModelName}_d{dim}.pt
                → returns metrics dict

## Available models

    Name │ Class            │ Type    │ What it does
    ─────┼──────────────────┼─────────┼────────────────────────────────────
    PCA  │ PCAModel         │ sklearn │ Linear projection, instant fit
    NMF  │ NMFModel         │ sklearn │ Non-negative decomposition, ~1s fit
    AE   │ Autoencoder      │ PyTorch │ Standard autoencoder (11→64→32→d→32→64→11)
    VAE  │ VAE              │ PyTorch │ Variational AE (adds KL regularization)
    MAE  │ MaskedAutoencoder│ PyTorch │ Randomly masks input channels during training
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from leaps.evaluation.metrics import compute_train_metrics
from leaps.models.base import EMGModel

# Generic logging callback type. Called as log_fn(metrics_dict, step).
# Set to None to disable logging. Used by train.py to wire up wandb.
LogFn = Callable[[dict[str, float], int], None] | None


def train_sklearn_model(
    model: EMGModel,
    x_train: np.ndarray,
    x_val: np.ndarray | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, float]:
    """Train a sklearn-based model (PCA or NMF).

    These models fit in a single call — no epochs, no GPU needed.
    PCA uses SVD decomposition; NMF uses coordinate descent.

    Args:
        model:    PCAModel or NMFModel instance.
        x_train:  Training data, shape (N, 11).
        x_val:    Validation data for computing val metrics.
        save_dir: Directory to save checkpoint (.pkl file).

    Returns:
        Metrics dict with keys: mse, r2, vaf, val_mse, val_r2, val_vaf,
        and model-specific keys (e.g. explained_variance_ratio for PCA).
    """
    metrics = model.fit(x_train, x_val)
    if save_dir is not None:
        save_path = Path(save_dir) / f"{model.name}_d{model.latent_dim}.pkl"
        model.save(save_path)
    return metrics


def train_pytorch_model(
    model: EMGModel,
    x_train: np.ndarray,
    x_val: np.ndarray | None = None,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    save_dir: str | Path | None = None,
    log_fn: LogFn = None,
) -> dict[str, float]:
    """Train a PyTorch-based model (AE, VAE, or MAE).

    Runs the model's fit() method which does the full training loop:
    Adam optimizer, MSE loss, optional KL term (VAE), optional masking (MAE).

    The log_fn callback is called every epoch with metrics and the epoch
    number. This is how wandb gets its data — the callback simply calls
    wandb.log(). If log_fn is None, training proceeds silently.

    Metrics logged each epoch (via log_fn):
      - train_loss:       training MSE (every epoch)
      - val_mse/r2/vaf:   validation metrics (every 10 epochs + last)
      - train_recon_loss:  VAE reconstruction loss (every epoch)
      - train_kl_loss:     VAE KL divergence (every epoch)

    Note: Set random seeds globally before calling this function
    (torch.manual_seed, np.random.seed, etc.) for reproducibility.

    Args:
        model:      Autoencoder, VAE, or MaskedAutoencoder instance.
        x_train:    Training data, shape (N, 11).
        x_val:      Validation data for computing val metrics.
        epochs:     Number of training epochs (default: 200).
        batch_size: Mini-batch size (default: 256). Use 512+ with GPU.
        lr:         Learning rate for Adam (default: 1e-3).
        save_dir:   Directory to save checkpoint (.pt file).
        log_fn:     Callback for logging: log_fn(metrics_dict, epoch).
                    Pass None to disable. train.py wires this to wandb.

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


def build_model(
    model_name: str, latent_dim: int, n_channels: int = 11, **kwargs
) -> EMGModel:
    """Factory function to create a model by name.

    Args:
        model_name: One of "PCA", "NMF", "AE", "VAE", "MAE".
        latent_dim: Dimensionality of the latent space (e.g. 4).
        n_channels: Number of input features (default: 11 EMG channels).
        **kwargs:   Model-specific args (beta for VAE, mask_ratio for MAE).

    Returns:
        An EMGModel instance ready for training.

    Raises:
        ValueError: If model_name is not recognized.
    """
    from leaps.models.pytorch_models import Autoencoder, MaskedAutoencoder, VAE, WAE_MMD
    from leaps.models.sklearn_models import NMFModel, PCAModel

    models = {
        "PCA": PCAModel,
        "NMF": NMFModel,
        "AE": Autoencoder,
        "VAE": VAE,
        "MAE": MaskedAutoencoder,
        "WAE": WAE_MMD,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name](latent_dim=latent_dim, n_channels=n_channels, **kwargs)


# Sets used by train.py to determine which training function to use
SKLEARN_MODELS = {"PCA", "NMF"}
PYTORCH_MODELS = {"AE", "VAE", "MAE", "WAE"}
