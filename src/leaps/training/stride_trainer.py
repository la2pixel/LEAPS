"""Stride model factory and name sets for train_strides.py."""

from leaps.models.old_code import ActionModel

STRIDE_SKLEARN_MODELS = {"StridePCA", "StrideNMF"}
STRIDE_PYTORCH_MODELS = {"StrideFlatAE", "StrideFlatVAE"}


def build_stride_model(
    model_name: str, latent_dim: int, n_channels: int = 11, stride_len: int = 101, **kwargs
) -> ActionModel:
    """Factory function to create a stride-level model by name.

    Args:
        model_name: One of "StridePCA", "StrideNMF", "StrideFlatAE", "StrideFlatVAE".
        latent_dim: Dimensionality of the latent space.
        n_channels: Number of EMG channels (default: 11).
        stride_len: Timepoints per stride (default: 101).
        **kwargs:   Model-specific args (beta, free_bits for StrideFlatVAE).

    Returns:
        An ActionModel instance ready for training.

    Raises:
        ValueError: If model_name is not recognized.
    """
    from leaps.models.old_code import (
        StrideFlatAE,
        StrideFlatVAE,
        StrideNMFModel,
        StridePCAModel,
    )

    models = {
        "StridePCA": StridePCAModel,
        "StrideNMF": StrideNMFModel,
        "StrideFlatAE": StrideFlatAE,
        "StrideFlatVAE": StrideFlatVAE,
    }
    if model_name not in models:
        raise ValueError(f"Unknown stride model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name](
        latent_dim=latent_dim, n_channels=n_channels, stride_len=stride_len, **kwargs
    )
