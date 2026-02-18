"""Training loops and utilities for EMG models."""

from leaps.training.trainer import (
    PYTORCH_MODELS,
    SKLEARN_MODELS,
    build_model,
    train_pytorch_model,
    train_sklearn_model,
)
from leaps.training.stride_trainer import (
    STRIDE_PYTORCH_MODELS,
    STRIDE_SKLEARN_MODELS,
    build_stride_model,
    train_stride_pytorch_model,
    train_stride_sklearn_model,
)

__all__ = [
    # Snapshot-level
    "build_model",
    "train_pytorch_model",
    "train_sklearn_model",
    "SKLEARN_MODELS",
    "PYTORCH_MODELS",
    # Stride-level
    "build_stride_model",
    "train_stride_pytorch_model",
    "train_stride_sklearn_model",
    "STRIDE_SKLEARN_MODELS",
    "STRIDE_PYTORCH_MODELS",
]
