"""Training loops and utilities for EMG models."""

from leaps.training.stride_trainer import (
    STRIDE_PYTORCH_MODELS,
    STRIDE_SKLEARN_MODELS,
    build_stride_model,
    train_stride_pytorch_model,
    train_stride_sklearn_model,
)

__all__ = [
    "build_stride_model",
    "train_stride_pytorch_model",
    "train_stride_sklearn_model",
    "STRIDE_SKLEARN_MODELS",
    "STRIDE_PYTORCH_MODELS",
]
