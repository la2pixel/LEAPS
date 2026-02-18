"""Abstract base class for EMG dimensionality reduction models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EMGModel(ABC):
    """Base interface for all EMG dimensionality reduction models.

    All models compress 11-channel EMG activations to a latent_dim-dimensional
    representation and can reconstruct the original signal.
    """

    def __init__(self, latent_dim: int, n_channels: int = 11):
        self.latent_dim = latent_dim
        self.n_channels = n_channels

    @abstractmethod
    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        """Train the model.

        Args:
            x_train: Training data, shape (N, n_channels).
            x_val: Validation data, shape (M, n_channels). Optional.

        Returns:
            Dictionary of final metrics.
        """

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Encode data to latent space.

        Args:
            x: Input data, shape (N, n_channels).

        Returns:
            Latent representation, shape (N, latent_dim).
        """

    @abstractmethod
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Encode then decode data.

        Args:
            x: Input data, shape (N, n_channels).

        Returns:
            Reconstructed data, shape (N, n_channels).
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model from disk."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
