"""EMG dataset utilities for loading HDF5 activations and creating train/val splits.

Normalization strategy:
    The raw preprocessed data uses reference-speed (1.35 m/s) min-max normalization,
    which produces values outside [0, 1] for faster speeds and non-treadmill modes.
    For autoencoder training, we clip to the per-channel 99th percentile and rescale
    to [0, 1]. This discards ~1% extreme outliers while keeping the data compatible
    with [0, 1] muscle actuator commands.

    The clip/rescale parameters are computed from the training data and can be saved
    alongside the model for use at inference time.
"""

import logging
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class NormParams:
    """Per-channel clip-and-rescale parameters.

    Computed from training data, applied to all data (train, val, inference).
    Save alongside model checkpoints so the same transform can be applied at
    RL deployment time.
    """

    clip_min: np.ndarray  # (n_channels,) — lower clip bound (usually ~0)
    clip_max: np.ndarray  # (n_channels,) — upper clip bound (99th percentile)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Clip and rescale to [0, 1]."""
        clipped = np.clip(x, self.clip_min, self.clip_max)
        denom = self.clip_max - self.clip_min
        denom[denom < 1e-8] = 1.0
        return (clipped - self.clip_min) / denom

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Undo the rescaling (for reconstruction evaluation)."""
        denom = self.clip_max - self.clip_min
        denom[denom < 1e-8] = 1.0
        return x * denom + self.clip_min

    def save(self, path: str) -> None:
        np.savez(path, clip_min=self.clip_min, clip_max=self.clip_max)

    @classmethod
    def load(cls, path: str) -> "NormParams":
        data = np.load(path)
        return cls(clip_min=data["clip_min"], clip_max=data["clip_max"])


def compute_norm_params(
    x: np.ndarray, lower_pct: float = 0.0, upper_pct: float = 99.0,
) -> NormParams:
    """Compute per-channel clip bounds from data.

    Args:
        x: Training data, shape (N, n_channels).
        lower_pct: Lower percentile for clipping (default 0 = keep all low values).
        upper_pct: Upper percentile for clipping (default 99).

    Returns:
        NormParams with per-channel clip bounds.
    """
    clip_min = np.percentile(x, lower_pct, axis=0)
    clip_max = np.percentile(x, upper_pct, axis=0)
    # Floor clip_min at 0 — EMG activations shouldn't be negative
    clip_min = np.maximum(clip_min, 0.0)
    logger.info(
        "Norm params: clip_min=[%.3f, %.3f], clip_max=[%.3f, %.3f]",
        clip_min.min(), clip_min.max(), clip_max.min(), clip_max.max(),
    )
    return NormParams(clip_min=clip_min, clip_max=clip_max)


class EMGDataset(Dataset):
    """PyTorch Dataset wrapping (N, n_channels) EMG activation data."""

    def __init__(self, activations: np.ndarray):
        self.data = torch.as_tensor(activations, dtype=torch.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def load_activations(
    path: str, with_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load EMG activations from an HDF5 file.

    Args:
        path: Path to the HDF5 file.
        with_metadata: If True, also return stride-level metadata
            (mode labels, subject labels).

    Returns:
        If with_metadata is False: activations array, shape (N, n_channels).
        If with_metadata is True: (activations, metadata) where metadata is
            {"modes": (M,) str array, "subjects": (M,) str array}.
            M = N / 101 (one label per stride, each stride = 101 snapshots).
    """
    with h5py.File(path, "r") as f:
        activations = np.array(f["activations"])
        if not with_metadata:
            return activations
        metadata = {}
        if "stride_modes" in f:
            metadata["modes"] = np.array(f["stride_modes"]).astype(str)
        if "stride_subjects" in f:
            metadata["subjects"] = np.array(f["stride_subjects"]).astype(str)
        return activations, metadata


def train_val_split(
    x: np.ndarray, val_fraction: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle and split data into train and validation sets.

    Args:
        x: Data array, shape (N, ...).
        val_fraction: Fraction of data to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (x_train, x_val).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    n_val = max(1, int(len(x) * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return x[train_idx], x[val_idx]
