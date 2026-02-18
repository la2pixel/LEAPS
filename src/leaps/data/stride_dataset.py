"""Stride-level dataset utilities for loading (n_strides, 101, 11) sequences.

The preprocessed HDF5 file stores stride data in two ways:
  - /{subject}/strides  — (n_strides, 101, 11) per subject (3D, temporal)
  - /activations        — (N, 11) all subjects flattened (2D, snapshot)

This module loads the 3D form for stride-level training.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class StrideDataset(Dataset):
    """PyTorch Dataset wrapping (n_strides, stride_len, n_channels) stride data."""

    def __init__(self, strides: np.ndarray):
        self.data = torch.as_tensor(strides, dtype=torch.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]  # (stride_len, n_channels)


def load_strides(
    path: str,
    subjects: list[str] | None = None,
    with_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load stride sequences from HDF5.

    Reads per-subject /{subject}/strides datasets and concatenates them.

    Args:
        path: Path to HDF5 file (output of leaps-preprocess).
        subjects: Subject IDs to load (default: all AB* groups in file).
        with_metadata: If True, also return per-stride mode and subject labels.

    Returns:
        If with_metadata is False:
            strides array, shape (n_strides, 101, 11).
        If with_metadata is True:
            (strides, metadata) where metadata = {
                "modes": (n_strides,) str array,
                "subjects": (n_strides,) str array,
            }
    """
    with h5py.File(path, "r") as f:
        if subjects is None:
            subjects = sorted(k for k in f.keys() if k.startswith("AB"))

        all_strides = []
        all_modes = []
        all_subjects = []

        for subj in subjects:
            if subj not in f:
                continue
            grp = f[subj]
            strides = np.array(grp["strides"])  # (n_strides, 101, 11)
            all_strides.append(strides)

            if with_metadata:
                modes = np.array(grp["modes"]).astype(str)
                all_modes.extend(modes)
                all_subjects.extend([subj] * len(strides))

        strides_concat = np.concatenate(all_strides, axis=0)

        if not with_metadata:
            return strides_concat

        metadata = {
            "modes": np.array(all_modes),
            "subjects": np.array(all_subjects),
        }
        return strides_concat, metadata


def stride_train_val_split(
    strides: np.ndarray,
    metadata: dict[str, np.ndarray] | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Split stride data into train/val at the subject level.

    Holds out entire subjects for validation to prevent data leakage.
    If metadata is not provided, falls back to random stride-level split.

    Args:
        strides: (n_strides, 101, 11) array.
        metadata: Dict with "subjects" and optionally "modes" arrays.
        val_fraction: Fraction of subjects held out for validation.
        seed: Random seed.

    Returns:
        If metadata is None:
            (strides_train, strides_val)
        If metadata is provided:
            (strides_train, strides_val, meta_train, meta_val)
    """
    rng = np.random.default_rng(seed)

    if metadata is not None and "subjects" in metadata:
        subjects = metadata["subjects"]
        unique_subjects = np.unique(subjects)
        rng.shuffle(unique_subjects)

        n_val_subjects = max(1, int(len(unique_subjects) * val_fraction))
        val_subjects = set(unique_subjects[:n_val_subjects])

        val_mask = np.array([s in val_subjects for s in subjects])
        train_mask = ~val_mask

        if metadata is not None:
            meta_train = {k: v[train_mask] for k, v in metadata.items()}
            meta_val = {k: v[val_mask] for k, v in metadata.items()}
            return strides[train_mask], strides[val_mask], meta_train, meta_val

        return strides[train_mask], strides[val_mask]

    # Fallback: random stride-level split
    indices = rng.permutation(len(strides))
    n_val = max(1, int(len(strides) * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    if metadata is not None:
        meta_train = {k: v[train_idx] for k, v in metadata.items()}
        meta_val = {k: v[val_idx] for k, v in metadata.items()}
        return strides[train_idx], strides[val_idx], meta_train, meta_val

    return strides[train_idx], strides[val_idx]
