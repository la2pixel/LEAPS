"""StrideDataset utilities for loading (n_strides, 101, 11) sequences."""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class StrideDataset(Dataset):
    """Dataset wrapping (n_strides, stride_len, n_channels)"""

    def __init__(self, strides: np.ndarray):
        self.data = torch.as_tensor(strides, dtype=torch.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]  #(stride_len, n_channels)


def load_strides(
    path: str,
    subjects: list[str] | None = None,
    modes: list[str] | None = None,
    conditions: list[str] | None = None,
    with_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load stride sequences from HDF5 with option to filter by mode.

    Args:
        path:       Path to HDF5 file (output of leaps preprocess).
        subjects:   Subject IDs to load (default: all AB* groups in file).
        modes:      Filter to specific modes, e.g. ["treadmill", "levelground"].
                    Default: all modes.
        conditions: Filter by condition label(s).
                    Examples:
                      ["1.20"]             — treadmill at 1.2 m/s
                      ["slow", "normal"]   — levelground slow + normal
                      ["5.2_up"]           — ramp 5.2° ascending only
                      ["102", "127"]       — stair heights 102 mm and 127 mm
                    Requires /{subject}/conditions dataset (regenerated HDF5).
                    Default: all conditions.
        with_metadata: If True, also return per-stride metadata arrays.

    Returns:
        If with_metadata is False:
            strides, shape (n_strides, 101, 11).
        If with_metadata is True:
            (strides, metadata) where metadata is a dict with keys:
                "modes"      — (n_strides,) str
                "subjects"   — (n_strides,) str
                "speeds"     — (n_strides,) float64, NaN for non-treadmill
                "conditions" — (n_strides,) str  (if available in HDF5)
    """
    all_strides: list[np.ndarray] = []
    all_meta_modes: list[str] = []
    all_meta_subjects: list[str] = []
    all_meta_speeds: list[float] = []
    all_meta_conditions: list[str] = []

    with h5py.File(path, "r") as f:
        if subjects is None:
            subjects = sorted(k for k in f.keys() if k.startswith("AB"))

        for subj in subjects:
            if subj not in f:
                continue
            grp = f[subj]
            if "strides" not in grp:
                continue

            strides_arr = np.array(grp["strides"])                  # (n, 101, 11)
            n = len(strides_arr)
            mode_arr = np.array(grp["modes"]).astype(str) if "modes" in grp else np.full(n, "unknown")
            speed_arr = np.array(grp["speeds"]) if "speeds" in grp else np.full(n, float("nan"))
            cond_arr = np.array(grp["conditions"]).astype(str) if "conditions" in grp else np.full(n, "unknown")

            # Build boolean mask for requested filters
            mask = np.ones(n, dtype=bool)
            if modes is not None:
                mask &= np.isin(mode_arr, modes)
            if conditions is not None:
                mask &= np.isin(cond_arr, conditions)

            if not mask.any():
                continue

            all_strides.append(strides_arr[mask])
            if with_metadata:
                all_meta_modes.extend(mode_arr[mask].tolist())
                all_meta_subjects.extend([subj] * int(mask.sum()))
                all_meta_speeds.extend(speed_arr[mask].tolist())
                all_meta_conditions.extend(cond_arr[mask].tolist())

    if not all_strides:
        raise RuntimeError(
            f"No strides found in {path} matching "
            f"modes={modes}, conditions={conditions}"
        )

    strides_concat = np.concatenate(all_strides, axis=0)

    if not with_metadata:
        return strides_concat

    metadata: dict[str, np.ndarray] = {
        "modes": np.array(all_meta_modes),
        "subjects": np.array(all_meta_subjects),
        "speeds": np.array(all_meta_speeds, dtype=np.float64),
        "conditions": np.array(all_meta_conditions),
    }
    return strides_concat, metadata


def stride_train_val_split(
    strides: np.ndarray,
    metadata: dict[str, np.ndarray] | None = None,
    val_fraction: float = 0.2,  #so like 4 subjects
    seed: int = 42,
) -> tuple:
    """Split stride data into train/val at the subject level.

    Args:
        strides: (n_strides, 101, 11) array.
        metadata: Dict with "subjects" and optionally "modes" arrays.
        val_fraction: Fraction of subjects held out for validation.
        seed: Random seed.

    Returns:
        If metadata is None: (strides_train, strides_val)
        If metadata is provided: (strides_train, strides_val, meta_train, meta_val)
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
