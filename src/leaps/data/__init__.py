"""Data loading and preprocessing for the Camargo EMG dataset."""

from leaps.data.loaders import (
    discover_trials,
    load_camargo_dataset,
    load_mat_table,
    parse_camargo,
)
from leaps.data.metadata import SUBJECTS, SubjectInfo
from leaps.data.processing import (
    compute_emg_normalization,
    compute_normalization,
    find_stride_intervals,
    normalize_emg,
    process_mode_trials,
    process_trial_emg,
    rectify_emg,
    segment_strides,
    time_normalize_stride,
)

from leaps.data.dataset import EMGDataset, load_activations, train_val_split
from leaps.data.stride_dataset import StrideDataset, load_strides, stride_train_val_split

__all__ = [
    # Loaders
    "discover_trials",
    "load_camargo_dataset",
    "load_mat_table",
    "parse_camargo",
    # Metadata
    "SUBJECTS",
    "SubjectInfo",
    # Processing
    "compute_emg_normalization",
    "find_stride_intervals",
    "normalize_emg",
    "compute_normalization",
    "process_mode_trials",
    "process_trial_emg",
    "rectify_emg",
    "segment_strides",
    "time_normalize_stride",
    # Dataset (snapshot-level)
    "EMGDataset",
    "load_activations",
    "train_val_split",
    # Dataset (stride-level)
    "StrideDataset",
    "load_strides",
    "stride_train_val_split",
]
