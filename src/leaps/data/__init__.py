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
    find_stride_intervals,
    normalize_emg,
    process_subject,
    process_trial_emg,
    rectify_emg,
    segment_strides,
    time_normalize_stride,
)

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
    "process_subject",
    "process_trial_emg",
    "rectify_emg",
    "segment_strides",
    "time_normalize_stride",
]
