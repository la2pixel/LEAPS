"""Camargo EMG dataset: raw parsing, processing, metadata, torch dataset.

raw .mat -> rectified, gait-cycle-segmented, normalized strides -> HDF5.
one file per pipeline stage, each named after (and cited against) the
Camargo .m script it replicates:

    metadata.py        <- channels, filter specs, subject demographics
    raw.py              <- .mat parsing, no .m equivalent (pure I/O)
    filters.py          <- rectify.m
    strides.py           <- segment_gc.m, STRIDES.m (stride labeling)
    normalize.py         <- getEMGNormalization.m
    build_h5.py           <- STRIDES.m (outer loop) + the "leaps preprocess" CLI
    stride_dataset.py     <- torch Dataset over the processed HDF5

quick start:
    python -m leaps.data.build_h5 --data-root /path/to/camargo --output out.h5

build_emg_h5 isn't re-exported here (only from leaps.data.build_h5) --
build_h5.py needs leaps.data.metadata, which would re-enter this file
mid-init if this file also needed something from build_h5.py.
"""

from leaps.data.metadata import SUBJECTS, SubjectInfo
from leaps.data.raw import discover_trials, load_camargo_dataset, load_conditions_table, load_mat_table
from leaps.data.filters import rectify_emg
from leaps.data.strides import N_GAIT_POINTS, classify_stride_label, find_stride_intervals, process_mode_trials, process_trial_emg, segment_strides, time_normalize_stride
from leaps.data.normalize import compute_emg_normalization, compute_normalization, normalize_emg
from leaps.data.stride_dataset import StrideDataset, load_strides, stride_train_val_split

__all__ = [
    # raw data access
    "discover_trials",
    "load_mat_table",
    "load_conditions_table",
    "load_camargo_dataset",
    # filters.py (rectify.m)
    "rectify_emg",
    # strides.py (segment_gc.m / STRIDES.m)
    "find_stride_intervals",
    "segment_strides",
    "time_normalize_stride",
    "classify_stride_label",
    "process_trial_emg",
    "process_mode_trials",
    "N_GAIT_POINTS",
    # normalize.py (getEMGNormalization.m)
    "compute_emg_normalization",
    "normalize_emg",
    "compute_normalization",
    # metadata
    "SUBJECTS",
    "SubjectInfo",
    # stride-level torch dataset
    "StrideDataset",
    "load_strides",
    "stride_train_val_split",
]
