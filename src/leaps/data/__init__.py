"""Data loading and preprocessing for the Camargo EMG dataset."""

from leaps.data.loaders import build_emg_h5, discover_trials, load_camargo_dataset, load_conditions_table, load_mat_table
from leaps.data.processing import classify_stride_label
from leaps.data.metadata import SUBJECTS, SubjectInfo
from leaps.data.stride_dataset import StrideDataset, load_strides, stride_train_val_split

__all__ = [
    # Pipeline
    "build_emg_h5",
    # Raw data access
    "discover_trials",
    "load_mat_table",
    "load_conditions_table",
    "load_camargo_dataset",
    # Stride classification
    "classify_stride_label",
    # Metadata
    "SUBJECTS",
    "SubjectInfo",
    # Stride-level dataset
    "StrideDataset",
    "load_strides",
    "stride_train_val_split",
]
