"""LEAPS: Learning Humanoid Locomotion from EMG-Based Latent Action Priors and Muscle Synergies."""

__version__ = "0.1.0"
__author__ = "Lalitha Sivakumar"

from leaps.data import (
    EMGDataset,
    discover_trials,
    load_activations,
    load_camargo_dataset,
    load_mat_table,
    parse_camargo,
    compute_normalization,
    process_mode_trials,
    rectify_emg,
    train_val_split,
)
from leaps.models import (
    EMGModel,
    NMFModel,
    PCAModel,
)

__all__ = [
    # Data
    "discover_trials",
    "load_camargo_dataset",
    "load_mat_table",
    "parse_camargo",
    "rectify_emg",
    "EMGDataset",
    "load_activations",
    "train_val_split",
    # Models
    "EMGModel",
    "PCAModel",
    "NMFModel",
]
