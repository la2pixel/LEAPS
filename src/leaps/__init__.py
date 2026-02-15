"""LEAPS: Learning Humanoid Locomotion from EMG-Based Latent Action Priors and Muscle Synergies."""

__version__ = "0.1.0"
__author__ = "Lalitha Sivakumar"

from leaps.data import (
    discover_trials,
    load_camargo_dataset,
    load_mat_table,
    parse_camargo,
    process_subject,
    rectify_emg,
)

__all__ = [
    "discover_trials",
    "load_camargo_dataset",
    "load_mat_table",
    "parse_camargo",
    "process_subject",
    "rectify_emg",
]
