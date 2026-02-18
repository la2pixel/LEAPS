"""Evaluation metrics for EMG dimensionality reduction."""

from leaps.evaluation.metrics import (
    compute_eval_metrics,
    compute_train_metrics,
    correlation_matrix_distance,
    gait_profile_correlation,
    latent_smoothness,
    mse,
    peak_amplitude_ratio,
    peak_timing_error,
    per_mode_r2,
    per_muscle_r2,
    r_squared,
    stride_mse_stats,
)

__all__ = [
    "mse",
    "r_squared",
    "per_muscle_r2",
    "per_mode_r2",
    "stride_mse_stats",
    "peak_timing_error",
    "peak_amplitude_ratio",
    "correlation_matrix_distance",
    "gait_profile_correlation",
    "latent_smoothness",
    "compute_train_metrics",
    "compute_eval_metrics",
]
