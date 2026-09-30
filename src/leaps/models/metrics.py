"""Reconstruction metrics for EMG dimensionality-reduction models.

minimal replacement for the leaps.evaluation package (dropped in the
2026-08-27 SCONE refactor) -- just the 2 functions legacy_models.py and
plot_reconstructions.py actually still use, not the full original module.
"""

import numpy as np


def mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Mean squared error between original and reconstructed signals."""
    return float(np.mean((x - x_hat) ** 2))


def r_squared(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot. 1.0 = perfect, 0.0 = predicting the mean,
    negative = worse than the mean.
    """
    ss_res = np.sum((x - x_hat) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def per_muscle_r2(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """R² per muscle channel -- which muscles are well-reconstructed?

    x, x_hat: (N, n_channels). Returns (n_channels,).
    """
    r2_vals = np.empty(x.shape[1])
    for i in range(x.shape[1]):
        ss_res = np.sum((x[:, i] - x_hat[:, i]) ** 2)
        ss_tot = np.sum((x[:, i] - np.mean(x[:, i])) ** 2)
        r2_vals[i] = (1.0 if ss_res == 0 else 0.0) if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return r2_vals


def compute_train_metrics(x: np.ndarray, x_hat: np.ndarray) -> dict[str, float]:
    """Quick metrics for the training loop: just MSE and R²."""
    return {"mse": mse(x, x_hat), "r2": r_squared(x, x_hat)}
