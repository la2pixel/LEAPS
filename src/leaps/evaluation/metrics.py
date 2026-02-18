"""Evaluation metrics for EMG dimensionality reduction models.

Two levels of evaluation:

1. Training metrics (logged every epoch):
   - mse, r2: reconstruction quality

2. Final evaluation metrics (computed once after training):
   - mse, r2: global reconstruction quality
   - per_muscle_r2: R² for each of the 11 muscles — are any muscles ignored?
   - per_mode_r2: R² for each locomotion mode — does it generalize?
   - stride_mse_stats: mean/std/p95/max of per-stride MSE — consistency
   - peak_timing_error: are gait events at the right phase?
   - peak_amplitude_ratio: are muscle bursts preserved or blunted?
   - correlation_matrix_distance: are synergy patterns preserved?
   - gait_profile_correlation: is the mean gait cycle shape correct?
   - latent_smoothness: are latent trajectories smooth? (for RL control)
"""

import numpy as np


def mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Mean squared error between original and reconstructed signals."""
    return float(np.mean((x - x_hat) ** 2))


def r_squared(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot. Measures how much variance in the data
    is explained by the reconstruction. 1.0 = perfect, 0.0 = predicting
    the mean, negative = worse than the mean.
    """
    ss_res = np.sum((x - x_hat) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def per_muscle_r2(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """R² per muscle channel — which muscles are well-reconstructed?

    Args:
        x: Original data, shape (N, n_channels).
        x_hat: Reconstructed data, shape (N, n_channels).

    Returns:
        Array of shape (n_channels,) with R² per muscle.
    """
    r2_vals = np.empty(x.shape[1])
    for i in range(x.shape[1]):
        ss_res = np.sum((x[:, i] - x_hat[:, i]) ** 2)
        ss_tot = np.sum((x[:, i] - np.mean(x[:, i])) ** 2)
        if ss_tot == 0:
            r2_vals[i] = 1.0 if ss_res == 0 else 0.0
        else:
            r2_vals[i] = 1.0 - ss_res / ss_tot
    return r2_vals


def stride_mse_stats(
    x: np.ndarray, x_hat: np.ndarray, stride_len: int = 101,
) -> dict[str, float]:
    """Per-stride MSE distribution stats.

    Reshapes the flat (N, 11) arrays back into strides of stride_len
    time points, computes MSE per stride, then returns distribution stats.

    Args:
        x: Original data, shape (N, n_channels) where N is divisible by stride_len.
        x_hat: Reconstructed data, same shape.
        stride_len: Number of time points per stride (default 101).

    Returns:
        Dict with mean, std, p95, max of per-stride MSE.
    """
    n = x.shape[0]
    n_strides = n // stride_len
    if n_strides == 0:
        return {"stride_mse_mean": mse(x, x_hat), "stride_mse_std": 0.0,
                "stride_mse_p95": mse(x, x_hat), "stride_mse_max": mse(x, x_hat)}

    # Reshape to (n_strides, stride_len, n_channels), compute MSE per stride
    x_s = x[: n_strides * stride_len].reshape(n_strides, stride_len, -1)
    x_hat_s = x_hat[: n_strides * stride_len].reshape(n_strides, stride_len, -1)
    stride_mses = np.mean((x_s - x_hat_s) ** 2, axis=(1, 2))  # (n_strides,)

    return {
        "stride_mse_mean": float(stride_mses.mean()),
        "stride_mse_std": float(stride_mses.std()),
        "stride_mse_p95": float(np.percentile(stride_mses, 95)),
        "stride_mse_max": float(stride_mses.max()),
    }


def per_mode_r2(
    x: np.ndarray,
    x_hat: np.ndarray,
    modes: np.ndarray,
    stride_len: int = 101,
) -> dict[str, float]:
    """R² per locomotion mode — does the model generalize across activities?

    Args:
        x: Original data, shape (N, n_channels).
        x_hat: Reconstructed data, same shape.
        modes: Mode label per stride, shape (n_strides,). e.g. ["treadmill", "ramp", ...]
        stride_len: Number of time points per stride.

    Returns:
        Dict mapping mode name to R² value, e.g. {"treadmill": 0.92, "ramp": 0.87, ...}
    """
    n_strides = len(modes)
    n = n_strides * stride_len
    # Trim to exact stride boundary
    x_trimmed = x[:n]
    x_hat_trimmed = x_hat[:n]

    result = {}
    for mode in np.unique(modes):
        stride_mask = modes == mode
        # Expand stride mask to snapshot mask (each stride = stride_len rows)
        row_mask = np.repeat(stride_mask, stride_len)
        x_mode = x_trimmed[row_mask]
        x_hat_mode = x_hat_trimmed[row_mask]
        result[mode] = r_squared(x_mode, x_hat_mode)
    return result


# -- Gait-aware metrics --------------------------------------------------------


def peak_timing_error(
    x: np.ndarray, x_hat: np.ndarray, stride_len: int = 101,
) -> dict[str, float]:
    """Peak timing error per muscle — do reconstructed EMG peaks land at the
    right gait cycle phase?

    For each muscle, finds the peak index within each stride for both original
    and reconstructed signals, then computes the mean absolute timing difference
    in % gait cycle.

    Args:
        x: Original data, shape (N, n_channels).
        x_hat: Reconstructed data, same shape.
        stride_len: Number of time points per stride (default 101).

    Returns:
        Dict with per-muscle timing error and aggregate stats, e.g.
        {"peak_timing_err_0": 2.3, ..., "peak_timing_err_mean": 3.1}
    """
    n = x.shape[0]
    n_channels = x.shape[1]
    n_strides = n // stride_len
    if n_strides == 0:
        return {}

    x_s = x[: n_strides * stride_len].reshape(n_strides, stride_len, n_channels)
    x_hat_s = x_hat[: n_strides * stride_len].reshape(n_strides, stride_len, n_channels)

    result = {}
    all_errors = []
    for ch in range(n_channels):
        orig_peaks = np.argmax(x_s[:, :, ch], axis=1)  # (n_strides,)
        recon_peaks = np.argmax(x_hat_s[:, :, ch], axis=1)
        # Timing error in % gait cycle (each index = 1%)
        errors = np.abs(orig_peaks.astype(float) - recon_peaks.astype(float))
        mean_err = float(errors.mean())
        result[f"peak_timing_err_{ch}"] = mean_err
        all_errors.append(mean_err)

    result["peak_timing_err_mean"] = float(np.mean(all_errors))
    result["peak_timing_err_max"] = float(np.max(all_errors))
    return result


def peak_amplitude_ratio(
    x: np.ndarray, x_hat: np.ndarray, stride_len: int = 101,
) -> dict[str, float]:
    """Peak amplitude preservation — are muscle bursts preserved or blunted?

    For each muscle, computes the ratio of mean reconstructed peak amplitude
    to mean original peak amplitude across all strides.
    Ratio near 1.0 = good, < 1.0 = blunted, > 1.0 = amplified.

    Args:
        x: Original data, shape (N, n_channels).
        x_hat: Reconstructed data, same shape.
        stride_len: Number of time points per stride (default 101).

    Returns:
        Dict with per-muscle amplitude ratio and aggregate stats.
    """
    n = x.shape[0]
    n_channels = x.shape[1]
    n_strides = n // stride_len
    if n_strides == 0:
        return {}

    x_s = x[: n_strides * stride_len].reshape(n_strides, stride_len, n_channels)
    x_hat_s = x_hat[: n_strides * stride_len].reshape(n_strides, stride_len, n_channels)

    result = {}
    ratios = []
    for ch in range(n_channels):
        orig_peaks = np.max(x_s[:, :, ch], axis=1)  # (n_strides,)
        recon_peaks = np.max(x_hat_s[:, :, ch], axis=1)
        mean_orig = float(orig_peaks.mean())
        mean_recon = float(recon_peaks.mean())
        ratio = mean_recon / mean_orig if mean_orig > 1e-10 else 1.0
        result[f"peak_amp_ratio_{ch}"] = ratio
        ratios.append(ratio)

    result["peak_amp_ratio_mean"] = float(np.mean(ratios))
    result["peak_amp_ratio_min"] = float(np.min(ratios))
    return result


def correlation_matrix_distance(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Inter-muscle correlation preservation.

    Computes the Frobenius norm between the correlation matrices of the
    original and reconstructed signals. Measures whether synergy patterns
    (coordinated muscle firing) survive the encoding.

    Lower = better. 0.0 = perfect preservation.

    Args:
        x: Original data, shape (N, n_channels).
        x_hat: Reconstructed data, same shape.

    Returns:
        Frobenius distance between correlation matrices.
    """
    # Handle constant columns (std=0) by adding tiny noise
    corr_x = np.corrcoef(x.T)
    corr_x_hat = np.corrcoef(x_hat.T)
    # Replace NaN with 0 (happens if a channel is constant)
    corr_x = np.nan_to_num(corr_x, nan=0.0)
    corr_x_hat = np.nan_to_num(corr_x_hat, nan=0.0)
    return float(np.linalg.norm(corr_x - corr_x_hat, ord="fro"))


def gait_profile_correlation(
    x: np.ndarray, x_hat: np.ndarray, stride_len: int = 101,
) -> dict[str, float]:
    """Gait cycle profile correlation per muscle.

    Averages the gait cycle shape (101 timepoints) across all strides for
    each muscle, then correlates the original vs reconstructed average profiles.
    Tests whether the characteristic EMG envelope shape is preserved.

    Args:
        x: Original data, shape (N, n_channels).
        x_hat: Reconstructed data, same shape.
        stride_len: Number of time points per stride (default 101).

    Returns:
        Dict with per-muscle profile correlation and aggregate stats.
    """
    n = x.shape[0]
    n_channels = x.shape[1]
    n_strides = n // stride_len
    if n_strides == 0:
        return {}

    x_s = x[: n_strides * stride_len].reshape(n_strides, stride_len, n_channels)
    x_hat_s = x_hat[: n_strides * stride_len].reshape(n_strides, stride_len, n_channels)

    # Average gait cycle profile per muscle
    mean_orig = x_s.mean(axis=0)  # (stride_len, n_channels)
    mean_recon = x_hat_s.mean(axis=0)

    result = {}
    corrs = []
    for ch in range(n_channels):
        orig_profile = mean_orig[:, ch]
        recon_profile = mean_recon[:, ch]
        # Pearson correlation of the two profiles
        if np.std(orig_profile) < 1e-10 or np.std(recon_profile) < 1e-10:
            corr = 1.0 if np.allclose(orig_profile, recon_profile) else 0.0
        else:
            corr = float(np.corrcoef(orig_profile, recon_profile)[0, 1])
        result[f"gait_profile_corr_{ch}"] = corr
        corrs.append(corr)

    result["gait_profile_corr_mean"] = float(np.mean(corrs))
    result["gait_profile_corr_min"] = float(np.min(corrs))
    return result


def latent_smoothness(latent: np.ndarray, stride_len: int = 101) -> float:
    """Latent trajectory smoothness — mean L2 distance between consecutive
    latent vectors within a stride.

    For RL control, smooth latent trajectories = smooth actions. A good
    representation should map adjacent gait cycle phases to nearby points
    in latent space.

    Args:
        latent: Latent representations, shape (N, latent_dim) where
            N = n_strides * stride_len.
        stride_len: Number of time points per stride.

    Returns:
        Mean L2 distance between consecutive latent vectors.
    """
    n_strides = latent.shape[0] // stride_len
    if n_strides == 0:
        return 0.0

    latent_s = latent[: n_strides * stride_len].reshape(n_strides, stride_len, -1)
    diffs = np.linalg.norm(latent_s[:, 1:, :] - latent_s[:, :-1, :], axis=2)
    return float(diffs.mean())


def compute_eval_metrics(
    x: np.ndarray,
    x_hat: np.ndarray,
    latent: np.ndarray | None = None,
    modes: np.ndarray | None = None,
    stride_len: int = 101,
) -> dict[str, float]:
    """Compute all evaluation metrics for a trained model.

    Aggregates basic reconstruction metrics with gait-aware metrics
    into a single dict for logging.

    Args:
        x: Original data, shape (N, n_channels).
        x_hat: Reconstructed data, same shape.
        latent: Latent representations, shape (N, latent_dim). Optional.
        modes: Mode label per stride, shape (n_strides,). Optional.
        stride_len: Number of time points per stride.

    Returns:
        Flat dict with all metric values.
    """
    metrics: dict[str, float] = {}

    # Basic reconstruction
    metrics["mse"] = mse(x, x_hat)
    metrics["r2"] = r_squared(x, x_hat)

    # Per-muscle R²
    muscle_r2 = per_muscle_r2(x, x_hat)
    metrics["per_muscle_r2_mean"] = float(muscle_r2.mean())
    metrics["per_muscle_r2_min"] = float(muscle_r2.min())

    # Stride consistency
    metrics.update(stride_mse_stats(x, x_hat, stride_len))

    # Peak timing
    metrics.update(peak_timing_error(x, x_hat, stride_len))

    # Peak amplitude
    metrics.update(peak_amplitude_ratio(x, x_hat, stride_len))

    # Synergy preservation
    metrics["corr_matrix_dist"] = correlation_matrix_distance(x, x_hat)

    # Gait profile shape
    metrics.update(gait_profile_correlation(x, x_hat, stride_len))

    # Latent smoothness
    if latent is not None:
        metrics["latent_smoothness"] = latent_smoothness(latent, stride_len)

    # Per-mode R²
    if modes is not None:
        mode_r2 = per_mode_r2(x, x_hat, modes, stride_len)
        for mode, r2_val in mode_r2.items():
            metrics[f"mode_{mode}_r2"] = r2_val

    return metrics


# -- Convenience functions used during training --------------------------------

def compute_train_metrics(x: np.ndarray, x_hat: np.ndarray) -> dict[str, float]:
    """Quick metrics for training loop: just MSE and R²."""
    return {
        "mse": mse(x, x_hat),
        "r2": r_squared(x, x_hat),
    }
