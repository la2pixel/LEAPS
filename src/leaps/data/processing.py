"""EMG preprocessing pipeline replicating the Camargo STRIDES.m workflow.

Transforms raw EMG from .mat files into autoencoder-ready (N, 11) activation
snapshots. The processing chain per subject is:

    1. Rectify EMG      — highpass → bandpass → abs → lowpass → abs (rectify.m)
    2. Normalize EMG     — min-max using mean stride at 1.35 m/s (getEMGNormalization.m)
    3. Segment strides   — cut at right heel strikes (segment_gc.m)
    4. Time-normalize    — interpolate each stride to 101 points (0–100% gait cycle)
    5. Flatten           — pool all (101, 11) strides into (N*101, 11) for autoencoder
"""

import logging

import numpy as np
from scipy.signal import butter, filtfilt, firwin

logger = logging.getLogger(__name__)

# Number of gait-cycle sample points per stride (0%, 1%, ..., 100%)
N_GAIT_POINTS = 101


# -- Step 1: EMG rectification ------------------------------------------------

def rectify_emg(emg: np.ndarray, fs: int = 1000) -> np.ndarray:
    """Apply the rectify.m filter chain to raw EMG.

    Args:
        emg: Raw EMG array, shape (n_samples, n_channels). Channels only,
             no Header column.
        fs:  Sampling rate in Hz (default 1000 for Camargo EMG).

    Returns:
        Rectified EMG envelope, same shape as input.

    Filter chain (matches rectify.m exactly):
        1. Highpass IIR  — Butterworth order 4, cutoff 10 Hz
        2. Bandpass FIR  — order 20 (21 taps), 10–450 Hz
        3. Full-wave rectification (abs)
        4. Lowpass IIR   — Butterworth order 4, cutoff 6 Hz (envelope)
        5. abs
    """
    x = emg.astype(np.float64)

    # 1. Highpass: remove DC offset and movement artifacts
    b, a = butter(4, 10.0, btype="high", fs=fs)
    x = filtfilt(b, a, x, axis=0)

    # 2. Bandpass: isolate muscle signal band
    b_fir = firwin(21, [10.0, 450.0], pass_zero=False, fs=fs)
    x = filtfilt(b_fir, [1.0], x, axis=0)

    # 3. Full-wave rectification
    x = np.abs(x)

    # 4. Lowpass: smooth into linear envelope
    b, a = butter(4, 6.0, btype="low", fs=fs)
    x = filtfilt(b, a, x, axis=0)

    # 5. Final abs (removes small numerical negatives from filtfilt)
    x = np.abs(x)

    return x


# -- Step 2: Stride segmentation ----------------------------------------------

def find_stride_intervals(gc_header: np.ndarray, heel_strike: np.ndarray) -> list[tuple[float, float]]:
    """Find heel-strike-to-heel-strike time intervals from the gait cycle signal.

    Replicates gc2intervals() in segment_gc.m. The HeelStrike column is a
    sawtooth wave (0–100%) that resets at each heel strike. We detect the
    reset points (troughs) to find stride boundaries.

    Args:
        gc_header:   Time vector from gcRight, shape (n,).
        heel_strike: HeelStrike column from gcRight, shape (n,).

    Returns:
        List of (start_time, end_time) tuples, one per stride.
    """
    # findpeaks(-gc) in MATLAB finds troughs = heel strike moments
    # We look for sharp negative drops in the sawtooth
    gc = heel_strike.copy()

    # Find trough indices: points lower than both neighbors
    trough_idx = []
    for i in range(1, len(gc) - 1):
        if gc[i] < gc[i - 1] and gc[i] <= gc[i + 1]:
            # Only count significant troughs (near zero), not noise
            if gc[i] < 20.0:
                trough_idx.append(i)

    # Also include the first and last points where gc > 0
    # (matching the MATLAB logic in gc2intervals)
    first_pos = np.argmax(gc > 0)
    if first_pos > 0:
        boundary_indices = [first_pos - 1] + trough_idx
    else:
        boundary_indices = trough_idx

    last_pos = len(gc) - 1 - np.argmax(gc[::-1] > 0)
    if last_pos < len(gc) - 1:
        boundary_indices = boundary_indices + [last_pos + 1]

    if len(boundary_indices) < 2:
        return []

    # Convert to time values and pair consecutive boundaries
    times = gc_header[boundary_indices]
    intervals = [(times[i], times[i + 1]) for i in range(len(times) - 1)]
    return intervals


def segment_strides(
    header: np.ndarray,
    data: np.ndarray,
    intervals: list[tuple[float, float]],
) -> list[np.ndarray]:
    """Cut a continuous signal into strides using time intervals.

    Args:
        header: Time vector, shape (n_samples,).
        data:   Signal array, shape (n_samples, n_channels).
        intervals: List of (start_time, end_time) from find_stride_intervals.

    Returns:
        List of arrays, each shape (n_stride_samples, n_channels).
    """
    strides = []
    for t_start, t_end in intervals:
        mask = (header >= t_start) & (header < t_end)
        segment = data[mask]
        if len(segment) > 10:  # skip degenerate strides
            strides.append(segment)
    return strides


# -- Step 3: Time normalization ------------------------------------------------

def time_normalize_stride(stride: np.ndarray, n_points: int = N_GAIT_POINTS) -> np.ndarray:
    """Interpolate a single stride to a fixed number of evenly-spaced points.

    Matches Topics.normalize + Topics.interpolate(0:0.01:1) from STRIDES.m.
    Maps the stride onto 0–100% gait cycle using linear interpolation.

    Args:
        stride:   Single stride array, shape (n_samples, n_channels).
        n_points: Number of output points (default 101 = 0%, 1%, ..., 100%).

    Returns:
        Time-normalized stride, shape (n_points, n_channels).
    """
    n_samples, n_channels = stride.shape
    t_original = np.linspace(0.0, 1.0, n_samples)
    t_target = np.linspace(0.0, 1.0, n_points)

    normalized = np.empty((n_points, n_channels))
    for ch in range(n_channels):
        normalized[:, ch] = np.interp(t_target, t_original, stride[:, ch])
    return normalized


# -- Step 4: EMG normalization -------------------------------------------------

def compute_emg_normalization(
    strides: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel min/max from the mean stride profile.

    Replicates getEMGNormalization.m: averages all strides (already
    time-normalized to 101 points), then takes the per-channel min and max
    of that mean profile.

    Args:
        strides: List of time-normalized strides, each shape (101, n_channels).

    Returns:
        (min_values, max_values), each shape (n_channels,).
    """
    stacked = np.stack(strides, axis=0)  # (n_strides, 101, n_channels)
    mean_profile = np.nanmean(stacked, axis=0)  # (101, n_channels)
    min_vals = mean_profile.min(axis=0)
    max_vals = mean_profile.max(axis=0)
    return min_vals, max_vals


def normalize_emg(
    emg: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
) -> np.ndarray:
    """Apply min-max normalization: (x - min) / (max - min).

    Args:
        emg:      EMG array, shape (n_samples, n_channels).
        min_vals: Per-channel minimums, shape (n_channels,).
        max_vals: Per-channel maximums, shape (n_channels,).

    Returns:
        Normalized EMG, same shape. Values near [0, 1] for reference-speed
        strides, can exceed this range for faster locomotion.
    """
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0  # avoid division by zero for dead channels
    return (emg - min_vals) / denom


# -- Full pipeline -------------------------------------------------------------

def process_trial_emg(
    emg_table: dict[str, np.ndarray],
    gc_table: dict[str, np.ndarray],
    fs: int = 1000,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Rectify EMG and segment into strides for a single trial.

    Args:
        emg_table: Output of load_mat_table for an EMG file.
                   Must contain "Header" and 11 muscle channel columns.
        gc_table:  Output of load_mat_table for a gcRight file.
                   Must contain "Header" and "HeelStrike".
        fs:        EMG sampling rate.

    Returns:
        (raw_strides, rectified_strides): Lists of per-stride arrays.
        Each stride has shape (n_samples, n_channels) before time normalization.
    """
    # Separate header from EMG channels
    emg_header = emg_table["Header"]
    channel_names = [k for k in emg_table if k != "Header"]
    raw_emg = np.column_stack([emg_table[ch] for ch in channel_names])

    # Rectify
    rect_emg = rectify_emg(raw_emg, fs=fs)

    # Get stride boundaries from gait cycle
    gc_header = gc_table["Header"]
    heel_strike = gc_table["HeelStrike"]
    intervals = find_stride_intervals(gc_header, heel_strike)

    # Segment both raw and rectified (raw is useful for diagnostic plots)
    raw_strides = segment_strides(emg_header, raw_emg, intervals)
    rect_strides = segment_strides(emg_header, rect_emg, intervals)

    return raw_strides, rect_strides


def process_subject(
    emg_tables: list[dict[str, np.ndarray]],
    gc_tables: list[dict[str, np.ndarray]],
    cond_tables: list[dict[str, np.ndarray]],
    ref_speed: float = 1.35,
    fs: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full STRIDES.m pipeline for one subject's treadmill data.

    Args:
        emg_tables:  load_mat_table output for each EMG trial file.
        gc_tables:   load_mat_table output for each gcRight trial file.
        cond_tables: load_mat_table output for each conditions trial file.
        ref_speed:   Treadmill speed for normalization reference (m/s).
        fs:          EMG sampling rate.

    Returns:
        (activations, strides_3d, norm_params):
            activations: Flat (N, 11) array of EMG snapshots for autoencoder.
            strides_3d:  Stride array (n_strides, 101, 11) before flattening,
                         useful for visualization.
            norm_params: Dict with "min_vals" and "max_vals" arrays.
    """
    n_trials = len(emg_tables)
    assert len(gc_tables) == n_trials and len(cond_tables) == n_trials, (
        f"Mismatched trial counts: {n_trials} emg, {len(gc_tables)} gc, {len(cond_tables)} cond"
    )

    # -- Pass 1: rectify all trials, find normalization strides at ref_speed --
    all_rect_strides = []  # (trial_idx, stride_data) for all trials
    ref_strides_tn = []    # time-normalized strides at reference speed only

    for i in range(n_trials):
        _, rect_strides = process_trial_emg(emg_tables[i], gc_tables[i], fs)

        # Check which strides fall within the reference speed window
        emg_header = emg_tables[i]["Header"]
        intervals = find_stride_intervals(gc_tables[i]["Header"], gc_tables[i]["HeelStrike"])

        cond_header = cond_tables[i].get("Header")
        speed = cond_tables[i].get("Speed")
        has_speed = cond_header is not None and speed is not None

        for stride, (t_start, t_end) in zip(rect_strides, intervals):
            all_rect_strides.append((i, stride, t_start, t_end))

            if has_speed:
                # Find speed values within this stride's time window
                mask = (cond_header >= t_start) & (cond_header < t_end)
                stride_speeds = speed[mask]
                if len(stride_speeds) > 0 and np.all(np.abs(stride_speeds - ref_speed) < 0.01):
                    ref_strides_tn.append(time_normalize_stride(stride))

    if not ref_strides_tn:
        logger.warning("No strides found at reference speed %.2f m/s", ref_speed)
        # Fall back: use all strides for normalization
        ref_strides_tn = [time_normalize_stride(s) for _, s, _, _ in all_rect_strides]

    # Compute normalization from mean stride at reference speed
    min_vals, max_vals = compute_emg_normalization(ref_strides_tn)
    logger.info(
        "Normalization from %d reference strides (speed=%.2f)",
        len(ref_strides_tn), ref_speed,
    )

    # -- Pass 2: normalize and time-normalize all strides --
    good_strides = []
    for trial_idx, stride, t_start, t_end in all_rect_strides:
        # Discard acceleration strides (speed changes within stride)
        cond_header = cond_tables[trial_idx].get("Header")
        speed = cond_tables[trial_idx].get("Speed")
        if cond_header is not None and speed is not None:
            mask = (cond_header >= t_start) & (cond_header < t_end)
            stride_speeds = speed[mask]
            if len(stride_speeds) > 0:
                unique_speeds = np.unique(np.round(stride_speeds, 2))
                if len(unique_speeds) > 2:
                    continue  # matches STRIDES.m: discard if >2 unique speeds

        normalized = normalize_emg(stride, min_vals, max_vals)
        tn = time_normalize_stride(normalized)
        good_strides.append(tn)

    if not good_strides:
        logger.warning("No valid strides after filtering")
        return np.empty((0, 11)), np.empty((0, N_GAIT_POINTS, 11)), {}

    strides_3d = np.stack(good_strides, axis=0)  # (n_strides, 101, 11)
    activations = strides_3d.reshape(-1, strides_3d.shape[-1])  # (N, 11)

    logger.info(
        "%d strides → %d activation snapshots, shape %s",
        len(good_strides), activations.shape[0], activations.shape,
    )

    norm_params = {"min_vals": min_vals, "max_vals": max_vals}
    return activations, strides_3d, norm_params
