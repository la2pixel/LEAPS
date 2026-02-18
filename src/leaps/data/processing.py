"""EMG preprocessing pipeline replicating the Camargo STRIDES.m workflow.

Transforms raw EMG from .mat files into autoencoder-ready (N, 11) activation
snapshots. This replicates the MATLAB scripts from the Camargo study:
  - rectify.m          → rectify_emg()
  - getEMGNormalization.m → compute_emg_normalization()
  - segment_gc.m       → find_stride_intervals() + segment_strides()
  - STRIDES.m          → compute_normalization() + process_mode_trials()

## Full pipeline per subject

    Raw EMG (mV, noisy, 11 channels x ~30k samples per trial)
        │
        ▼
    1. RECTIFY (rectify_emg)
       Highpass 10Hz → Bandpass 10-450Hz → |abs| → Lowpass 6Hz → |abs|
       Removes noise, extracts smooth muscle activation envelope
        │
        ▼
    2. SEGMENT INTO STRIDES (find_stride_intervals + segment_strides)
       Use gcRight HeelStrike sawtooth (0-100%) to find stride boundaries
       Each stride = one full gait cycle (heel strike → next heel strike)
        │
        ▼
    3. TIME-NORMALIZE (time_normalize_stride)
       Interpolate each stride to exactly 101 points (0%, 1%, ..., 100%)
       So all strides have the same length regardless of walking speed
        │
        ▼
    4. NORMALIZE AMPLITUDE (compute_emg_normalization + normalize_emg)
       Min-max normalize using the mean stride profile at 1.35 m/s
       This makes values comparable across subjects and muscles
        │
        ▼
    5. FLATTEN (reshape in preprocess_data.py)
       Stack all (101, 11) strides → (N*101, 11) matrix
       Each row = one time-point snapshot of 11 muscle activations
       This is the input to the autoencoder / NMF / PCA
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

    Replicates gc2intervals() in segment_gc.m.

    The HeelStrike column from gcRight looks like a sawtooth wave:
        0% ──── 50% ──── 99% ╲ 0% ──── 50% ──── 99% ╲ 0% ────
                              ↑                        ↑
                         heel strike               heel strike

    Each reset from ~100% back to ~0% marks a heel strike. One stride =
    the interval between two consecutive heel strikes.

    We detect heel strikes by finding troughs (local minima near 0) in
    the sawtooth. This matches the MATLAB approach of findpeaks(-gc).

    Args:
        gc_header:   Time vector from gcRight, shape (n,). Sampled at 200 Hz.
        heel_strike: HeelStrike column from gcRight, shape (n,). Values 0–100%.

    Returns:
        List of (start_time, end_time) tuples in seconds, one per stride.
        Typically ~30-50 strides per trial (30s of walking at ~1 stride/sec).
    """
    gc = heel_strike.copy()

    # Find trough indices: points lower than both neighbors (= heel strikes).
    # The threshold < 20% filters out noise — real heel strikes reset to near 0%.
    trough_idx = []
    for i in range(1, len(gc) - 1):
        if gc[i] < gc[i - 1] and gc[i] <= gc[i + 1]:
            if gc[i] < 20.0:
                trough_idx.append(i)

    # Handle recording boundaries: the first/last stride may be partial.
    # Include the edge of the valid region (where gc > 0) as a boundary.
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

    # Pair consecutive boundaries into (start, end) time intervals
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


def compute_normalization(
    emg_tables: list[dict[str, np.ndarray]],
    gc_tables: list[dict[str, np.ndarray]],
    cond_tables: list[dict[str, np.ndarray]],
    ref_speed: float = 1.35,
    fs: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel min/max normalization from treadmill at reference speed.

    Replicates STRIDES.m: normalization is always derived from treadmill walking
    at a single reference speed (1.35 m/s), then applied to ALL modes.

    Args:
        emg_tables:  load_mat_table output for each treadmill EMG trial.
        gc_tables:   load_mat_table output for each treadmill gcRight trial.
        cond_tables: load_mat_table output for each treadmill conditions trial.
        ref_speed:   Reference treadmill speed in m/s.
        fs:          EMG sampling rate.

    Returns:
        (min_vals, max_vals), each shape (n_channels,).
    """
    n_trials = len(emg_tables)
    ref_strides_tn = []

    for i in range(n_trials):
        _, rect_strides = process_trial_emg(emg_tables[i], gc_tables[i], fs)
        intervals = find_stride_intervals(gc_tables[i]["Header"], gc_tables[i]["HeelStrike"])

        cond_header = cond_tables[i].get("Header")
        speed = cond_tables[i].get("Speed")
        has_speed = cond_header is not None and speed is not None

        for stride, (t_start, t_end) in zip(rect_strides, intervals):
            if has_speed:
                mask = (cond_header >= t_start) & (cond_header < t_end)
                stride_speeds = speed[mask]
                if len(stride_speeds) > 0 and np.all(np.abs(stride_speeds - ref_speed) < 0.01):
                    ref_strides_tn.append(time_normalize_stride(stride))

    if not ref_strides_tn:
        logger.warning("No strides at ref speed %.2f — using all treadmill strides", ref_speed)
        for i in range(n_trials):
            _, rect_strides = process_trial_emg(emg_tables[i], gc_tables[i], fs)
            ref_strides_tn.extend(time_normalize_stride(s) for s in rect_strides)

    min_vals, max_vals = compute_emg_normalization(ref_strides_tn)
    logger.info("Normalization from %d ref strides (speed=%.2f)", len(ref_strides_tn), ref_speed)
    return min_vals, max_vals


# Heuristic stride filtering thresholds for non-treadmill modes
_MIN_STRIDE_DURATION = 0.3   # seconds — shorter is likely noise/partial
_MAX_STRIDE_DURATION = 3.0   # seconds — longer is likely standing/idle
_MIN_EMG_AMPLITUDE = 0.01    # fraction of max — lower is likely resting


def process_mode_trials(
    emg_tables: list[dict[str, np.ndarray]],
    gc_tables: list[dict[str, np.ndarray]],
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    cond_tables: list[dict[str, np.ndarray]] | None = None,
    fs: int = 1000,
) -> np.ndarray:
    """Rectify, segment, normalize, and time-normalize trials for any mode.

    For treadmill (when cond_tables provided with Speed): filters out
    acceleration strides where speed changes mid-stride.

    For non-treadmill (no cond_tables or no Speed column): uses heuristic
    filtering based on stride duration and EMG amplitude.

    Args:
        emg_tables:  load_mat_table output for each EMG trial.
        gc_tables:   load_mat_table output for each gcRight trial.
        min_vals:    Per-channel normalization min (from compute_normalization).
        max_vals:    Per-channel normalization max.
        cond_tables: Conditions tables (only needed for treadmill speed filtering).
        fs:          EMG sampling rate.

    Returns:
        strides_3d: (n_strides, 101, 11) array of normalized stride profiles.
    """
    n_trials = len(emg_tables)
    good_strides = []

    for i in range(n_trials):
        _, rect_strides = process_trial_emg(emg_tables[i], gc_tables[i], fs)
        gc_header = gc_tables[i]["Header"]
        intervals = find_stride_intervals(gc_header, gc_tables[i]["HeelStrike"])

        # Get speed data if available (treadmill mode)
        has_speed = False
        if cond_tables is not None and i < len(cond_tables):
            cond_header = cond_tables[i].get("Header")
            speed = cond_tables[i].get("Speed")
            has_speed = cond_header is not None and speed is not None

        for stride, (t_start, t_end) in zip(rect_strides, intervals):
            duration = t_end - t_start

            if has_speed:
                # Treadmill: discard acceleration strides (>2 unique speeds)
                mask = (cond_header >= t_start) & (cond_header < t_end)
                stride_speeds = speed[mask]
                if len(stride_speeds) > 0:
                    unique_speeds = np.unique(np.round(stride_speeds, 2))
                    if len(unique_speeds) > 2:
                        continue
            else:
                # Non-treadmill: heuristic filtering
                if duration < _MIN_STRIDE_DURATION or duration > _MAX_STRIDE_DURATION:
                    continue
                if np.max(np.abs(stride)) < _MIN_EMG_AMPLITUDE * np.max(max_vals):
                    continue

            normalized = normalize_emg(stride, min_vals, max_vals)
            tn = time_normalize_stride(normalized)
            good_strides.append(tn)

    if not good_strides:
        logger.warning("No valid strides after filtering")
        return np.empty((0, N_GAIT_POINTS, 11))

    strides_3d = np.stack(good_strides, axis=0)
    logger.info("%d strides → shape %s", len(good_strides), strides_3d.shape)
    return strides_3d
