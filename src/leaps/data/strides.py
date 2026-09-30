"""Gait-cycle segmentation and stride labeling
"""

from __future__ import annotations

import logging

import numpy as np

from leaps.data.filters import rectify_emg

logger = logging.getLogger(__name__)

N_GAIT_POINTS = 101


def find_stride_intervals(gc_header: np.ndarray, heel_strike: np.ndarray) -> list[tuple[float, float]]:
    """Return (t_start, t_end) pairs from the gait-cycle sawtooth (replicates gc2intervals).
    """
    gc = heel_strike.copy()
    trough_idx = [
        i for i in range(1, len(gc) - 1)
        if gc[i] < gc[i - 1] and gc[i] <= gc[i + 1] and gc[i] < 20.0
    ]

    first_pos = np.argmax(gc > 0)
    last_pos  = len(gc) - 1 - np.argmax(gc[::-1] > 0)
    boundaries = (
        ([first_pos - 1] if first_pos > 0 else [])
        + trough_idx
        + ([last_pos + 1] if last_pos < len(gc) - 1 else [])
    )

    if len(boundaries) < 2:
        return []

    times = gc_header[boundaries]
    return [(times[i], times[i + 1]) for i in range(len(times) - 1)]


def segment_strides(
    header: np.ndarray,
    data: np.ndarray,
    intervals: list[tuple[float, float]],
) -> list[np.ndarray]:
    result = []
    for t0, t1 in intervals:
        seg = data[(header >= t0) & (header < t1)]
        if len(seg) > 10:
            result.append(seg)
    return result


def time_normalize_stride(stride: np.ndarray, n_points: int = N_GAIT_POINTS) -> np.ndarray:
    """Interpolate stride to n_points over [0, 1] gait cycle (replicates Topics.interpolate)."""
    n_samples, n_channels = stride.shape
    t_in  = np.linspace(0.0, 1.0, n_samples)
    t_out = np.linspace(0.0, 1.0, n_points)
    out = np.empty((n_points, n_channels))
    for ch in range(n_channels):
        out[:, ch] = np.interp(t_out, t_in, stride[:, ch])
    return out


def process_trial_emg(
    emg_table: dict[str, np.ndarray],
    gc_table: dict[str, np.ndarray],
    fs: int = 1000,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Rectify EMG and segment into strides; return (rect_strides, intervals)."""
    emg_header = emg_table["Header"]
    channels   = [k for k in emg_table if k != "Header"]
    rect_emg   = rectify_emg(np.column_stack([emg_table[ch] for ch in channels]), fs=fs)
    intervals  = find_stride_intervals(gc_table["Header"], gc_table["HeelStrike"])
    return segment_strides(emg_header, rect_emg, intervals), intervals


def classify_stride_label(labels: list[str], mode: str, meta: dict) -> str | None:
    """Classify a stride's phase label. Returns the label string to keep ("walk", "rampascent", "turnccw", etc.) or None to discard.
    """
    if not labels:
        return None

    first     = labels[0]
    last      = labels[-1]
    label_set = set(labels)
    has_turn  = any("turn" in l for l in labels)

    if mode == "levelground":
        if all(l == "idle" for l in labels) or first == "idle":
            return None
        leg_start = str(meta.get("leadingLegStart", ""))
        leg_stop  = str(meta.get("leadingLegStop",  ""))
        turn      = str(meta.get("turn", ""))
        if first == "stand-walk" and leg_start == "l":
            return "stand-walk"
        if first == "stand" and last == "stand-walk" and leg_start == "r":
            return None
        if first == "stand" and last == "stand-walk" and leg_start == "l":
            return "stand-walk"
        if has_turn and turn == "ccw":
            return "turnccw"
        if has_turn and turn == "cw":
            return "turncw"
        if all(l == "walk" for l in labels):
            return "walk"
        if last == "walk-stand" and leg_stop == "r":
            return "walk-stand"
        if first == "walk-stand" and last == "stand" and leg_stop == "r":
            return "stand"
        if first == "stand-walk" and last == "walk" and leg_start == "r":
            return "walk"
        if first == "walk" and last == "walk-stand" and leg_stop == "l":
            return "walk-stand"
        if first == "walk-stand" and leg_stop == "l":
            return None
        if first == "walk-stand" and last == "idle" and "stand" in label_set:
            return None
        if first == "walk-stand" and last == "stand" and leg_stop == "l":
            return "stand"
        if (first == "walk" and last == "idle" and leg_stop == "r"
                and "walk-stand" in label_set and "stand" in label_set):
            return None
        if first == "stand" and leg_start == "l":
            return "stand-walk"
        logger.debug("classify_stride_label: unhandled levelground case first=%r last=%r", first, last)
        return None

    elif mode == "ramp":
        if all(l == "idle" for l in labels) or first == "idle":
            return None
        asc  = list(meta.get("transLegAscent",  []))
        desc = list(meta.get("transLegDescent", []))
        if first == "rampascent-walk" and last == "walk-rampdescent" and "idle" in label_set:
            return None
        if first == "walk-rampascent" and asc[:1] == ["l"]:
            return "rampascent"
        if first == "walk-rampascent" and asc[:1] == ["r"]:
            return "walk-rampascent"
        if all(l == "rampascent" for l in labels):
            return "rampascent"
        if last == "rampascent-walk":
            return "rampascent-walk"
        if first == "rampascent-walk" and last == "idle":
            return None
        if first == "walk-rampdescent" and desc[:1] == ["l"]:
            return "rampdescent"
        if first == "walk-rampdescent" and desc[:1] == ["r"]:
            return "walk-rampdescent"
        if all(l == "rampdescent" for l in labels):
            return "rampdescent"
        if last == "rampdescent-walk":
            return "rampdescent-walk"
        if first == "rampdescent-walk" and last == "idle":
            return None
        if first == "rampdescent-walk" and last == "walk-rampascent" and asc[:1] == ["r"]:
            return None
        if first == "rampascent-walk" and last == "rampdescent" and "walk-rampdescent" in label_set:
            return None
        logger.debug("classify_stride_label: unhandled ramp case first=%r last=%r", first, last)
        return None

    elif mode == "stair":
        if all(l == "idle" for l in labels):
            return None
        asc  = list(meta.get("transLegAscent",  []))
        desc = list(meta.get("transLegDescent", []))
        asc0  = asc[0]  if len(asc)  > 0 else ""
        asc1  = asc[1]  if len(asc)  > 1 else ""
        desc0 = desc[0] if len(desc) > 0 else ""
        desc1 = desc[1] if len(desc) > 1 else ""
        if first == "idle" and last == "walk-stairascent" and asc0 == "l":
            return "walk-stairascent"
        if first == "walk-stairascent" and last == "stairascent" and asc0 == "l":
            return "stairascent"
        if first == "stairascent" and last == "stairascent-walk" and asc1 == "r":
            return "stairascent-walk"
        if first == "stairascent-walk" and last == "idle" and asc1 == "r":
            return None
        if first == "idle" and last == "walk-stairdescent" and desc0 == "l":
            return "walk-stairdescent"
        if first == "walk-stairdescent" and last == "stairdescent" and desc0 == "l":
            return "stairdescent"
        if first == "stairdescent" and last == "stairdescent-walk" and desc1 == "r":
            return "stairdescent-walk"
        if first == "stairdescent-walk" and last == "idle" and desc1 == "r":
            return None
        if first == "idle" and last == "walk-stairascent" and asc0 == "r":
            return None
        if first == "walk-stairascent" and last == "stairascent" and asc0 == "r":
            return "stairascent"
        if all(l == "stairascent" for l in labels):
            return "stairascent"
        if first == "stairascent" and last == "stairascent-walk" and asc0 == "r":
            return "stairascent-walk"
        if first == "stairascent-walk" and last == "idle" and asc1 == "l":
            return None
        if first == "idle" and last == "walk-stairdescent" and desc0 == "r":
            return None
        if first == "walk-stairdescent" and last == "stairdescent" and desc0 == "r":
            return "stairdescent"
        if all(l == "stairdescent" for l in labels):
            return "stairdescent"
        if first == "stairdescent" and last == "stairdescent-walk" and desc1 == "l":
            return "stairdescent-walk"
        if first == "stairdescent-walk" and last == "idle" and desc1 == "l":
            return None
        if first == "stairascent-walk" and last == "walk-stairdescent" and "idle" in label_set and asc1 == "l":
            return None
        if first == "stairascent-walk" and last == "walk-stairdescent" and asc1 == "r":
            return None
        logger.debug("classify_stride_label: unhandled stair case first=%r last=%r", first, last)
        return None

    return None


def process_mode_trials(
    emg_tables: list[dict[str, np.ndarray]],
    gc_tables: list[dict[str, np.ndarray]],
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    cond_tables: list[dict[str, np.ndarray]] | None = None,
    condition_labels: list[str | None] | None = None,
    cond_label_data: list[tuple | None] | None = None,
    mode: str = "",
    fs: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    """Rectify, segment, normalize, and time-normalize trials for any mode.

    Values here might lie outside [0, 1]!
            conditions (n,), labels (n,), durations (n,).
    """
    from leaps.data.normalize import normalize_emg

    strides, speeds, trial_indices, conditions, labels, durations = [], [], [], [], [], []

    for i, (emg_t, gc_t) in enumerate(zip(emg_tables, gc_tables)):
        rect_strides, intervals = process_trial_emg(emg_t, gc_t, fs)

        cond_header = cond_speed = None
        if cond_tables and i < len(cond_tables):
            cond_header = cond_tables[i].get("Header")
            cond_speed  = cond_tables[i].get("Speed")
        has_speed = cond_header is not None and cond_speed is not None

        label_entry = cond_label_data[i] if cond_label_data and i < len(cond_label_data) else None
        trial_cond  = (
            condition_labels[i]
            if condition_labels and i < len(condition_labels) and condition_labels[i]
            else "unknown"
        )

        for stride, (t0, t1) in zip(rect_strides, intervals):
            mean_speed = float("nan")

            if has_speed:
                s = cond_speed[(cond_header >= t0) & (cond_header < t1)]
                if len(s) == 0:
                    continue
                # discard acceleration strides — MATLAB: numel(unique(round(speed, 2))) > 2
                if len(np.unique(np.round(s, 2))) > 2:
                    continue
                mean_speed   = float(np.mean(s))
                stride_label = "walk"

            elif label_entry is not None:
                lbl_header, per_sample_labels, meta = label_entry
                stride_sample_labels = [
                    per_sample_labels[k] for k in np.where((lbl_header >= t0) & (lbl_header < t1))[0]
                ]
                classified = classify_stride_label(stride_sample_labels, mode, meta)
                if classified is None:
                    continue
                stride_label = classified

            else:
                continue

            tn = time_normalize_stride(normalize_emg(stride, min_vals, max_vals))
            condition = f"{mean_speed:.2f}" if not np.isnan(mean_speed) else trial_cond

            strides.append(tn)
            speeds.append(mean_speed)
            trial_indices.append(i)
            conditions.append(condition)
            labels.append(stride_label)
            durations.append(t1 - t0)

    if not strides:
        logger.warning("No valid strides after filtering")
        return (
            np.empty((0, N_GAIT_POINTS, 11), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            [], [],
            np.empty(0, dtype=np.float64),
        )

    logger.info("%d strides → shape (%d, %d, 11)", len(strides), len(strides), N_GAIT_POINTS)
    return (
        np.stack(strides, axis=0),
        np.array(speeds,        dtype=np.float64),
        np.array(trial_indices, dtype=np.int32),
        conditions,
        labels,
        np.array(durations,     dtype=np.float64),
    )
