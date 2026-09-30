"""Per-subject EMG normalization (Camargo getEMGNormalization.m)
"""

from __future__ import annotations

import logging

import numpy as np

from leaps.data.strides import process_trial_emg, time_normalize_stride

logger = logging.getLogger(__name__)


def compute_emg_normalization(strides: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """per-channel (min, max) of the mean stride profile"""
    mean_profile = np.nanmean(np.stack(strides, axis=0), axis=0)
    return mean_profile.min(axis=0), mean_profile.max(axis=0)


def normalize_emg(emg: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    denom = np.where(max_vals == min_vals, 1.0, max_vals - min_vals)
    return (emg - min_vals) / denom


def compute_normalization(
    emg_tables: list[dict[str, np.ndarray]],
    gc_tables: list[dict[str, np.ndarray]],
    cond_tables: list[dict[str, np.ndarray]],
    ref_speed: float = 1.35,
    fs: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """per-channel (min, max) from treadmill strides at ref_speed (from STRIDES.m at 1.35m/s)"""
    all_tn: list[np.ndarray] = []
    ref_tn: list[np.ndarray] = []

    for emg_t, gc_t, cond_t in zip(emg_tables, gc_tables, cond_tables):
        rect_strides, intervals = process_trial_emg(emg_t, gc_t, fs)
        cond_header = cond_t.get("Header")
        speed_arr   = cond_t.get("Speed")
        has_speed   = cond_header is not None and speed_arr is not None

        for stride, (t0, t1) in zip(rect_strides, intervals):
            tn = time_normalize_stride(stride)
            all_tn.append(tn)
            if has_speed:
                s = speed_arr[(cond_header >= t0) & (cond_header < t1)]
                if len(s) > 0 and np.all(np.abs(s - ref_speed) < 0.01):
                    ref_tn.append(tn)

    if not ref_tn:
        logger.warning("No strides at reference speed %.2f- using all treadmill strides", ref_speed)
        ref_tn = all_tn

    min_vals, max_vals = compute_emg_normalization(ref_tn)
    logger.info("Normalization from %d ref strides (speed=%.2f)", len(ref_tn), ref_speed)
    return min_vals, max_vals
