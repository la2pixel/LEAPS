"""Execute full preprocessing pipeline to store as .h5 file 
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from leaps.data.metadata import ALL_MODES, CAMARGO_DATA_ROOT, EMG_CHANNELS, LEAPS_H5_PATH, SUBJECTS
from leaps.data.normalize import compute_normalization
from leaps.data.raw import (
    ALL_SUBJECTS,
    discover_trials,
    extract_condition_labels,
    load_conditions_table,
    load_mat_table,
)
from leaps.data.strides import process_mode_trials

logger = logging.getLogger(__name__)


def build_emg_h5(
    data_root: str | Path,
    output_path: str | Path,
    subjects: list[str] | None = None,
    ref_speed: float = 1.35,
) -> None:
    """Preprocess raw Camargo EMG → HDF5 (replicates STRIDES.m).

    HDF5 layout per subject (all arrays indexed by stride)::

        /{subject}.attrs          — age, gender, height_m, mass_kg
        /{subject}/strides        — (n, 101, 11) float32
        /{subject}/modes          — (n,) str  — treadmill | levelground | ramp | stair
        /{subject}/speeds         — (n,) float64  — m/s for treadmill, NaN otherwise
        /{subject}/conditions     — (n,) str  — speed "1.35" | "slow/normal/fast" | angle | height
        /{subject}/labels         — (n,) str  — phase label: "walk" | "rampascent" | "turnccw" | …
        /{subject}/durations      — (n,) float64  — stride duration in seconds
        /{subject}/trial_indices  — (n,) int32
        /{subject}/min_vals       — (11,) normalization min
        /{subject}/max_vals       — (11,) normalization max

    Pooled across all subjects::

        /activations        — (N*101, 11) float32 snapshot matrix for AE training
        /stride_modes       — (N,) str
        /stride_subjects    — (N,) str
        /stride_speeds      — (N,) float64
        /stride_conditions  — (N,) str
        /stride_labels      — (N,) str
        /stride_durations   — (N,) float64
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subjects = subjects or ALL_SUBJECTS
    trials = discover_trials(
        data_root,
        subjects=subjects,
        modes=ALL_MODES,
        sensors=["emg", "gcRight", "conditions"],
    )

    all_activations:       list[np.ndarray] = []
    all_stride_modes:      list[str]        = []
    all_stride_subjects:   list[str]        = []
    all_stride_speeds:     list[float]      = []
    all_stride_conditions: list[str]        = []
    all_stride_labels:     list[str]        = []
    all_stride_durations:  list[float]      = []

    str_dt = h5py.string_dtype()

    with h5py.File(output_path, "w") as hf:
        hf.attrs["emg_channels"] = EMG_CHANNELS

        for i, subj in enumerate(sorted(trials), 1):
            logger.info("[%d/%d] Processing %s", i, len(trials), subj)
            subj_trials = trials[subj]

            if "treadmill" not in subj_trials:
                logger.warning("Skipping %s: no treadmill data for normalization", subj)
                continue

            tm = subj_trials["treadmill"]
            tm_emg  = [load_mat_table(f) for f in sorted(tm.get("emg",        []))]
            tm_gc   = [load_mat_table(f) for f in sorted(tm.get("gcRight",     []))]
            tm_cond = [load_mat_table(f) for f in sorted(tm.get("conditions",  []))]

            if not all(tm_emg) or not all(tm_gc) or not all(tm_cond):
                logger.warning("Skipping %s: incomplete treadmill data", subj)
                continue

            min_vals, max_vals = compute_normalization(tm_emg, tm_gc, tm_cond, ref_speed=ref_speed)

            subj_strides:       list[np.ndarray] = []
            subj_modes:         list[str]        = []
            subj_speeds:        list[np.ndarray] = []
            subj_trial_indices: list[np.ndarray] = []
            subj_conditions:    list[str]        = []
            subj_labels:        list[str]        = []
            subj_durations:     list[np.ndarray] = []

            for mode in ALL_MODES:
                if mode not in subj_trials:
                    continue

                mode_data = subj_trials[mode]
                emg_files = sorted(mode_data.get("emg",     []))
                gc_files  = sorted(mode_data.get("gcRight", []))
                if not emg_files or not gc_files:
                    continue

                emg_tables = [load_mat_table(f) for f in emg_files]
                gc_tables  = [load_mat_table(f) for f in gc_files]
                if not all(emg_tables) or not all(gc_tables):
                    logger.warning("  %s/%s: some files failed to parse", subj, mode)
                    continue

                cond_files  = sorted(mode_data.get("conditions", []))
                cond_tables = [load_mat_table(f) for f in cond_files] if cond_files else None

                condition_labels = cond_label_data = None
                if mode != "treadmill" and cond_files:
                    condition_labels = [extract_condition_labels(cf, mode) for cf in cond_files]
                    cond_label_data  = [load_conditions_table(cf, mode) for cf in cond_files]
                    n_failed = sum(1 for d in cond_label_data if d is None)
                    if n_failed:
                        logger.warning(
                            "  %s/%s: failed to load per-sample labels for %d/%d trials",
                            subj, mode, n_failed, len(cond_files),
                        )

                strides_3d, mode_speeds, mode_trial_idx, mode_conditions, mode_labels, mode_durations = (
                    process_mode_trials(
                        emg_tables, gc_tables, min_vals, max_vals,
                        cond_tables=cond_tables,
                        condition_labels=condition_labels,
                        cond_label_data=cond_label_data,
                        mode=mode,
                    )
                )

                n = strides_3d.shape[0]
                if n == 0:
                    continue

                if mode != "treadmill":
                    dist = "  ".join(f"{k}:{v}" for k, v in sorted(Counter(mode_conditions).items()))
                    logger.info("  %s/%s: %d strides  [%s]", subj, mode, n, dist)
                else:
                    logger.info("  %s/%s: %d strides", subj, mode, n)

                subj_strides.append(strides_3d)
                subj_modes.extend([mode] * n)
                subj_speeds.append(mode_speeds)
                subj_trial_indices.append(mode_trial_idx)
                subj_conditions.extend(mode_conditions)
                subj_labels.extend(mode_labels)
                subj_durations.append(mode_durations)

            if not subj_strides:
                logger.warning("Skipping %s: no valid strides", subj)
                continue

            all_strides   = np.concatenate(subj_strides,       axis=0)
            all_speeds    = np.concatenate(subj_speeds,         axis=0)
            all_trial_idx = np.concatenate(subj_trial_indices,  axis=0)
            all_durations = np.concatenate(subj_durations,      axis=0)
            activations   = all_strides.reshape(-1, all_strides.shape[-1])

            grp = hf.create_group(subj)
            info = SUBJECTS.get(subj)
            if info:
                grp.attrs["age"]       = info.age
                grp.attrs["gender"]    = info.gender
                grp.attrs["height_m"]  = info.height
                grp.attrs["mass_kg"]   = info.mass

            grp.create_dataset("strides",       data=all_strides,                              compression="gzip")
            grp.create_dataset("modes",          data=np.array(subj_modes,       dtype=object), dtype=str_dt)
            grp.create_dataset("speeds",         data=all_speeds)
            grp.create_dataset("conditions",     data=np.array(subj_conditions,  dtype=object), dtype=str_dt)
            grp.create_dataset("labels",         data=np.array(subj_labels,      dtype=object), dtype=str_dt)
            grp.create_dataset("durations",      data=all_durations)
            grp.create_dataset("trial_indices",  data=all_trial_idx)
            grp.create_dataset("min_vals",       data=min_vals)
            grp.create_dataset("max_vals",       data=max_vals)

            all_activations.extend([activations])
            all_stride_modes.extend(subj_modes)
            all_stride_subjects.extend([subj] * len(subj_modes))
            all_stride_speeds.extend(all_speeds.tolist())
            all_stride_conditions.extend(subj_conditions)
            all_stride_labels.extend(subj_labels)
            all_stride_durations.extend(all_durations.tolist())

            logger.info("  %s total: %d strides, %d snapshots", subj, all_strides.shape[0], activations.shape[0])

        if not all_activations:
            logger.error("No data processed.")
            return

        pooled = np.concatenate(all_activations, axis=0)
        hf.create_dataset("activations",       data=pooled,                                               compression="gzip")
        hf.create_dataset("stride_modes",      data=np.array(all_stride_modes,      dtype=object),       dtype=str_dt)
        hf.create_dataset("stride_subjects",   data=np.array(all_stride_subjects,   dtype=object),       dtype=str_dt)
        hf.create_dataset("stride_speeds",     data=np.array(all_stride_speeds,     dtype=np.float64))
        hf.create_dataset("stride_conditions", data=np.array(all_stride_conditions, dtype=object),       dtype=str_dt)
        hf.create_dataset("stride_labels",     data=np.array(all_stride_labels,     dtype=object),       dtype=str_dt)
        hf.create_dataset("stride_durations",  data=np.array(all_stride_durations,  dtype=np.float64))

    logger.info(
        "Wrote %s — %d strides, %d snapshots from %d subjects",
        output_path, len(all_stride_modes), pooled.shape[0], len(all_activations),
    )
    for mode in ALL_MODES:
        count = all_stride_modes.count(mode)
        if count:
            logger.info("  %s: %d strides", mode, count)


# CLI entry point: "leaps preprocess"

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Camargo EMG → HDF5.")
    parser.add_argument("--data-root", default=CAMARGO_DATA_ROOT)
    parser.add_argument("--output", default=LEAPS_H5_PATH)
    parser.add_argument("--subjects", nargs="+", default=None,
                        help=f"Subjects to process (default: all {len(ALL_SUBJECTS)})")
    parser.add_argument("--ref-speed", type=float, default=1.35)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    build_emg_h5(
        data_root=args.data_root,
        output_path=args.output,
        subjects=args.subjects,
        ref_speed=args.ref_speed,
    )


if __name__ == "__main__":
    main()
