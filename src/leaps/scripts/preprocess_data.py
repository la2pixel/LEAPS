"""Preprocess raw Camargo EMG data into autoencoder-ready HDF5.

Runs the full STRIDES.m pipeline in Python: rectify → normalize → segment →
time-normalize → flatten. Processes ALL 4 locomotion modes (treadmill,
levelground, ramp, stair). Normalization is computed from treadmill at 1.35 m/s
and applied to all modes (matching STRIDES.m).

## Output HDF5 structure

    /activations          — (N, 11) float64 — ALL subjects & modes pooled.
    │                       Each row = one time-point snapshot of 11 muscles.
    │                       This is the autoencoder's training set.
    │
    ├── attrs:
    │   └── emg_channels  — list of 11 muscle names (column labels)
    │
    /stride_modes         — (M,) string — mode label per stride (e.g. "treadmill")
    /stride_subjects      — (M,) string — subject ID per stride (e.g. "AB09")
    │                       M = N / 101 (each stride = 101 time points)
    │
    /{subject}/strides    — (n_strides, 101, 11) per-subject stride profiles
    /{subject}/modes      — (n_strides,) string — mode label per stride
    /{subject}/min_vals   — (11,) normalization min per channel
    /{subject}/max_vals   — (11,) normalization max per channel

## How to inspect the output

    python -c "
    import h5py, numpy as np
    with h5py.File('emg_activations.h5', 'r') as f:
        print('Activations shape:', f['activations'].shape)
        modes = np.array(f['stride_modes']).astype(str)
        for m in np.unique(modes):
            print(f'  {m}: {np.sum(modes == m)} strides')
    "

Usage:
    leaps-preprocess --data-root /fast/lsivakumar/datasets/camargo \\
                     --output /fast/lsivakumar/data/processed/emg_activations.h5

    # Single subject for testing:
    leaps-preprocess --data-root /fast/lsivakumar/datasets/camargo \\
                     --output test.h5 --subjects AB09
"""

import argparse
import logging
import sys

import h5py
import numpy as np

from leaps.data import discover_trials, load_mat_table
from leaps.data.loaders import ALL_SUBJECTS
from leaps.data.metadata import ALL_MODES, EMG_CHANNELS
from leaps.data.processing import compute_normalization, process_mode_trials

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Camargo EMG → autoencoder HDF5.")
    parser.add_argument("--data-root", required=True, help="Path to raw Camargo dataset root")
    parser.add_argument("--output", required=True, help="Output HDF5 file path")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help=f"Subjects to process (default: all {len(ALL_SUBJECTS)})")
    parser.add_argument("--ref-speed", type=float, default=1.35,
                        help="Treadmill speed for normalization (default: 1.35 m/s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    subjects = args.subjects or ALL_SUBJECTS

    # Discover trials for ALL modes — we need emg + gcRight always,
    # conditions only for treadmill (speed filtering)
    trials = discover_trials(
        args.data_root,
        subjects=subjects,
        modes=ALL_MODES,
        sensors=["emg", "gcRight", "conditions"],
    )

    all_activations = []
    all_stride_modes = []
    all_stride_subjects = []

    with h5py.File(args.output, "w") as hf:
        hf.attrs["emg_channels"] = EMG_CHANNELS

        for i, subj in enumerate(sorted(trials), 1):
            logger.info("[%d/%d] Processing %s", i, len(trials), subj)
            subj_trials = trials[subj]

            # --- Step 1: compute normalization from treadmill ---
            if "treadmill" not in subj_trials:
                logger.warning("Skipping %s: no treadmill data for normalization", subj)
                continue

            tm = subj_trials["treadmill"]
            tm_emg = [load_mat_table(f) for f in sorted(tm.get("emg", []))]
            tm_gc = [load_mat_table(f) for f in sorted(tm.get("gcRight", []))]
            tm_cond = [load_mat_table(f) for f in sorted(tm.get("conditions", []))]

            if not tm_emg or not tm_gc or not tm_cond:
                logger.warning("Skipping %s: incomplete treadmill data", subj)
                continue
            if not all(tm_emg) or not all(tm_gc) or not all(tm_cond):
                logger.warning("Skipping %s: some treadmill files failed to parse", subj)
                continue

            min_vals, max_vals = compute_normalization(
                tm_emg, tm_gc, tm_cond, ref_speed=args.ref_speed,
            )

            # --- Step 2: process all modes with shared normalization ---
            subj_strides = []
            subj_modes = []

            for mode in ALL_MODES:
                if mode not in subj_trials:
                    continue

                mode_data = subj_trials[mode]
                emg_files = sorted(mode_data.get("emg", []))
                gc_files = sorted(mode_data.get("gcRight", []))

                if not emg_files or not gc_files:
                    logger.debug("  %s/%s: no emg/gcRight files, skipping", subj, mode)
                    continue

                emg_tables = [load_mat_table(f) for f in emg_files]
                gc_tables = [load_mat_table(f) for f in gc_files]

                if not all(emg_tables) or not all(gc_tables):
                    logger.warning("  %s/%s: some files failed to parse", subj, mode)
                    continue

                # For treadmill, pass conditions for speed-based filtering
                cond_tables = None
                if mode == "treadmill":
                    cond_tables = tm_cond

                strides_3d = process_mode_trials(
                    emg_tables, gc_tables, min_vals, max_vals,
                    cond_tables=cond_tables,
                )

                n_strides = strides_3d.shape[0]
                if n_strides == 0:
                    logger.debug("  %s/%s: 0 strides after filtering", subj, mode)
                    continue

                subj_strides.append(strides_3d)
                subj_modes.extend([mode] * n_strides)
                logger.info("  %s/%s: %d strides", subj, mode, n_strides)

            if not subj_strides:
                logger.warning("Skipping %s: no valid strides across all modes", subj)
                continue

            # Concatenate all modes for this subject
            all_strides = np.concatenate(subj_strides, axis=0)
            activations = all_strides.reshape(-1, all_strides.shape[-1])

            # Write per-subject data
            grp = hf.create_group(subj)
            grp.create_dataset("strides", data=all_strides, compression="gzip")
            dt = h5py.string_dtype()
            grp.create_dataset("modes", data=np.array(subj_modes, dtype=object), dtype=dt)
            grp.create_dataset("min_vals", data=min_vals)
            grp.create_dataset("max_vals", data=max_vals)

            all_activations.append(activations)
            all_stride_modes.extend(subj_modes)
            all_stride_subjects.extend([subj] * len(subj_modes))

            logger.info(
                "  %s total: %d strides, %d snapshots",
                subj, all_strides.shape[0], activations.shape[0],
            )

        # Write pooled data
        if all_activations:
            pooled = np.concatenate(all_activations, axis=0)
            hf.create_dataset("activations", data=pooled, compression="gzip")

            dt = h5py.string_dtype()
            hf.create_dataset(
                "stride_modes", data=np.array(all_stride_modes, dtype=object), dtype=dt,
            )
            hf.create_dataset(
                "stride_subjects", data=np.array(all_stride_subjects, dtype=object), dtype=dt,
            )

            logger.info(
                "Total: %d strides, %d snapshots from %d subjects",
                len(all_stride_modes), pooled.shape[0], len(all_activations),
            )
            for mode in ALL_MODES:
                count = all_stride_modes.count(mode)
                if count > 0:
                    logger.info("  %s: %d strides", mode, count)
        else:
            logger.error("No data processed.")

    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
