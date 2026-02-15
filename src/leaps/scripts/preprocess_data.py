"""Preprocess raw Camargo EMG data into autoencoder-ready HDF5.

Runs the full STRIDES.m pipeline in Python: rectify → normalize → segment →
time-normalize → flatten. Outputs an HDF5 file with:

    /activations          — (N, 11) float64, all EMG snapshots pooled
    /{subject}/strides    — (n_strides, 101, 11) per-subject stride profiles
    /{subject}/min_vals   — (11,) normalization minimums
    /{subject}/max_vals   — (11,) normalization maximums

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

from leaps.data.loaders import ALL_SUBJECTS, discover_trials, load_mat_table
from leaps.data.metadata import EMG_CHANNELS
from leaps.data.processing import process_subject

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

    # Discover treadmill trials (we need emg + gcRight + conditions per trial)
    trials = discover_trials(
        args.data_root,
        subjects=subjects,
        modes=["treadmill"],
        sensors=["emg", "gcRight", "conditions"],
    )

    all_activations = []

    with h5py.File(args.output, "w") as hf:
        # Store channel names as a top-level attribute
        hf.attrs["emg_channels"] = EMG_CHANNELS

        for i, subj in enumerate(sorted(trials), 1):
            logger.info("[%d/%d] Processing %s", i, len(trials), subj)

            subj_trials = trials[subj]["treadmill"]
            emg_files = subj_trials["emg"]
            gc_files = subj_trials["gcRight"]
            cond_files = subj_trials["conditions"]

            # Load all trial files for this subject
            emg_tables = [load_mat_table(f) for f in sorted(emg_files)]
            gc_tables = [load_mat_table(f) for f in sorted(gc_files)]
            cond_tables = [load_mat_table(f) for f in sorted(cond_files)]

            # Skip subjects with missing data
            if not all(emg_tables) or not all(gc_tables) or not all(cond_tables):
                logger.warning("Skipping %s: some files failed to parse", subj)
                continue

            activations, strides_3d, norm_params = process_subject(
                emg_tables, gc_tables, cond_tables,
                ref_speed=args.ref_speed,
            )

            if activations.shape[0] == 0:
                logger.warning("Skipping %s: no valid strides", subj)
                continue

            # Write per-subject data
            grp = hf.create_group(subj)
            grp.create_dataset("strides", data=strides_3d, compression="gzip")
            grp.create_dataset("min_vals", data=norm_params["min_vals"])
            grp.create_dataset("max_vals", data=norm_params["max_vals"])

            all_activations.append(activations)
            logger.info("  %s: %d strides, %d snapshots", subj, strides_3d.shape[0], activations.shape[0])

        # Write pooled activations (the autoencoder training set)
        if all_activations:
            pooled = np.concatenate(all_activations, axis=0)
            hf.create_dataset("activations", data=pooled, compression="gzip")
            logger.info("Total: %d activation snapshots from %d subjects", pooled.shape[0], len(all_activations))
        else:
            logger.error("No data processed.")

    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
