"""Preprocess raw Camargo EMG data into files of HDF5 format

I implement the Camargo STRIDES.m example file in Python.
Explanation found in: https://www.notion.so/Post-processing-Steps-334ab61e5b1e803fbe94d773c1e1d86c
    

Run the module:
    leaps-preprocess --data-root /fast/lsivakumar/datasets/camargo \\
                     --output /fast/lsivakumar/data/processed/emg_activations.h5

    #to test for a single subject:
    leaps-preprocess --data-root /fast/lsivakumar/datasets/camargo \\
                     --output test.h5 --subjects AB09
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

from leaps.data import discover_trials, load_mat_table
from leaps.data.loaders import ALL_SUBJECTS, extract_condition_labels
from leaps.data.metadata import ALL_MODES, CAMARGO_DATA_ROOT, EMG_CHANNELS, LEAPS_H5_PATH
from leaps.data.processing import compute_normalization, process_mode_trials

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Camargo EMG → autoencoder HDF5.")
    parser.add_argument("--data-root", default=CAMARGO_DATA_ROOT, help="Path to raw Camargo dataset root")
    parser.add_argument("--output", default=LEAPS_H5_PATH, help="Output HDF5 file path")
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

    #Discover trials for ALL modes.
    #we need emg + gcRight always, conditions only for treadmill to filter based on speed
    trials = discover_trials(
        args.data_root,
        subjects=subjects,
        modes=ALL_MODES,
        sensors=["emg", "gcRight", "conditions"],
    )

    all_activations = []
    all_stride_modes = []
    all_stride_subjects = []
    all_stride_speeds = []
    all_stride_conditions = []

    with h5py.File(args.output, "w") as hf:
        hf.attrs["emg_channels"] = EMG_CHANNELS

        for i, subj in enumerate(sorted(trials), 1):
            logger.info("[%d/%d] Processing %s", i, len(trials), subj)
            subj_trials = trials[subj]

            #Step 2: time normalize with treadmill trials
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
                tm_emg, tm_gc, tm_cond, ref_speed=args.ref_speed,  #normalized here
            )

            #Process all modes with normalization
            subj_strides = []
            subj_modes = []
            subj_speeds = []
            subj_trial_indices = []
            subj_conditions = []

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

                #Load conditions tables for ALL modes:
                #   - treadmill for speed-based stride filtering
                #   - others for per-sample label time-alignment
                cond_files = sorted(mode_data.get("conditions", []))
                cond_tables = [load_mat_table(f) for f in cond_files] if cond_files else None

                #Extract condition labels for non-treadmill modes.
                #for treadmill the condition is derived from the numeric Speed column.
                condition_labels = None
                if mode != "treadmill" and cond_files:
                    condition_labels = []
                    for cf, ct in zip(cond_files, (cond_tables or [])):
                        # Pass n_rows so per-sample reconstruction can be attempted
                        n_rows = len(ct.get("Header", [])) if ct else None
                        label = extract_condition_labels(cf, mode, n_rows=n_rows)
                        condition_labels.append(label)
                        if label is None:
                            logger.debug(
                                "  %s/%s: no condition label for %s",
                                subj, mode, Path(cf).name,
                            )
                        else:
                            n_labels = len(label) if isinstance(label, list) else 1
                            logger.debug(
                                "  %s/%s: %s → %s (%d samples)",
                                subj, mode, Path(cf).name,
                                label if isinstance(label, str) else f"[{label[0]}…]",
                                n_labels,
                            )

                strides_3d, mode_speeds, mode_trial_indices, mode_conditions = process_mode_trials(
                    emg_tables, gc_tables, min_vals, max_vals,
                    cond_tables=cond_tables,
                    condition_labels=condition_labels,
                )

                n_strides = strides_3d.shape[0]
                if n_strides == 0:
                    logger.debug("  %s/%s: 0 strides after filtering", subj, mode)
                    continue

                #log condition distribution for this mode
                if mode != "treadmill":
                    from collections import Counter
                    cond_counts = Counter(mode_conditions)
                    dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(cond_counts.items()))
                    logger.info("  %s/%s: %d strides  [%s]", subj, mode, n_strides, dist_str)
                else:
                    logger.info("  %s/%s: %d strides", subj, mode, n_strides)

                subj_strides.append(strides_3d)
                subj_modes.extend([mode] * n_strides)
                subj_speeds.append(mode_speeds)
                subj_trial_indices.append(mode_trial_indices)
                subj_conditions.extend(mode_conditions)

            if not subj_strides:
                logger.warning("Skipping %s: no valid strides across all modes", subj)
                continue

            #concat all modes for this subject
            all_strides = np.concatenate(subj_strides, axis=0)
            all_speeds = np.concatenate(subj_speeds, axis=0)
            all_trial_indices = np.concatenate(subj_trial_indices, axis=0)
            activations = all_strides.reshape(-1, all_strides.shape[-1])

            #Write per-subject data
            grp = hf.create_group(subj)
            grp.create_dataset("strides", data=all_strides, compression="gzip")
            str_dt = h5py.string_dtype()
            grp.create_dataset("modes", data=np.array(subj_modes, dtype=object), dtype=str_dt)
            grp.create_dataset("speeds", data=all_speeds)       # NaN for non-treadmill
            grp.create_dataset("trial_indices", data=all_trial_indices)
            grp.create_dataset(
                "conditions",
                data=np.array(subj_conditions, dtype=object),
                dtype=str_dt,
            )
            grp.create_dataset("min_vals", data=min_vals)
            grp.create_dataset("max_vals", data=max_vals)

            all_activations.append(activations)
            all_stride_modes.extend(subj_modes)
            all_stride_subjects.extend([subj] * len(subj_modes))
            all_stride_speeds.extend(all_speeds.tolist())
            all_stride_conditions.extend(subj_conditions)

            logger.info(
                "  %s total: %d strides, %d snapshots",
                subj, all_strides.shape[0], activations.shape[0],
            )

        # Write pooled data
        if all_activations:
            pooled = np.concatenate(all_activations, axis=0)
            hf.create_dataset("activations", data=pooled, compression="gzip")

            str_dt = h5py.string_dtype()
            hf.create_dataset(
                "stride_modes", data=np.array(all_stride_modes, dtype=object), dtype=str_dt,
            )
            hf.create_dataset(
                "stride_subjects", data=np.array(all_stride_subjects, dtype=object), dtype=str_dt,
            )
            hf.create_dataset(
                "stride_speeds", data=np.array(all_stride_speeds, dtype=np.float64),
            )
            hf.create_dataset(
                "stride_conditions",
                data=np.array(all_stride_conditions, dtype=object),
                dtype=str_dt,
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
