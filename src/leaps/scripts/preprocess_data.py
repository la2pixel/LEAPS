"""CLI entry point for preprocessing raw Camargo EMG data into HDF5.

Usage:
    leaps preprocess
    leaps preprocess --subjects AB09 AB10
    leaps preprocess --data-root /path/to/camargo --output /path/to/out.h5
"""

import argparse
import logging
import sys

from leaps.data.loaders import ALL_SUBJECTS, build_emg_h5
from leaps.data.metadata import CAMARGO_DATA_ROOT, LEAPS_H5_PATH


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
