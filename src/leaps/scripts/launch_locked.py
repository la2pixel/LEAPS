"""Opinionated launcher for the locked-in production experiment set.

k=6, w=0.5, unclipped, mirror on, pooled decoder, net256 -- the parameters
locked in 2026-08-15 -- are hardcoded, not flags. Only asks for what
actually varies run to run: which body models, which reward variants,
which seeds, which category. For anything outside this locked set (other
k/w/clip values, AB06/AB20 decoders, style rewards) use launch_explore.py
instead, which shares this same generator core.

Usage:
    python -m leaps.scripts.launch_locked --body h0918 h1622 h2190 \\
        --reward-variant onlyvelrew --seeds 0 1 2 --dry-run

    # generate for real and print the run_queue.sh command to launch:
    python -m leaps.scripts.launch_locked --body h0918 --seeds 0 1 \\
        --queue-out run_scripts/queue_locked.txt --rolling 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from leaps.configs.spec import BODIES, Category, RewardVariant, RunSpec
from leaps.scripts._launch_common import generate_and_queue

# The 2026-08-15 lock-in. Not exposed as flags -- that's the point.
_LOCKED_DIM_LATENT = 6
_LOCKED_MAPPED_RESIDUAL_WEIGHT = 0.5
_LOCKED_CLIP = False
_LOCKED_MIRROR_LEFT = True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", nargs="+", required=True, choices=sorted(BODIES), help="body model(s)")
    p.add_argument(
        "--reward-variant",
        nargs="+",
        default=["onlyvelrew"],
        choices=[v.value for v in RewardVariant],
        help="reward variant(s), default: onlyvelrew",
    )
    p.add_argument("--seeds", nargs="+", required=True, type=int, help="seed(s)")
    p.add_argument(
        "--category",
        nargs="+",
        default=["emg_lap"],
        choices=[c.value for c in Category],
        help="category/categories, default: emg_lap (the main mechanism)",
    )
    p.add_argument("--queue-out", type=Path, default=None, help="write a run_queue.sh-format queue file here")
    p.add_argument("--rolling", type=int, default=None, help="use rolling mode with N concurrent (implies --queue-out)")
    p.add_argument("--dry-run", action="store_true", help="print what would be written, write nothing")
    p.add_argument("--run", action="store_true", help="actually invoke run_queue.sh (default: print the command only)")
    p.add_argument("--overwrite", action="store_true", help="allow overwriting an existing config.yaml")
    return p


def main() -> None:
    args = build_parser().parse_args()

    specs = [
        RunSpec(
            category=Category(category),
            body=body,
            reward_variant=RewardVariant(reward_variant),
            seed=seed,
            clip=_LOCKED_CLIP,
            dim_latent=_LOCKED_DIM_LATENT,
            mapped_residual_weight=_LOCKED_MAPPED_RESIDUAL_WEIGHT,
            mirror_left=_LOCKED_MIRROR_LEFT,
        )
        for category in args.category
        for body in args.body
        for reward_variant in args.reward_variant
        for seed in args.seeds
    ]

    generate_and_queue(
        specs,
        queue_out=args.queue_out,
        rolling=args.rolling,
        dry_run=args.dry_run,
        run=args.run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
