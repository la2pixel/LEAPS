"""Opinionated launcher for the locked-in production experiment set.

k=6, w=0.5, unclipped, mirror on, AB06 decoder (priors/), net256 -- the parameters
locked in 2026-08-15 -- are hardcoded, not flags. Only asks for what
actually varies run to run: which body models, which reward variants,
which seeds, which category. For anything outside this locked set (other
k/w/clip values, control arms) use launch_explore.py
instead, which shares this same generator core.

Usage:
    python -m leaps.scripts.launch_locked --body h0918 h1622 h2190 \\
        --reward-variant onlyvelrew --seeds 0 1 2 --dry-run

    # generate for real, list them in a queue file:
    python -m leaps.scripts.launch_locked --body h0918 --seeds 0 1 \\
        --queue-out queue_locked.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from leaps.configs.spec import BODIES, Category, DecoderSource, RewardVariant, RunSpec
from leaps.scripts._launch_common import generate_and_queue

# The 2026-08-15 lock-in. Not exposed as flags -- that's the point.
_LOCKED_DIM_LATENT = 6
_LOCKED_MAPPED_RESIDUAL_WEIGHT = 0.5
_LOCKED_CLIP = False
_LOCKED_MIRROR_LEFT = True
_LOCKED_DECODER_SOURCE = DecoderSource.AB06_CORRECTED


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
    p.add_argument("--queue-out", type=Path, default=None, help="also list the configs in this file, one per line")
    p.add_argument("--dry-run", action="store_true", help="print what would be written, write nothing")
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
            decoder_source=_LOCKED_DECODER_SOURCE,
        )
        for category in args.category
        for body in args.body
        for reward_variant in args.reward_variant
        for seed in args.seeds
    ]

    generate_and_queue(
        specs,
        queue_out=args.queue_out,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
