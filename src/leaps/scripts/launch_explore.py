"""Fully general launcher: one flag per RunSpec knob, sweeps the cross
product of any multi-valued flags.

For the locked-in production set (k=6, w=0.5, unclipped) use
launch_locked.py instead, which has a narrower, harder-to-misuse surface.
Use this one for the w/k sweeps, the clip ablation, and the
untrained/null/DEP-content control arms.

Usage:
    # single run:
    python -m leaps.scripts.launch_explore --category emg_lap --body h0918 \\
        --k 6 --w 0.5 --clip false --seeds 0 --dry-run

    # sweep: every combination of k x w x clip, seeds 0-1:
    python -m leaps.scripts.launch_explore --category emg_lap --body h0918 \\
        --k 2 6 11 --w 0.0 0.1 0.5 --clip false true --seeds 0 1 \\
        --queue-out queue_sweep.txt
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from leaps.configs.spec import BODIES, Category, DecoderSource, RewardVariant, RunSpec
from leaps.scripts._launch_common import generate_and_queue


def _bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", nargs="+", required=True, choices=[c.value for c in Category])
    p.add_argument("--body", nargs="+", required=True, choices=sorted(BODIES))
    p.add_argument("--reward-variant", nargs="+", default=["onlyvelrew"], choices=[v.value for v in RewardVariant])
    p.add_argument("--seeds", nargs="+", required=True, type=int)
    p.add_argument("--clip", nargs="+", default=[False], type=_bool)
    p.add_argument("--k", "--dim-latent", dest="k", nargs="+", default=[6], type=int)
    p.add_argument("--w", "--mapped-residual-weight", dest="w", nargs="+", default=[0.5], type=float)
    p.add_argument("--residual-weight", nargs="+", default=[1.0], type=float)
    p.add_argument("--mirror", nargs="+", default=[True], type=_bool)
    p.add_argument("--mirror-mode", nargs="+", default=["static"], choices=["static"])
    p.add_argument(
        "--decoder-source", nargs="+", default=["AB06_corrected"], choices=[s.value for s in DecoderSource]
    )
    p.add_argument("--dep-kappa", nargs="+", default=[1000], type=int)  # verified against authors' shipped configs, see RunSpec.dep_kappa
    p.add_argument("--net-size", nargs="+", default=[256], type=int)
    p.add_argument("--resume", type=_bool, default=True)
    p.add_argument("--queue-out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    combos = itertools.product(
        args.category,
        args.body,
        args.reward_variant,
        args.seeds,
        args.clip,
        args.k,
        args.w,
        args.residual_weight,
        args.mirror,
        args.mirror_mode,
        args.decoder_source,
        args.dep_kappa,
        args.net_size,
    )
    specs = [
        RunSpec(
            category=Category(category),
            body=body,
            reward_variant=RewardVariant(reward_variant),
            seed=seed,
            clip=clip,
            dim_latent=k,
            mapped_residual_weight=w,
            residual_weight=residual_weight,
            mirror_left=mirror,
            mirror_mode=mirror_mode,
            decoder_source=DecoderSource(decoder_source),
            dep_kappa=dep_kappa,
            net_size=net_size,
            resume=args.resume,
        )
        for category, body, reward_variant, seed, clip, k, w, residual_weight, mirror, mirror_mode, decoder_source, dep_kappa, net_size in combos
    ]

    generate_and_queue(
        specs,
        queue_out=args.queue_out,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
