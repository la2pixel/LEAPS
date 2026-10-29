"""Writes deprl config.yaml files for the cross product of the given flags.
Defaults are the main setting (k=6, w=0.5, unclipped, mirrored, net256).

    leaps launch --category emg_lap --body h0918 --w 0.1 --seeds 0 1 2 --dry-run
    leaps launch --category mpo dep-mpo --body h1622 --seeds 0 1 2 --queue-out queue.txt
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from leaps.configs import build_config, config_path, write_config, write_queue_file
from leaps.configs.spec import BODIES, Category, RewardVariant, RunSpec


def generate_and_queue(
    specs: list[RunSpec],
    *,
    queue_out: Path | None,
    dry_run: bool,
    overwrite: bool,
) -> list[Path]:
    """validate -> build -> write configs, optionally list them in a queue
    file (one config.yaml per line), then print the training commands."""
    if not specs:
        print("no run specs to generate (check your filters)")
        return []

    paths: list[Path] = []
    for spec in specs:
        cfg = build_config(spec)  # raises FileNotFoundError here if a decoder is missing
        path = config_path(spec)
        if dry_run:
            print(f"[dry-run] {path}")
        else:
            written = write_config(spec, cfg, overwrite=overwrite)
            print(f"wrote {written}")
        paths.append(path)

    print(f"\n{len(paths)} config(s) {'would be written' if dry_run else 'written'}.")

    if queue_out is not None:
        if not dry_run:
            write_queue_file(paths, queue_out)
        print(f"{'[dry-run] would write' if dry_run else 'wrote'} queue file: {queue_out}")

    print("\nto train:")
    for path in paths:
        print(f"  python -m deprl.main {path}")
    return paths


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
    p.add_argument("--dep-kappa", nargs="+", default=[1000], type=int)
    p.add_argument("--net-size", nargs="+", default=[256], type=int)
    p.add_argument("--resume", type=_bool, default=True)
    p.add_argument("--steps", default="2e7", help="training budget in environment steps")
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
            dep_kappa=dep_kappa,
            net_size=net_size,
            resume=args.resume,
            steps=args.steps,
        )
        for category, body, reward_variant, seed, clip, k, w, residual_weight, mirror, dep_kappa, net_size in combos
    ]

    generate_and_queue(
        specs,
        queue_out=args.queue_out,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
