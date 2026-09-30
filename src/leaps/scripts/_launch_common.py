"""Shared config-generation logic for launch_locked.py and launch_explore.py.

Not registered in cli.py directly -- it's the core both launcher CLIs call
after parsing their own argparse surface into a list of RunSpecs.
"""

from __future__ import annotations

from pathlib import Path

from leaps.configs import build_config, config_path, write_config, write_queue_file
from leaps.configs.spec import RunSpec


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
