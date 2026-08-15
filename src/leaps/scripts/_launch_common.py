"""Shared generate-then-queue logic for launch_locked.py and launch_explore.py.

Not registered in cli.py directly -- it's the core both launcher CLIs call
after parsing their own argparse surface into a list of RunSpecs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from leaps.configs import build_config, config_path, write_config, write_queue_file
from leaps.configs.spec import RunSpec

RUN_QUEUE_SH = "/home/nadinebadie/lalitha/run_scripts/run_queue.sh"


def generate_and_queue(
    specs: list[RunSpec],
    *,
    queue_out: Path | None,
    rolling: int | None,
    dry_run: bool,
    run: bool,
    overwrite: bool,
) -> list[Path]:
    """validate -> build -> write -> collect paths -> optionally write a
    queue file -> print (or, with run=True, execute) the run_queue.sh
    command. Never invokes run_queue.sh unless run=True is explicit --
    config generation should be safe to iterate on before committing to a
    multi-day job."""
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

    if queue_out is None:
        return paths

    if not dry_run:
        write_queue_file(paths, queue_out)
        print(f"queue file: {queue_out}")
    else:
        print(f"[dry-run] would write queue file: {queue_out}")

    cmd = [RUN_QUEUE_SH]
    if rolling is not None:
        cmd += ["--rolling", str(queue_out), str(rolling)]
    else:
        cmd += [str(queue_out)]

    if run and not dry_run:
        print(f"launching: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        print(f"to launch: {' '.join(cmd)}")

    return paths
