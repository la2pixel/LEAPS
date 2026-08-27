"""Writes a built config dict to disk, and queue files for run_queue.sh.

write_config won't overwrite an existing config.yaml by default, this is just for new runs after locking in final approach with a (k,w) pair confirmed
"""

from __future__ import annotations

from pathlib import Path

import yaml

from leaps.configs.naming import derive_run_name
from leaps.configs.spec import ExperimentGroup, RunSpec


def config_path(spec: RunSpec) -> Path:
    """Physical layout -- must match naming.derive_tonic_name() exactly,
    see that function's docstring. Non-default-recipe final_experiments
    runs (net512, clip=True, ...) land under an extra /other/ segment,
    2026-08-19, so the main tree stays net256/noclip-only by construction
    and its names never need to spell that out."""
    root = Path(spec.baselines_root) / spec.experiment_group.value
    if spec.experiment_group == ExperimentGroup.FINAL_EXPERIMENTS and not spec.is_default_recipe:
        root = root / "other"
    return root / spec.category.value / spec.body / derive_run_name(spec) / "config.yaml"


def write_config(spec: RunSpec, cfg: dict, *, overwrite: bool = False) -> Path:
    path = config_path(spec)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    return path


_GROUP_DIR_NAMES = {g.value for g in ExperimentGroup}


def _infer_baselines_root(path: Path) -> Path | None:
    """Walk up from a config.yaml to the <group> dir (final_experiments/
    early_tests) and return its parent. Depth-independent on purpose --
    2026-08-19's /other/ segment for non-default-recipe final_experiments
    runs makes a fixed parents[N] index wrong for exactly those paths."""
    for parent in path.parents:
        if parent.name in _GROUP_DIR_NAMES:
            return parent.parent
    return None


def write_queue_file(paths: list[Path], out_path: Path, *, batches: list[list[Path]] | None = None) -> Path:
    """run_queue.sh queue-file format: one config.yaml path per line,
    blank line separates batches (batch mode) or is just ignored (rolling
    mode). Paths written relative to run_queue.sh's $BASELINES when possible"""
    baselines_root = None
    lines: list[str] = []

    def _rel(p: Path) -> str:
        nonlocal baselines_root
        try:
            return str(p.relative_to(baselines_root)) if baselines_root else str(p)
        except ValueError:
            return str(p)

    if paths:
        baselines_root = _infer_baselines_root(paths[0])

    if batches:
        for i, batch in enumerate(batches):
            if i > 0:
                lines.append("")
            lines.extend(_rel(p) for p in batch)
    else:
        lines.extend(_rel(p) for p in paths)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
