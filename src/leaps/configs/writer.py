"""Writes a built config dict to disk, and queue files.

write_config does not overwrite an existing config.yaml unless overwrite=True.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from leaps.configs.naming import derive_run_name
from leaps.configs.spec import RunSpec


def config_path(spec: RunSpec) -> Path:
    """Must match naming.derive_tonic_name(). Non-default recipes in
    final_experiments go under other/."""
    root = Path(spec.baselines_root) / "final_experiments"
    if not spec.is_default_recipe:
        root = root / "other"
    return root / spec.category.value / spec.body / derive_run_name(spec) / "config.yaml"


def write_config(spec: RunSpec, cfg: dict, *, overwrite: bool = False) -> Path:
    path = config_path(spec)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    return path


def _infer_baselines_root(path: Path) -> Path | None:
    """Parent of the final_experiments dir above a config.yaml."""
    for parent in path.parents:
        if parent.name == "final_experiments":
            return parent.parent
    return None


def write_queue_file(paths: list[Path], out_path: Path, *, batches: list[list[Path]] | None = None) -> Path:
    """One config.yaml path per line, relative to baselines_root where possible;
    a blank line separates batches."""
    baselines_root = None
    lines: list[str] = []

    def _rel(p: Path) -> str:
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
