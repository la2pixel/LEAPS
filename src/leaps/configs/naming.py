"""Derives run_name/tonic.name/wandb fields from a RunSpec.

Pure string functions, no I/O. Kept separate from builder.py because this
is the part most likely to keep getting hand-tuned as conventions settle --
independently testable against the real naming/tagging inconsistency in
the 217 existing configs without touching the YAML-shape logic.
"""

from leaps.configs.spec import Category, DecoderSource, RewardVariant, RunSpec

# 2026-08-15: locked-in default flipped from w=0.1 to w=0.5. Tag whichever
# value is NOT the current default, so "untagged" always means "current
# default" in wandb filters instead of a frozen historical value.
_DEFAULT_MAPPED_RESIDUAL_WEIGHT = 0.5


def _clip_str(spec: RunSpec) -> str:
    return "clip" if spec.clip else "noclip"


def _decoder_suffix(spec: RunSpec) -> str:
    return "" if spec.decoder_source == DecoderSource.POOLED else f"_{spec.decoder_source.value}"


def derive_run_name(spec: RunSpec) -> str:
    """Matches the real baseline/LAP directory-naming conventions exactly:
    baselines omit k/w/mirror (they have no LatentActionPriorWrapper),
    LAP runs include them. E.g.
    h0918_k11_w01_mirror_noclip_net256_onlyvelrew_AB06_seed0 (LAP) or
    h2190_clip_net256_onlyvelrew_seed1 (baseline)."""
    parts = [spec.body]
    if spec.is_lap:
        parts.append(f"k{spec.dim_latent}")
        parts.append(f"w{int(round(spec.mapped_residual_weight * 10)):02d}")
        if spec.mirror_left:
            parts.append("mirror")
    parts.append(_clip_str(spec))
    parts.append(f"net{spec.net_size}")
    parts.append(spec.reward_variant.value)
    if spec.is_lap:
        suffix = _decoder_suffix(spec)
        if suffix:
            parts.append(suffix.lstrip("_"))
    parts.append(f"seed{spec.seed}")
    return "_".join(parts)


def derive_tonic_name(spec: RunSpec) -> str:
    """lalitha/<category>/<body>/<run_name> -- the tonic.name convention
    every launch in this project uses, and what main.py's lalitha/-prefix
    warning checks for."""
    return f"lalitha/{spec.category.value}/{spec.body}/{derive_run_name(spec)}"


def derive_tags(spec: RunSpec) -> list[str]:
    """Deterministic rule, not a copied template -- the existing 217
    configs disagree with each other on this (61/217 missing tags
    entirely), so there's no single real file to match exactly. Tags a
    knob only when it's away from its current default, so the tag list
    stays informative rather than repeating every knob on every run."""
    tags: list[str] = []
    tags.append("hardconstraint" if spec.is_lap else spec.category.value)
    tags.append(spec.body)

    if spec.is_lap:
        tags.append(f"k{spec.dim_latent}")
        if spec.mapped_residual_weight != _DEFAULT_MAPPED_RESIDUAL_WEIGHT:
            tags.append(f"w{int(round(spec.mapped_residual_weight * 10)):02d}")
        if spec.mirror_left:
            tags.append("mirror")

    if not spec.clip:
        tags.append("noclip")
    if spec.reward_variant != RewardVariant.FULL:
        tags.append(spec.reward_variant.value)
    if spec.is_lap and spec.decoder_source != DecoderSource.POOLED:
        tags.extend(["single_subject", spec.decoder_source.value])
    if spec.category == Category.NO_LAP:
        tags.append("null_prior")
    if spec.category == Category.UNTRAINED_LAP:
        tags.append("untrained_decoder")

    tags.append(f"seed{spec.seed}")
    return tags


def derive_wandb_group(spec: RunSpec) -> str:
    """run_name with the trailing seed stripped -- groups every seed of the
    same run together in the wandb UI, the standard wandb grouping idiom."""
    name = derive_run_name(spec)
    suffix = f"_seed{spec.seed}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def derive_wandb_name(spec: RunSpec) -> str:
    return derive_run_name(spec)
