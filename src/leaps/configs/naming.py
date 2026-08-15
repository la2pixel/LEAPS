"""Derives run_name/tonic.name/wandb fields from a RunSpec.
"""

from leaps.configs.spec import Category, DecoderSource, RewardVariant, RunSpec

# Based on the reference paper (Humanoid-v4 variant), we fix omega as w=0.5 default. 
# Will report results from constraining it to 0.1 based on already run ablations, we wont revert again

_DEFAULT_MAPPED_RESIDUAL_WEIGHT = 0.5


def _clip_str(spec: RunSpec) -> str:
    return "clip" if spec.clip else "noclip"


def _decoder_suffix(spec: RunSpec) -> str:
    return "" if spec.decoder_source == DecoderSource.POOLED else f"_{spec.decoder_source.value}"


def derive_run_name(spec: RunSpec) -> str:
    """ Logging structure Ex.,
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
    """lalitha/<experiment_group>/<category>/<body>/<run_name> -- must match
    config_path()'s physical layout exactly, since this is what deprl
    actually uses to locate/resume a run's results (not the physical path
    directly). See the 2026-08-15 baselines_DEPRL reorg."""
    return f"lalitha/{spec.experiment_group.value}/{spec.category.value}/{spec.body}/{derive_run_name(spec)}"


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
    """run_name without trailing seed(mainly for wandb groupping)"""
    name = derive_run_name(spec)
    suffix = f"_seed{spec.seed}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def derive_wandb_name(spec: RunSpec) -> str:
    return derive_run_name(spec)
