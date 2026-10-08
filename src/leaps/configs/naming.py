"""Derives run_name / tonic.name / wandb fields from a RunSpec."""

from leaps.configs.spec import Category, DecoderSource, ExperimentGroup, RewardVariant, RunSpec

# Based on the reference paper (Humanoid-v4 variant), we fix omega as w=0.5 default. 
# w = 0.5 is the default, as in the reference paper
_DEFAULT_MAPPED_RESIDUAL_WEIGHT = 0.5


def _clip_str(spec: RunSpec) -> str:
    return "clip" if spec.clip else "noclip"


def _decoder_suffix(spec: RunSpec) -> str:
    return "" if spec.decoder_source == DecoderSource.POOLED else f"_{spec.decoder_source.value}"


def derive_run_name(spec: RunSpec) -> str:
    """e.g. h0918_k11_w01_mirror_noclip_net256_onlyvelrew_AB06_seed0.
    In final_experiments the clip/net tokens are dropped for the default
    recipe (noclip, net256); other recipes keep them and go to other/."""
    final_experiments = spec.experiment_group == ExperimentGroup.FINAL_EXPERIMENTS
    parts = [spec.body]
    if spec.is_lap:
        parts.append(f"k{spec.dim_latent}")
        parts.append(f"w{int(round(spec.mapped_residual_weight * 10)):02d}")
        if spec.loosened_actuators:
            short = "".join(sorted({n.removesuffix("_r").removesuffix("_l") for n in spec.loosened_actuators}))
            parts.append(f"loosen{short}{int(round(spec.loosened_residual_weight * 10)):02d}")
        # no mirroring with the DEP prior
        if spec.category != Category.DEP_PRIOR:
            # final_experiments only tags the non-default mirror_left=False
            if not final_experiments:
                if spec.mirror_left:
                    parts.append("mirror")
            elif not spec.mirror_left:
                parts.append("nomirror")
    if not final_experiments or spec.clip:
        parts.append(_clip_str(spec))
    if not final_experiments or spec.net_size != 256:
        parts.append(f"net{spec.net_size}")
    parts.append(spec.reward_variant.value)
    if spec.is_lap:
        suffix = _decoder_suffix(spec)
        if suffix:
            parts.append(suffix.lstrip("_"))
    # category marker: matched no_lap / untrained_lap / emg_lap configs would
    # otherwise get identical run names (early_tests names stay as they were)
    if final_experiments:
        if spec.category == Category.NO_LAP:
            parts.append("null")
        elif spec.category == Category.UNTRAINED_LAP:
            parts.append("untrainedprior")
        elif spec.category == Category.DEP_PRIOR:
            parts.append("depcontent")
        elif spec.category == Category.DEP_MPO:
            # keeps MPO and DEP-MPO run names apart
            parts.append("dep")
    parts.append(f"seed{spec.seed}")
    return "_".join(parts)


def derive_tonic_name(spec: RunSpec) -> str:
    """lalitha/<experiment_group>[/other]/<category>/<body>/<run_name>.
    Must match config_path(): deprl locates and resumes runs by this name."""
    group = spec.experiment_group.value
    if spec.experiment_group == ExperimentGroup.FINAL_EXPERIMENTS and not spec.is_default_recipe:
        group = f"{group}/other"
    return f"lalitha/{group}/{spec.category.value}/{spec.body}/{derive_run_name(spec)}"


def derive_tags(spec: RunSpec) -> list[str]:
    """Tags a knob only when it differs from its default."""
    tags: list[str] = []
    tags.append("hardconstraint" if spec.is_lap else spec.category.value)
    tags.append(spec.body)

    if spec.is_lap:
        tags.append(f"k{spec.dim_latent}")
        if spec.mapped_residual_weight != _DEFAULT_MAPPED_RESIDUAL_WEIGHT:
            tags.append(f"w{int(round(spec.mapped_residual_weight * 10)):02d}")
        if not spec.mirror_left and spec.category != Category.DEP_PRIOR:
            tags.append("nomirror")
        if spec.loosened_actuators:
            tags.append("loosened_residual_weight")

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
    if spec.category == Category.DEP_PRIOR:
        tags.append("dep_prior")

    tags.append(f"seed{spec.seed}")
    return tags


def derive_wandb_group(spec: RunSpec) -> str:
    """run_name without trailing seed(mainly for wandb groupping)"""
    name = derive_run_name(spec)
    suffix = f"_seed{spec.seed}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def derive_wandb_name(spec: RunSpec) -> str:
    return derive_run_name(spec)
