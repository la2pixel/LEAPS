"""Derives run_name / tonic.name / wandb fields from a RunSpec."""

from leaps.configs.spec import Category, RewardVariant, RunSpec

# w = 0.5 is the default, as in the reference paper
_DEFAULT_MAPPED_RESIDUAL_WEIGHT = 0.5


def derive_run_name(spec: RunSpec) -> str:
    """e.g. h0918_k6_w01_onlyvelrew_seed0. Clip/net tokens only appear for
    non-default recipes, which go to other/."""
    parts = [spec.body]
    if spec.is_lap:
        parts.append(f"k{spec.dim_latent}")
        parts.append(f"w{int(round(spec.mapped_residual_weight * 10)):02d}")
        if spec.loosened_actuators:
            short = "".join(sorted({n.removesuffix("_r").removesuffix("_l") for n in spec.loosened_actuators}))
            parts.append(f"loosen{short}{int(round(spec.loosened_residual_weight * 10)):02d}")
        if not spec.mirror_left and spec.category != Category.DEP_PRIOR:
            parts.append("nomirror")
    if spec.clip:
        parts.append("clip")
    if spec.net_size != 256:
        parts.append(f"net{spec.net_size}")
    parts.append(spec.reward_variant.value)
    # matched emg_lap / no_lap / untrained_lap (and mpo / dep-mpo) runs need distinct names
    marker = {
        Category.NO_LAP: "null",
        Category.UNTRAINED_LAP: "untrainedprior",
        Category.DEP_PRIOR: "depcontent",
        Category.DEP_MPO: "dep",
    }.get(spec.category)
    if marker:
        parts.append(marker)
    parts.append(f"seed{spec.seed}")
    return "_".join(parts)


def derive_tonic_name(spec: RunSpec) -> str:
    """lalitha/final_experiments[/other]/<category>/<body>/<run_name>.
    Must match config_path(): deprl locates and resumes runs by this name."""
    group = "final_experiments" if spec.is_default_recipe else "final_experiments/other"
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
