"""Derives run_name/tonic.name/wandb fields from a RunSpec.
"""

from leaps.configs.spec import Category, DecoderSource, ExperimentGroup, RewardVariant, RunSpec

# Based on the reference paper (Humanoid-v4 variant), we fix omega as w=0.5 default. 
# Will report results from constraining it to 0.1 based on already run ablations, we wont revert again

_DEFAULT_MAPPED_RESIDUAL_WEIGHT = 0.5


def _clip_str(spec: RunSpec) -> str:
    return "clip" if spec.clip else "noclip"


def _decoder_suffix(spec: RunSpec) -> str:
    return "" if spec.decoder_source == DecoderSource.POOLED else f"_{spec.decoder_source.value}"


def derive_run_name(spec: RunSpec) -> str:
    """ Logging structure Ex.,
    h0918_k11_w01_mirror_noclip_net256_onlyvelrew_AB06_seed0 (LAP, outside
    final_experiments) or h2190_clip_net256_onlyvelrew_seed1 (baseline,
    outside final_experiments). Within final_experiments, clip/net{size}
    are omitted when they match that group's only populated recipe
    (noclip, net256, checked via spec.is_default_recipe) -- every one of
    its 55 current configs is that recipe, so spelling it out on every
    single name was pure noise. A non-default recipe there (net512,
    clip=True, ...) keeps the tokens AND gets physically routed to
    final_experiments/other/ by writer.config_path()/derive_tonic_name(),
    so the main tree's names never need to carry these tokens at all."""
    final_experiments = spec.experiment_group == ExperimentGroup.FINAL_EXPERIMENTS
    parts = [spec.body]
    if spec.is_lap:
        parts.append(f"k{spec.dim_latent}")
        parts.append(f"w{int(round(spec.mapped_residual_weight * 10)):02d}")
        if spec.loosened_actuators:
            short = "".join(sorted({n.removesuffix("_r").removesuffix("_l") for n in spec.loosened_actuators}))
            parts.append(f"loosen{short}{int(round(spec.loosened_residual_weight * 10)):02d}")
        # mirror_left/mirror_mode are inert for DEP_PRIOR (no
        # EMGToMuscleMapper/PhaseMirror involved, see dep_content branch in
        # latent_env.py) -- "mirror" in the name would misleadingly imply
        # gait-mirroring logic that never runs for this category.
        if spec.category != Category.DEP_PRIOR:
            # mirror_left=True is the default and, as of 2026-08-20, every
            # final_experiments config uses it -- same zero-information-token
            # situation the 2026-08-19 fix already resolved for clip/net_size,
            # just missed for this one. Same final_experiments-only scoping:
            # early_tests/ is frozen history (golden-file tests assert
            # byte-for-byte fidelity to real files that predate this fix) and
            # keeps the old unconditional "mirror" tag; final_experiments
            # only tags the non-default (mirror_left=False) deviation.
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
    # early_tests/ is frozen history (2026-08-15 reorg) and its golden-file
    # tests assert byte-for-byte fidelity to real files that never had this
    # marker, so only apply it going forward. Needed once final_experiments
    # started deliberately generating matched no_lap/untrained_lap/emg_lap
    # configs at identical k/w/net/mirror/clip/seed for controlled
    # comparisons -- without it those categories produce identical
    # run_names, and run_queue.sh's job_name_for() (basename of the
    # config's directory) collides across them, silently treating the
    # second job as "already running" and never launching it.
    if final_experiments:
        if spec.category == Category.NO_LAP:
            parts.append("null")
        elif spec.category == Category.UNTRAINED_LAP:
            parts.append("untrainedprior")
        elif spec.category == Category.DEP_PRIOR:
            parts.append("depcontent")
        elif spec.category == Category.DEP_MPO:
            # neither MPO nor DEP_MPO is is_lap, so nothing above
            # distinguishes them -- same body/clip/net/reward/seed collides
            # on job name, and run_queue.sh silently treats the second one
            # as already running. Hit exactly this 2026-08-16: 3 queued mpo
            # net512 jobs never launched because same-named dep-mpo net512
            # jobs were already running under the un-tagged name.
            parts.append("dep")
        # "static" disambiguator dropped 2026-08-19: mirror_mode can only be
        # "static" now (phase mirroring removed 2026-08-18, see
        # latent_env.py), so it was the exact same invariant-token noise as
        # net256/noclip above -- nothing left to disambiguate from.
    parts.append(f"seed{spec.seed}")
    return "_".join(parts)


def derive_tonic_name(spec: RunSpec) -> str:
    """lalitha/<experiment_group>[/other]/<category>/<body>/<run_name> --
    must match config_path()'s physical layout exactly, since this is what
    deprl actually uses to locate/resume a run's results (not the physical
    path directly). See the 2026-08-15 baselines_DEPRL reorg. The optional
    /other/ segment appears iff spec.is_default_recipe is False within
    final_experiments (2026-08-19) -- net512/clip=True/etc ablations are
    physically segregated from the main net256/noclip tree there, not just
    differently named."""
    group = spec.experiment_group.value
    if spec.experiment_group == ExperimentGroup.FINAL_EXPERIMENTS and not spec.is_default_recipe:
        group = f"{group}/other"
    return f"lalitha/{group}/{spec.category.value}/{spec.body}/{derive_run_name(spec)}"


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
