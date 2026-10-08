"""RunSpec -> config.yaml dict. See tests/test_config_gen.py."""

from __future__ import annotations

from pathlib import Path

from leaps.configs.naming import derive_tags, derive_tonic_name, derive_wandb_group, derive_wandb_name
from leaps.configs.spec import Category, DecoderSource, ExperimentGroup, RewardVariant, RunSpec

# decoder layout under results/synergy_priors/ for non-released decoders
_DECODER_CATEGORY = {
    DecoderSource.POOLED: "pooled_subjects",
    DecoderSource.AB06: "single_subject",
    DecoderSource.AB20: "single_subject",
    DecoderSource.AB06_CORRECTED: "single_subject",
}

# Reward variants
_ENV_ARGS_BY_VARIANT = {
    RewardVariant.FULL: dict(
        vel_coeff=10,
        grf_coeff=-0.17281,
        joint_limit_coeff=-0.1307,
        nmuscle_coeff=-1.57929,
        smooth_coeff=-0.097,
        self_contact_coeff=0,
    ),
    RewardVariant.ONLYVELREW: dict(
        vel_coeff=1, grf_coeff=0, joint_limit_coeff=0, nmuscle_coeff=0, smooth_coeff=0, self_contact_coeff=0
    ),
    RewardVariant.GAUSSIANVEL: dict(
        vel_coeff=1, grf_coeff=0, joint_limit_coeff=0, nmuscle_coeff=0, smooth_coeff=0, self_contact_coeff=0
    ),
}

# Gym id for registration in sconegym/sconegym/init_v0.py.
_GYM_SUFFIX_BY_VARIANT = {
    RewardVariant.FULL: "",
    RewardVariant.ONLYVELREW: "_onlyVelRew",
    RewardVariant.GAUSSIANVEL: "_gaussianVel",
}

# DEP parameters from the DEP-RL authors' shipped configs for these bodies
# (martius-lab/depRL, experiments/hyfydy/scone_walk_*.yaml)
_DEP_BLOCK_CONSTANTS = dict(
    bias_rate=0.002,
    buffer_size=200,
    intervention_length=8,
    intervention_proba=0.00371,
    normalization="independent",
    q_norm_selector="l2",
    regularization=32,
    s4avg=2,
    sensor_delay=1,
    tau=40,
    test_episode_every=3,
    time_dist=5,
    with_learning=True,
)


def _env_args(spec: RunSpec) -> dict:
    coeffs = _ENV_ARGS_BY_VARIANT[spec.reward_variant]
    return dict(
        clip_actions=spec.clip,
        vel_coeff=coeffs["vel_coeff"],
        grf_coeff=coeffs["grf_coeff"],
        joint_limit_coeff=coeffs["joint_limit_coeff"],
        nmuscle_coeff=coeffs["nmuscle_coeff"],
        smooth_coeff=coeffs["smooth_coeff"],
        self_contact_coeff=coeffs["self_contact_coeff"],
        step_size=0.025,
        run=False,
        init_activations_mean=0.01,
        init_activations_std=0,
    )


def _gym_id(spec: RunSpec) -> str:
    return f"sconewalk_{spec.body}{_GYM_SUFFIX_BY_VARIANT[spec.reward_variant]}-v1"


def _decoder_paths(spec: RunSpec) -> tuple[str, str]:
    suffix = "" if spec.decoder_source.value == "pooled" else f"_{spec.decoder_source.value}"
    decoder_name = f"decoder_k{spec.dim_latent}{suffix}"
    if spec.experiment_group == ExperimentGroup.FINAL_EXPERIMENTS and spec.decoder_source == DecoderSource.AB06_CORRECTED:
        decoder_dir = f"{spec.priors_root}/{decoder_name}"
    elif spec.experiment_group == ExperimentGroup.FINAL_EXPERIMENTS:
        category = _DECODER_CATEGORY[spec.decoder_source]
        decoder_dir = f"{spec.synergy_root}_priors/{category}/{decoder_name}"
    else:
        decoder_dir = f"{spec.synergy_root}/{decoder_name}"
    decoder_path = f"{decoder_dir}/decoder.pt"
    norm_path = f"{decoder_dir}/norm.npz"
    # early_tests configs are history: their decoders may no longer exist
    if spec.experiment_group == ExperimentGroup.FINAL_EXPERIMENTS and (
        not Path(decoder_path).exists() or not Path(norm_path).exists()
    ):
        raise FileNotFoundError(
            f"decoder for k={spec.dim_latent}, source={spec.decoder_source.value} does not exist "
            f"at {decoder_dir}/ - train it first, or pick a different dim_latent/decoder_source."
        )
    return decoder_path, norm_path


def _agent_expr(spec: RunSpec) -> str:
    dep_mix = 3 if spec.is_dep else 0
    steps_before_batches = "2e5" if spec.is_dep else "1e5"
    return (
        f"deprl.custom_agents.dep_factory({dep_mix}, deprl.custom_mpo_torch.TunedMPO())"
        f"(replay=deprl.custom_replay_buffers.AdaptiveEnergyBuffer(return_steps=1, "
        f"batch_size=256, steps_between_batches=1000, batch_iterations=30, "
        f"steps_before_batches={steps_before_batches}, num_acts={spec.num_acts}))"
    )


def _environment_expr(spec: RunSpec) -> str:
    gym_expr = f"deprl.environments.Gym('{_gym_id(spec)}', scaled_actions=False)"
    if not spec.is_lap:
        return gym_expr

    parts = [gym_expr, f"model_name='{spec.body}'"]
    # the DEP prior has no decoder
    if spec.category != Category.DEP_PRIOR:
        decoder_path, norm_path = _decoder_paths(spec)
        parts.append(f"decoder_path='{decoder_path}'")
        parts.append(f"norm_path='{norm_path}'")
    parts.append(f"dim_latent={spec.dim_latent}")
    parts.append(f"residual_weight={spec.residual_weight}")
    parts.append(f"mirror_left={spec.mirror_left}")
    parts.append(f"mapped_residual_weight={spec.mapped_residual_weight}")
    if spec.loosened_actuators:
        parts.append(f"loosened_actuators={list(spec.loosened_actuators)!r}")
        parts.append(f"loosened_residual_weight={spec.loosened_residual_weight}")
    if spec.category == Category.NO_LAP:
        parts.append("null_prior=True")
    elif spec.category == Category.UNTRAINED_LAP:
        parts.append("untrained_decoder=True")
    elif spec.category == Category.DEP_PRIOR:
        parts.append("dep_content=True")
    return "leaps.envs.LatentActionPriorWrapper(" + ", ".join(parts) + ")"


def _dep_block(spec: RunSpec) -> dict | None:
    if not spec.is_dep:
        return None
    return dict(kappa=spec.dep_kappa, **_DEP_BLOCK_CONSTANTS)


def _mpo_args_block(spec: RunSpec) -> dict | None:
    if spec.net_size == 256:
        return None
    return dict(hidden_size=spec.net_size)


def build_config(spec: RunSpec) -> dict:
    """RunSpec -> same dict structure as yaml.safe_dump writes to config.yaml (see _decoder_paths)."""
    header = "import deprl, gym, sconegym" + (", leaps.envs" if spec.is_lap else "")

    cfg: dict = {
        "env_args": _env_args(spec),
        "tonic": {
            "after_training": "",
            "header": header,
            "agent": _agent_expr(spec),
            "before_training": "",
            "checkpoint": "last",
            "environment": _environment_expr(spec),
            "full_save": 1,
            "name": derive_tonic_name(spec),
            "resume": spec.resume,
            "seed": spec.seed,
            "parallel": 20,
            "sequential": 10,
            "test_environment": None,
            "trainer": f"deprl.custom_trainer.Trainer(steps=int({spec.steps}), epoch_steps=int(2e5), save_steps=int(2e6))",
        },
        "wandb": {
            "project": "leaps",
            "name": derive_wandb_name(spec),
            "group": derive_wandb_group(spec),
            "tags": derive_tags(spec),
        },
        "working_dir": "IGNORED_FOR_HYFYDY",
    }

    dep_block = _dep_block(spec)
    if dep_block is not None:
        cfg["DEP"] = dep_block

    mpo_args_block = _mpo_args_block(spec)
    if mpo_args_block is not None:
        cfg["mpo_args"] = mpo_args_block

    return cfg
