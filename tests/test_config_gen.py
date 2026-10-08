"""Golden-file tests for leaps.configs: the builder must reproduce
already-launched config.yaml files exactly."""

from pathlib import Path

import pytest
import yaml

from leaps.configs.builder import _decoder_paths, build_config
from leaps.configs.spec import Category, DecoderSource, ExperimentGroup, RewardVariant, RunSpec
from leaps.configs.writer import config_path, write_config
from leaps.paths import RUNS_DIR

BASELINES_ROOT = RUNS_DIR

# (RunSpec kwargs, path of the config.yaml it should match), all under early_tests/
_EARLY = dict(experiment_group=ExperimentGroup.EARLY_TESTS)
CASES = [
    (
        dict(_EARLY, category=Category.MPO, body="h2190", reward_variant=RewardVariant.ONLYVELREW, seed=1, clip=True),
        "early_tests/mpo/h2190/h2190_clip_net256_onlyvelrew_seed1/config.yaml",
    ),
    (
        dict(_EARLY, category=Category.DEP_MPO, body="h0918", reward_variant=RewardVariant.ONLYVELREW, seed=0, clip=True),
        "early_tests/dep-mpo/h0918/h0918_clip_net256_onlyvelrew_seed0/config.yaml",
    ),
    (
        dict(
            _EARLY,
            category=Category.EMG_LAP,
            body="h0918",
            reward_variant=RewardVariant.ONLYVELREW,
            seed=0,
            clip=True,
            dim_latent=11,
            mapped_residual_weight=0.1,
        ),
        "early_tests/emg_lap/h0918/h0918_k11_w01_mirror_clip_net256_onlyvelrew_seed0/config.yaml",
    ),
    (
        dict(
            _EARLY,
            category=Category.EMG_LAP,
            body="h0918",
            reward_variant=RewardVariant.ONLYVELREW,
            seed=0,
            clip=False,
            dim_latent=11,
            mapped_residual_weight=0.1,
            decoder_source=DecoderSource.AB06,
        ),
        "early_tests/emg_lap/h0918/h0918_k11_w01_mirror_noclip_net256_onlyvelrew_AB06_seed0/config.yaml",
    ),
    (
        dict(
            _EARLY,
            category=Category.NO_LAP,
            body="h2190",
            reward_variant=RewardVariant.ONLYVELREW,
            seed=0,
            clip=True,
            dim_latent=6,
            mapped_residual_weight=0.5,
        ),
        "early_tests/no_lap/h2190/h2190_k6_w05_mirror_clip_net256_onlyvelrew_seed0/config.yaml",
    ),
    (
        dict(
            _EARLY,
            category=Category.UNTRAINED_LAP,
            body="h1622",
            reward_variant=RewardVariant.ONLYVELREW,
            seed=0,
            clip=False,
            dim_latent=6,
            mapped_residual_weight=0.1,
        ),
        "early_tests/untrained_lap/h1622/h1622_k6_w01_mirror_noclip_net256_onlyvelrew_seed0/config.yaml",
    ),
]


@pytest.mark.parametrize("spec_kwargs,real_path", CASES, ids=[c[1] for c in CASES])
def test_builder_reproduces_real_config(spec_kwargs, real_path):
    if not (BASELINES_ROOT / real_path).exists():
        pytest.skip("golden config only exists in the original run tree")
    generated = build_config(RunSpec(**spec_kwargs))
    real = yaml.safe_load((BASELINES_ROOT / real_path).read_text())
    # the wandb block is not compared: its conventions differ across old configs
    for key in ("env_args", "tonic", "DEP", "mpo_args"):
        if key in real or key in generated:
            assert generated.get(key) == real.get(key), f"{key} mismatch for {real_path}"
    assert generated["working_dir"] == real["working_dir"]


def test_decoder_validation_raises_for_missing_k6_ab20():
    """A decoder that does not exist on disk must fail, not write a broken config."""
    spec = RunSpec(category=Category.EMG_LAP, body="h0918", dim_latent=6, decoder_source=DecoderSource.AB20)
    with pytest.raises(FileNotFoundError, match="decoder_k6_AB20"):
        build_config(spec)


def test_decoder_paths_released_priors_exist():
    """The released priors/ decoders resolve."""
    for k in (2, 4, 6, 8, 11):
        spec = RunSpec(category=Category.EMG_LAP, body="h0918", dim_latent=k, decoder_source=DecoderSource.AB06_CORRECTED)
        decoder_path, norm_path = _decoder_paths(spec)
        assert Path(decoder_path).exists()
        assert Path(norm_path).exists()


def test_deprecated_precorrected_decoders_not_resolved_by_final_experiments():
    """final_experiments must not resolve to a deprecated decoder."""
    spec = RunSpec(category=Category.EMG_LAP, body="h0918", dim_latent=11, decoder_source=DecoderSource.POOLED)
    with pytest.raises(FileNotFoundError):
        _decoder_paths(spec)


@pytest.mark.parametrize(
    "spec_kwargs,real_dir",
    [(c[0], c[1].rsplit("/", 2)[-2]) for c in CASES],
    ids=[c[1] for c in CASES],
)
def test_run_name_matches_real_directory_convention(spec_kwargs, real_dir):
    from leaps.configs.naming import derive_run_name

    assert derive_run_name(RunSpec(**spec_kwargs)) == real_dir


def test_never_overwrites_existing_config(tmp_path):
    spec = RunSpec(category=Category.MPO, body="h0918", baselines_root=str(tmp_path))
    cfg = build_config(spec)
    path = write_config(spec, cfg)
    assert path.exists()

    with pytest.raises(FileExistsError):
        write_config(spec, cfg)

    # overwrite=True is the explicit escape hatch:
    write_config(spec, cfg, overwrite=True)


def test_w_tag_inversion():
    """w=0.5 is the default and untagged; w=0.1 is tagged."""
    from leaps.configs.naming import derive_tags

    default_w = RunSpec(category=Category.EMG_LAP, body="h0918", mapped_residual_weight=0.5)
    other_w = RunSpec(category=Category.EMG_LAP, body="h0918", mapped_residual_weight=0.1)
    assert not any(t.startswith("w0") for t in derive_tags(default_w))
    assert "w01" in derive_tags(other_w)


def test_dep_prior_omits_decoder_and_sets_dep_content():
    """The DEP prior never touches a decoder, so none is required on disk."""
    spec = RunSpec(category=Category.DEP_PRIOR, body="h0918", reward_variant=RewardVariant.ONLYVELREW, clip=False)
    env_expr = build_config(spec)["tonic"]["environment"]
    assert "decoder_path" not in env_expr
    assert "norm_path" not in env_expr
    assert "dep_content=True" in env_expr
    # mirror_mode is meaningless for the DEP prior
    assert "mirror_mode" not in env_expr


def test_config_path_layout():
    spec = RunSpec(
        category=Category.EMG_LAP,
        body="h0918",
        reward_variant=RewardVariant.ONLYVELREW,
        dim_latent=6,
        mapped_residual_weight=0.5,
        clip=False,
        seed=2,
    )
    path = config_path(spec)
    # default-recipe tokens (noclip, net256, mirror) are omitted in final_experiments
    assert path == (
        BASELINES_ROOT / "final_experiments" / "emg_lap" / "h0918"
        / "h0918_k6_w05_onlyvelrew_seed2" / "config.yaml"
    )


def test_config_path_routes_non_default_recipe_to_other():
    """Non-default recipes go under final_experiments/other/, and
    derive_tonic_name() matches that path."""
    from leaps.configs.naming import derive_tonic_name

    spec = RunSpec(category=Category.MPO, body="h0918", reward_variant=RewardVariant.ONLYVELREW, net_size=512, seed=0)
    assert not spec.is_default_recipe
    path = config_path(spec)
    assert path == (
        BASELINES_ROOT / "final_experiments" / "other" / "mpo" / "h0918"
        / "h0918_net512_onlyvelrew_seed0" / "config.yaml"
    )
    assert derive_tonic_name(spec) == "lalitha/final_experiments/other/mpo/h0918/h0918_net512_onlyvelrew_seed0"

    clip_spec = RunSpec(category=Category.MPO, body="h0918", clip=True, seed=0)
    assert not clip_spec.is_default_recipe
    assert "other" in config_path(clip_spec).parts

    default_spec = RunSpec(category=Category.MPO, body="h0918", seed=0)
    assert default_spec.is_default_recipe
    assert "other" not in config_path(default_spec).parts
