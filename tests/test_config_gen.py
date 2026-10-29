from pathlib import Path

import pytest

from leaps.configs.builder import _decoder_paths, build_config
from leaps.configs.naming import derive_run_name
from leaps.configs.spec import Category, RewardVariant, RunSpec
from leaps.configs.writer import config_path, write_config
from leaps.paths import RUNS_DIR

BASELINES_ROOT = RUNS_DIR


def test_released_decoders_resolve():
    for k in (2, 4, 6, 8, 11):
        decoder_path, norm_path = _decoder_paths(RunSpec(category=Category.EMG_LAP, body="h0918", dim_latent=k))
        assert Path(decoder_path).exists()
        assert Path(norm_path).exists()


def test_missing_decoder_raises():
    with pytest.raises(FileNotFoundError, match="decoder_k3_AB06"):
        build_config(RunSpec(category=Category.EMG_LAP, body="h0918", dim_latent=3))


@pytest.mark.parametrize(
    "kwargs,name",
    [
        (dict(category=Category.EMG_LAP, mapped_residual_weight=0.1), "h0918_k6_w01_onlyvelrew_seed0"),
        (dict(category=Category.NO_LAP), "h0918_k6_w05_onlyvelrew_null_seed0"),
        (dict(category=Category.UNTRAINED_LAP), "h0918_k6_w05_onlyvelrew_untrainedprior_seed0"),
        (dict(category=Category.MPO), "h0918_onlyvelrew_seed0"),
        (dict(category=Category.DEP_MPO), "h0918_onlyvelrew_dep_seed0"),
        (dict(category=Category.MPO, clip=True, net_size=512), "h0918_clip_net512_onlyvelrew_seed0"),
    ],
)
def test_run_names(kwargs, name):
    spec = RunSpec(body="h0918", reward_variant=RewardVariant.ONLYVELREW, **kwargs)
    assert derive_run_name(spec) == name


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
