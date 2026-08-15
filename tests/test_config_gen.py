"""Golden-file regression tests for leaps.configs: the builder must keep
reproducing real, already-launched config.yaml files exactly. Golden cases
are deliberately picked from configs with the newer, non-drifted 10-key
env_args (see builder.py's module docstring) -- some older emg_lap/no_lap
"full" configs are missing self_contact_coeff/run, which the generator
intentionally does not reproduce.
"""

from pathlib import Path

import pytest
import yaml

from leaps.configs.builder import _decoder_paths, build_config
from leaps.configs.spec import Category, DecoderSource, RewardVariant, RunSpec
from leaps.configs.writer import config_path, write_config

BASELINES_ROOT = Path("/home/nadinebadie/lalitha/LEAPS/baselines_DEPRL")

# (RunSpec kwargs, path to a real, non-drifted config.yaml it should match)
CASES = [
    (
        dict(category=Category.MPO, body="h2190", reward_variant=RewardVariant.ONLYVELREW, seed=1, clip=True),
        "mpo/h2190/h2190_clip_net256_onlyvelrew_seed1/config.yaml",
    ),
    (
        dict(category=Category.DEP_MPO, body="h2190", reward_variant=RewardVariant.ONLYVELREW, seed=1, clip=True),
        "dep-mpo/h2190/h2190_clip_net256_onlyvelrew_seed1/config.yaml",
    ),
    (
        dict(
            category=Category.EMG_LAP,
            body="h0918",
            reward_variant=RewardVariant.ONLYVELREW,
            seed=0,
            clip=True,
            dim_latent=11,
            mapped_residual_weight=0.1,
        ),
        "emg_lap/h0918/h0918_k11_w01_mirror_clip_net256_onlyvelrew_seed0/config.yaml",
    ),
    (
        dict(
            category=Category.EMG_LAP,
            body="h0918",
            reward_variant=RewardVariant.ONLYVELREW,
            seed=0,
            clip=False,
            dim_latent=11,
            mapped_residual_weight=0.1,
            decoder_source=DecoderSource.AB06,
        ),
        "emg_lap/h0918/h0918_k11_w01_mirror_noclip_net256_onlyvelrew_AB06_seed0/config.yaml",
    ),
    (
        dict(
            category=Category.NO_LAP,
            body="h2190",
            reward_variant=RewardVariant.ONLYVELREW,
            seed=0,
            clip=True,
            dim_latent=6,
            mapped_residual_weight=0.5,
        ),
        "no_lap/h2190/h2190_k6_w05_mirror_clip_net256_onlyvelrew_seed0/config.yaml",
    ),
    (
        dict(
            category=Category.UNTRAINED_LAP,
            body="h1622",
            reward_variant=RewardVariant.ONLYVELREW,
            seed=0,
            clip=False,
            dim_latent=6,
            mapped_residual_weight=0.1,
        ),
        "untrained_lap/h1622/h1622_k6_w01_mirror_noclip_net256_onlyvelrew_seed0/config.yaml",
    ),
]


@pytest.mark.parametrize("spec_kwargs,real_path", CASES, ids=[c[1] for c in CASES])
def test_builder_reproduces_real_config(spec_kwargs, real_path):
    generated = build_config(RunSpec(**spec_kwargs))
    real = yaml.safe_load((BASELINES_ROOT / real_path).read_text())
    # wandb block is excluded: name/group/tags conventions were already
    # inconsistent across the real 217 configs before this generator
    # existed (61/217 missing tags entirely), and this generator
    # deliberately establishes a new, internally consistent convention
    # rather than reproducing that drift -- see naming.py.
    for key in ("env_args", "tonic", "DEP", "mpo_args"):
        if key in real or key in generated:
            assert generated.get(key) == real.get(key), f"{key} mismatch for {real_path}"
    assert generated["working_dir"] == real["working_dir"]


def test_decoder_validation_raises_for_missing_k6_ab20():
    """decoder_k6_AB20 does not exist on disk (confirmed 2026-08-15: only
    decoder_k6, decoder_k6_AB06 exist at k=6; AB20 only exists at k=11).
    The builder must fail loudly here, not write a broken config."""
    spec = RunSpec(category=Category.EMG_LAP, body="h0918", dim_latent=6, decoder_source=DecoderSource.AB20)
    with pytest.raises(FileNotFoundError, match="decoder_k6_AB20"):
        build_config(spec)


def test_decoder_paths_pooled_k11_exist():
    """Sanity check the validator's happy path against a decoder that's
    confirmed to exist, so the AB20 test above is known to be testing the
    missing-file branch and not some unrelated bug."""
    spec = RunSpec(category=Category.EMG_LAP, body="h0918", dim_latent=11)
    decoder_path, norm_path = _decoder_paths(spec)
    assert Path(decoder_path).exists()
    assert Path(norm_path).exists()


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
    """2026-08-15: w=0.5 is now the locked default and goes untagged;
    w=0.1 (the old default) now gets tagged, inverted from the legacy
    convention. See naming.py's _DEFAULT_MAPPED_RESIDUAL_WEIGHT."""
    from leaps.configs.naming import derive_tags

    default_w = RunSpec(category=Category.EMG_LAP, body="h0918", mapped_residual_weight=0.5)
    other_w = RunSpec(category=Category.EMG_LAP, body="h0918", mapped_residual_weight=0.1)
    assert not any(t.startswith("w0") for t in derive_tags(default_w))
    assert "w01" in derive_tags(other_w)


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
    assert path == BASELINES_ROOT / "emg_lap" / "h0918" / "h0918_k6_w05_mirror_noclip_net256_onlyvelrew_seed2" / "config.yaml"
