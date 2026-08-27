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
from leaps.configs.spec import Category, DecoderSource, ExperimentGroup, RewardVariant, RunSpec
from leaps.configs.writer import config_path, write_config

BASELINES_ROOT = Path("/home/nadinebadie/lalitha/LEAPS/baselines_DEPRL")

# (RunSpec kwargs, path to a real, non-drifted config.yaml it should match)
# All real files live under early_tests/ (2026-08-15 baselines_DEPRL reorg --
# everything that existed before that date got grouped there).
_EARLY = dict(experiment_group=ExperimentGroup.EARLY_TESTS)
CASES = [
    (
        dict(_EARLY, category=Category.MPO, body="h2190", reward_variant=RewardVariant.ONLYVELREW, seed=1, clip=True),
        "early_tests/mpo/h2190/h2190_clip_net256_onlyvelrew_seed1/config.yaml",
    ),
    (
        # h0918/seed0/clip, not h2190/seed1 -- that file predates the
        # 2026-08-18 DEP-block fix (verified against the authors' own
        # shipped configs) and no longer matches what the builder produces.
        # This one already used the verified preset.
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
    missing-file branch and not some unrelated bug.

    2026-08-19: pooled/k11 (the original target here) moved to
    synergy_priors/deprecated_precorrected_decoder/ along with every other
    pre-correction decoder -- only decoder_k6_AB06_corrected is still live
    in the active final_experiments path, so that's the happy-path case
    now."""
    spec = RunSpec(category=Category.EMG_LAP, body="h0918", dim_latent=6, decoder_source=DecoderSource.AB06_CORRECTED)
    decoder_path, norm_path = _decoder_paths(spec)
    assert Path(decoder_path).exists()
    assert Path(norm_path).exists()


def test_deprecated_precorrected_decoders_not_resolved_by_final_experiments():
    """Every decoder except AB06_corrected was trained under the old
    (pre-2026-08-16 threshold/scale, pre-2026-08-17 z-domain-fix) recipe
    and moved out of the active synergy_priors/{single_subject,
    pooled_subjects}/ into a deprecated/ sibling, 2026-08-19 -- final_experiments
    must never resolve to one of these, even by an old/default DecoderSource."""
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
    """2026-08-15: w=0.5 is now the locked default and goes untagged;
    w=0.1 (the old default) now gets tagged, inverted from the legacy
    convention. See naming.py's _DEFAULT_MAPPED_RESIDUAL_WEIGHT."""
    from leaps.configs.naming import derive_tags

    default_w = RunSpec(category=Category.EMG_LAP, body="h0918", mapped_residual_weight=0.5)
    other_w = RunSpec(category=Category.EMG_LAP, body="h0918", mapped_residual_weight=0.1)
    assert not any(t.startswith("w0") for t in derive_tags(default_w))
    assert "w01" in derive_tags(other_w)


def test_dep_prior_omits_decoder_and_sets_dep_content():
    """DEP_PRIOR is brand new (2026-08-15) -- no real config file exists
    yet to golden-file against, so this is a build-time correctness check,
    not a fidelity check. It never touches a decoder (see
    envs/dep_content_prior.py), so build_config must not require one to
    exist on disk, unlike every other is_lap category."""
    spec = RunSpec(category=Category.DEP_PRIOR, body="h0918", reward_variant=RewardVariant.ONLYVELREW, clip=False)
    env_expr = build_config(spec)["tonic"]["environment"]
    assert "decoder_path" not in env_expr
    assert "norm_path" not in env_expr
    assert "dep_content=True" in env_expr
    # mirror_left=True is still harmlessly emitted (every category gets it),
    # but mirror_mode is meaningless for DEP_PRIOR and must not appear.
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
    # 2026-08-19: clip/net{size}/"static" are omitted within
    # final_experiments when they match its only populated recipe
    # (noclip, net256) -- all 55 configs there are that recipe, so
    # spelling it out on every name was pure noise. "static" dropped
    # entirely (phase mirroring removed 2026-08-18, nothing left to
    # disambiguate from). See naming.derive_run_name()/RunSpec.is_default_recipe.
    # 2026-08-20: "mirror" dropped the same way -- mirror_left=True is the
    # default and every final_experiments config uses it, same
    # zero-information-token situation, just missed in the original pass.
    assert path == (
        BASELINES_ROOT / "final_experiments" / "emg_lap" / "h0918"
        / "h0918_k6_w05_onlyvelrew_seed2" / "config.yaml"
    )


def test_config_path_routes_non_default_recipe_to_other():
    """net512 (or clip=True) within final_experiments must physically
    segregate under .../final_experiments/other/... -- 2026-08-19, so the
    main tree never mixes recipes and its names never need net/clip
    tokens. derive_tonic_name() must match this exactly (see its
    docstring) since that's what deprl uses to resume a run."""
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
