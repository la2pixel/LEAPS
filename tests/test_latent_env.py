"""Unit tests for the two 2026-08-19 mechanism fixes in latent_env.py:

1. a_hat must be bounded [0,1] per mapped actuator (computed from the
   decoder's raw Sigmoid output, not the raw-EMG-unit rescale) -- tested
   directly against EMGToMuscleMapper, no simulator needed.
2. null_prior must ignore mapped_residual_weight entirely (always full-range
   residual_weight=1.0) -- tested against LatentActionPriorWrapper with
   dep_content=True, which skips decoder/norm loading and needs no real
   SCONE env, just a minimal fake with the attributes gym.Wrapper reads.
"""

import numpy as np
import pytest
from gym.spaces import Box

from leaps.envs.emg_mapping import EMG_CHANNELS, EMGToMuscleMapper
from leaps.envs.latent_env import LatentActionPriorWrapper


@pytest.mark.parametrize(
    "model_name,actuator_names",
    [
        # h1622: original coverage for this test, includes glut_med_r
        # (h1622-only -- see H1622_MAP = {**H0918_MAP, "gluteusmedius": ...}).
        ("h1622", ["glut_med_r", "hamstrings_r", "rect_fem_r", "vasti_r", "gastroc_r"]),
        # h0918: added 2026-08-20 -- this fix was only ever unit-tested
        # against h1622's actuator set before now, never h0918's. The fix
        # logic itself is body-agnostic (operates on actuator_names/
        # n_actuators generically, no branching on model_name anywhere in
        # latent_env.py/emg_mapping.py), but that was previously verified by
        # code reading only, not by an actual test run against h0918's own
        # (smaller, glut_med_r-free) mapped set. H0918_MAP's distinct targets:
        # gastroc_r/tib_ant_r/soleus_r/rect_fem_r/vasti_r/hamstrings_r.
        ("h0918", ["hamstrings_r", "rect_fem_r", "vasti_r", "gastroc_r", "tib_ant_r", "soleus_r"]),
    ],
)
def test_a_hat_bounded_unit_interval(model_name, actuator_names):
    """map_emg_to_muscles(emg_u) must stay in [0,1] for every mapped
    actuator given any emg_u in [0,1]^11 -- the fix's whole point. Rows are
    0/one-hot/an even split summing to <=1 (see EMGToMuscleMapper docstring),
    so a weighted average of [0,1] inputs is provably bounded the same way."""
    mapper = EMGToMuscleMapper(model_name, actuator_names, mirror_left=False)

    rng = np.random.default_rng(0)
    for _ in range(200):
        emg_u = rng.uniform(0.0, 1.0, size=len(EMG_CHANNELS)).astype(np.float32)
        a_hat = mapper.map_emg_to_muscles(emg_u)
        assert (a_hat >= 0.0).all()
        assert (a_hat <= 1.0).all()

    # edge cases: every channel pinned at the sigmoid's extremes
    for edge in (0.0, 1.0):
        emg_u = np.full(len(EMG_CHANNELS), edge, dtype=np.float32)
        a_hat = mapper.map_emg_to_muscles(emg_u)
        assert np.allclose(a_hat[a_hat != 0], edge)


class _FakeActuator:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class _FakeModel:
    def __init__(self, names):
        self._names = names

    def actuators(self):
        return [_FakeActuator(n) for n in self._names]


class _FakeUnwrapped:
    def __init__(self, names):
        self.model = _FakeModel(names)


class _FakeEnv:
    """Minimal stand-in satisfying gym.Wrapper.__init__'s attribute reads --
    dep_content=True skips decoder/norm.npz loading entirely, so nothing
    else about the env matters for this test."""

    def __init__(self, names):
        n = len(names)
        self.unwrapped = _FakeUnwrapped(names)
        self.action_space = Box(low=-1.0, high=1.0, shape=(6 + n,))
        self.observation_space = Box(low=-1.0, high=1.0, shape=(10,))
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {}


# h1622: original coverage, includes glut_med_r (h1622-only mapped target).
# h0918: added 2026-08-20, same rationale as test_a_hat_bounded_unit_interval
# above -- closes the previously-untested-for-h0918 gap for this fix too.
_ACTUATOR_NAMES_BY_BODY = {
    "h1622": ["glut_med_r", "add_mag_r", "hamstrings_r", "rect_fem_r", "vasti_r", "gastroc_r", "iliopsoas_r"],
    "h0918": ["hamstrings_r", "rect_fem_r", "vasti_r", "gastroc_r", "tib_ant_r", "soleus_r", "iliopsoas_r"],
}


@pytest.mark.parametrize("model_name", ["h1622", "h0918"])
def test_null_prior_ignores_mapped_residual_weight(model_name):
    """A null_prior=True wrapper must use full-range residual_weight=1.0
    everywhere, regardless of mapped_residual_weight -- a_hat is always 0
    for null_prior, so pinning a low weight on mapped actuators only caps
    their reachable activation (final_action = w * a_full, ceiling = w) for
    no reason. See the __init__ fix comment for the closed-form proof."""
    env = _FakeEnv(_ACTUATOR_NAMES_BY_BODY[model_name])
    wrapper = LatentActionPriorWrapper(
        env, model_name=model_name, dim_latent=6, residual_weight=1.0,
        null_prior=True, mapped_residual_weight=0.1, dep_content=True,
    )
    assert np.isscalar(wrapper.residual_weight) or (
        isinstance(wrapper.residual_weight, np.ndarray)
        and np.all(wrapper.residual_weight == 1.0)
    )


@pytest.mark.parametrize("model_name", ["h1622", "h0918"])
def test_non_null_prior_still_respects_mapped_residual_weight(model_name):
    """Guard against over-correcting: a real/untrained prior (null_prior=
    False) must still get the per-actuator override -- only null_prior is
    exempt."""
    env = _FakeEnv(_ACTUATOR_NAMES_BY_BODY[model_name])
    wrapper = LatentActionPriorWrapper(
        env, model_name=model_name, dim_latent=6, residual_weight=1.0,
        null_prior=False, mapped_residual_weight=0.1, dep_content=True,
    )
    assert isinstance(wrapper.residual_weight, np.ndarray)
    assert wrapper.residual_weight[wrapper._mapped_mask].max() == pytest.approx(0.1)
    assert wrapper.residual_weight[~wrapper._mapped_mask].max() == pytest.approx(1.0)


def test_add_mag_is_unmapped_on_h1622():
    """Sanity check behind the loosened_actuators ablation idea: add_mag_r has
    no H1622_MAP entry, so it must already be sitting at the full, unconstrained
    residual_weight (not tightened by mapped_residual_weight at all) -- it's
    glut_med_r (which *is* mapped) that's the one actually over-constrained on
    this antagonist pair, not add_mag_r. Confirms which actuator the ablation
    below should target before spending any compute on it."""
    env = _FakeEnv(_ACTUATOR_NAMES_BY_BODY["h1622"])
    wrapper = LatentActionPriorWrapper(
        env, model_name="h1622", dim_latent=6, residual_weight=1.0,
        null_prior=False, mapped_residual_weight=0.1, dep_content=True,
    )
    names = [a.name() for a in env.unwrapped.model.actuators()]
    assert wrapper.residual_weight[names.index("add_mag_r")] == pytest.approx(1.0)
    assert wrapper.residual_weight[names.index("glut_med_r")] == pytest.approx(0.1)


def test_loosened_actuators_overrides_only_named_actuators():
    """loosened_actuators should override mapped_residual_weight for exactly
    the named actuators, leaving every other mapped actuator at
    mapped_residual_weight and every unmapped actuator at the base
    residual_weight -- a targeted override, not a body-wide change."""
    env = _FakeEnv(_ACTUATOR_NAMES_BY_BODY["h1622"])
    wrapper = LatentActionPriorWrapper(
        env, model_name="h1622", dim_latent=6, residual_weight=1.0,
        null_prior=False, mapped_residual_weight=0.1, dep_content=True,
        loosened_actuators=["glut_med_r"], loosened_residual_weight=0.5,
    )
    names = [a.name() for a in env.unwrapped.model.actuators()]
    assert wrapper.residual_weight[names.index("glut_med_r")] == pytest.approx(0.5)
    # other mapped actuators (e.g. hamstrings_r) untouched, still at mapped_residual_weight
    assert wrapper.residual_weight[names.index("hamstrings_r")] == pytest.approx(0.1)
    # unmapped actuators (add_mag_r, iliopsoas_r) untouched, still at base residual_weight
    assert wrapper.residual_weight[names.index("add_mag_r")] == pytest.approx(1.0)
    assert wrapper.residual_weight[names.index("iliopsoas_r")] == pytest.approx(1.0)


def test_loosened_actuators_requires_mapped_residual_weight():
    env = _FakeEnv(_ACTUATOR_NAMES_BY_BODY["h1622"])
    with pytest.raises(ValueError, match="mapped_residual_weight"):
        LatentActionPriorWrapper(
            env, model_name="h1622", dim_latent=6, residual_weight=1.0,
            null_prior=False, mapped_residual_weight=None, dep_content=True,
            loosened_actuators=["glut_med_r"], loosened_residual_weight=0.5,
        )


def test_loosened_actuators_rejects_null_prior():
    env = _FakeEnv(_ACTUATOR_NAMES_BY_BODY["h1622"])
    with pytest.raises(ValueError, match="mapped_residual_weight"):
        LatentActionPriorWrapper(
            env, model_name="h1622", dim_latent=6, residual_weight=1.0,
            null_prior=True, mapped_residual_weight=0.1, dep_content=True,
            loosened_actuators=["glut_med_r"], loosened_residual_weight=0.5,
        )


def test_loosened_actuators_rejects_unknown_name():
    env = _FakeEnv(_ACTUATOR_NAMES_BY_BODY["h1622"])
    with pytest.raises(ValueError, match="not found"):
        LatentActionPriorWrapper(
            env, model_name="h1622", dim_latent=6, residual_weight=1.0,
            null_prior=False, mapped_residual_weight=0.1, dep_content=True,
            loosened_actuators=["not_a_real_muscle"], loosened_residual_weight=0.5,
        )
