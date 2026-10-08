"""latent_env.py: a_hat stays in [0, 1] per mapped actuator, and null_prior
ignores mapped_residual_weight. No simulator needed."""

import numpy as np
import pytest
from gym.spaces import Box

from leaps.envs.emg_mapping import EMG_CHANNELS, EMGToMuscleMapper
from leaps.envs.latent_env import LatentActionPriorWrapper


@pytest.mark.parametrize(
    "model_name,actuator_names",
    [
        ("h1622", ["glut_med_r", "hamstrings_r", "rect_fem_r", "vasti_r", "gastroc_r"]),
        ("h0918", ["hamstrings_r", "rect_fem_r", "vasti_r", "gastroc_r", "tib_ant_r", "soleus_r"]),
    ],
)
def test_a_hat_bounded_unit_interval(model_name, actuator_names):
    """map_emg_to_muscles(emg_u) stays in [0, 1] for any emg_u in [0, 1]^11."""
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
    """Minimal stand-in for the attributes gym.Wrapper reads."""

    def __init__(self, names):
        n = len(names)
        self.unwrapped = _FakeUnwrapped(names)
        self.action_space = Box(low=-1.0, high=1.0, shape=(6 + n,))
        self.observation_space = Box(low=-1.0, high=1.0, shape=(10,))
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {}


_ACTUATOR_NAMES_BY_BODY = {
    "h1622": ["glut_med_r", "add_mag_r", "hamstrings_r", "rect_fem_r", "vasti_r", "gastroc_r", "iliopsoas_r"],
    "h0918": ["hamstrings_r", "rect_fem_r", "vasti_r", "gastroc_r", "tib_ant_r", "soleus_r", "iliopsoas_r"],
}


@pytest.mark.parametrize("model_name", ["h1622", "h0918"])
def test_null_prior_ignores_mapped_residual_weight(model_name):
    """null_prior uses residual_weight=1.0 everywhere, whatever mapped_residual_weight is."""
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
    """A real or untrained prior still gets the per-actuator weight."""
    env = _FakeEnv(_ACTUATOR_NAMES_BY_BODY[model_name])
    wrapper = LatentActionPriorWrapper(
        env, model_name=model_name, dim_latent=6, residual_weight=1.0,
        null_prior=False, mapped_residual_weight=0.1, dep_content=True,
    )
    assert isinstance(wrapper.residual_weight, np.ndarray)
    assert wrapper.residual_weight[wrapper._mapped_mask].max() == pytest.approx(0.1)
    assert wrapper.residual_weight[~wrapper._mapped_mask].max() == pytest.approx(1.0)


def test_add_mag_is_unmapped_on_h1622():
    """add_mag_r has no EMG channel, so it sits at the full residual_weight."""
    env = _FakeEnv(_ACTUATOR_NAMES_BY_BODY["h1622"])
    wrapper = LatentActionPriorWrapper(
        env, model_name="h1622", dim_latent=6, residual_weight=1.0,
        null_prior=False, mapped_residual_weight=0.1, dep_content=True,
    )
    names = [a.name() for a in env.unwrapped.model.actuators()]
    assert wrapper.residual_weight[names.index("add_mag_r")] == pytest.approx(1.0)
    assert wrapper.residual_weight[names.index("glut_med_r")] == pytest.approx(0.1)


def test_loosened_actuators_overrides_only_named_actuators():
    """loosened_actuators overrides the weight of exactly the named actuators."""
    env = _FakeEnv(_ACTUATOR_NAMES_BY_BODY["h1622"])
    wrapper = LatentActionPriorWrapper(
        env, model_name="h1622", dim_latent=6, residual_weight=1.0,
        null_prior=False, mapped_residual_weight=0.1, dep_content=True,
        loosened_actuators=["glut_med_r"], loosened_residual_weight=0.5,
    )
    names = [a.name() for a in env.unwrapped.model.actuators()]
    assert wrapper.residual_weight[names.index("glut_med_r")] == pytest.approx(0.5)
    # other mapped actuators stay at mapped_residual_weight
    assert wrapper.residual_weight[names.index("hamstrings_r")] == pytest.approx(0.1)
    # unmapped actuators stay at residual_weight
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
