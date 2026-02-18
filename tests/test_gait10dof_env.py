"""Tests for the Gait10dof18Musc LocoMuJoCo environment."""

import numpy as np
import pytest


@pytest.fixture
def env():
    from leaps.envs import Gait10dof18Musc

    e = Gait10dof18Musc()
    yield e
    e.stop()


@pytest.fixture
def env_with_terminal():
    from leaps.envs import Gait10dof18Musc

    e = Gait10dof18Musc.generate()
    yield e
    e.stop()


class TestGait10dof18Musc:
    def test_instantiation(self, env):
        assert env is not None

    def test_action_dim(self, env):
        assert env.action_dim == 18

    def test_obs_shape(self, env):
        obs = env.reset()
        assert obs.shape == (18,), f"Expected (18,), got {obs.shape}"

    def test_step(self, env):
        env.reset()
        action = np.zeros(18)
        result = env.step(action)
        obs = result[0]
        assert obs.shape == (18,)

    def test_step_with_ones(self, env):
        """Full activation should not crash."""
        env.reset()
        action = np.ones(18)  # LocoMuJoCo expects [-1, 1]
        result = env.step(action)
        obs = result[0]
        assert obs.shape == (18,)
        assert np.all(np.isfinite(obs))

    def test_free_jnt_ids_empty(self, env):
        assert env.free_jnt_qpos_id.shape == (0, 7)
        assert env.free_jnt_qvel_id.shape == (0, 6)

    def test_info_properties(self, env):
        assert env.root_body_name == "pelvis"
        assert env.upper_body_xml_name == "torso"
        assert env.root_height_healthy_range == (0.5, 1.3)

    def test_multiple_steps(self, env):
        """Run 100 steps without crashing."""
        env.reset()
        for _ in range(100):
            action = np.zeros(18)
            env.step(action)


class TestRegistration:
    def test_env_registered(self):
        from loco_mujoco.core.mujoco_base import Mujoco

        assert "Gait10dof18Musc" in Mujoco.registered_envs

    def test_terminal_handler_registered(self):
        from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler

        assert "HeightJointTerminalStateHandler" in TerminalStateHandler.registered


class TestTerminalState:
    def test_generate_with_terminal(self, env_with_terminal):
        obs = env_with_terminal.reset()
        assert obs.shape == (18,)

    def test_terminal_on_fall(self, env_with_terminal):
        """Forcing pelvis_ty below threshold should trigger terminal."""
        env_with_terminal.reset()
        # Set pelvis_ty to very low value
        pelvis_ty_id = 1  # pelvis_ty is the second joint (index 1 in qpos)
        env_with_terminal._data.qpos[pelvis_ty_id] = 0.1  # below healthy range (0.5, 1.3)

        obs = np.zeros(18)
        info = {}
        absorbing, _ = env_with_terminal._terminal_state_handler.is_absorbing(
            env_with_terminal, obs, info, env_with_terminal._data, None
        )
        assert absorbing, "Should be terminal when pelvis_ty < 0.5"

    def test_not_terminal_at_default(self, env_with_terminal):
        """Default pose (pelvis_ty=0.95) should not be terminal."""
        env_with_terminal.reset()
        obs = np.zeros(18)
        info = {}
        absorbing, _ = env_with_terminal._terminal_state_handler.is_absorbing(
            env_with_terminal, obs, info, env_with_terminal._data, None
        )
        assert not absorbing, "Should not be terminal at default pelvis_ty=0.95"
