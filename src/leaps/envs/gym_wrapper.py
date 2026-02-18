"""Gymnasium wrappers for the Gait10dof18Musc LocoMuJoCo environment.

Provides SB3-compatible Gymnasium envs:
  - Gait10dof18MuscGymEnv: direct 18-dim muscle action space
  - LatentActionGymEnv: latent_dim action space, frozen decoder maps z → muscles
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gymnasium as gym  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium import spaces

from leaps.envs.emg_mapping import EMGToMuscleMapper
from leaps.envs.gait10dof_env import Gait10dof18Musc


def _make_loco_env(target_velocity: float = 1.25, **kwargs) -> Gait10dof18Musc:
    """Create a Gait10dof18Musc env with ForwardVelocityReward."""
    kwargs.setdefault("terminal_state_type", "HeightJointTerminalStateHandler")
    kwargs.setdefault("goal_type", "NoGoal")
    kwargs.setdefault("reward_type", "ForwardVelocityReward")
    kwargs.setdefault("reward_params", {"target_velocity": target_velocity})
    return Gait10dof18Musc(**kwargs)


class Gait10dof18MuscGymEnv(gym.Env):
    """Gymnasium wrapper for Gait10dof18Musc with direct 18-dim muscle actions.

    Action space: Box(-1, 1, shape=(18,)) — LocoMuJoCo convention.
    Observation space: Box(-inf, inf, shape=(obs_dim,)).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        target_velocity: float = 1.25,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        **loco_kwargs,
    ):
        super().__init__()
        if render_mode == "human":
            loco_kwargs["headless"] = False
        else:
            loco_kwargs.setdefault("headless", True)

        self._env = _make_loco_env(target_velocity=target_velocity, **loco_kwargs)
        self._max_steps = max_episode_steps
        self._step_count = 0
        self.render_mode = render_mode

        # Convert LocoMuJoCo Box to gymnasium Box
        loco_act = self._env.info.action_space
        loco_obs = self._env.info.observation_space
        self.action_space = spaces.Box(
            low=np.asarray(loco_act.low, dtype=np.float32),
            high=np.asarray(loco_act.high, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.asarray(loco_obs.low, dtype=np.float32),
            high=np.asarray(loco_obs.high, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        key = jax.random.key(seed if seed is not None else 0)
        obs = self._env.reset(key=key)
        self._step_count = 0
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(action))
        self._step_count += 1
        if self._step_count >= self._max_steps:
            truncated = True
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        if self.render_mode is not None:
            return self._env.render(record=(self.render_mode == "rgb_array"))
        return None

    def close(self):
        self._env.stop()


class LatentActionGymEnv(gym.Env):
    """Gymnasium wrapper with latent action space and frozen decoder.

    Action space: Box(-1, 1, shape=(latent_dim,)).
    The decoder maps z → (1111,) → reshape (101, 11) → stride mean → EMGToMuscleMapper → 18 muscles.

    This is a simple stride-mean approach: decode the full stride, average over
    the 101 timepoints, then map the 11 EMG channels to 18 muscles.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        decoder_fn,
        latent_dim: int,
        target_velocity: float = 1.25,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        **loco_kwargs,
    ):
        """
        Args:
            decoder_fn: Callable (latent_dim,) → (1111,) numpy array in [0,1].
                        Typically StrideFlatAE.decode with frozen weights.
            latent_dim: Dimension of the latent action space.
            target_velocity: Target forward velocity for the reward.
            max_episode_steps: Max steps per episode.
        """
        super().__init__()
        if render_mode == "human":
            loco_kwargs["headless"] = False
        else:
            loco_kwargs.setdefault("headless", True)

        self._env = _make_loco_env(target_velocity=target_velocity, **loco_kwargs)
        self._decoder_fn = decoder_fn
        self._latent_dim = latent_dim
        self._mapper = EMGToMuscleMapper()
        self._max_steps = max_episode_steps
        self._step_count = 0
        self.render_mode = render_mode

        # Latent action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(latent_dim,), dtype=np.float32,
        )

        # Observation space from underlying env
        loco_obs = self._env.info.observation_space
        self.observation_space = spaces.Box(
            low=np.asarray(loco_obs.low, dtype=np.float32),
            high=np.asarray(loco_obs.high, dtype=np.float32),
            dtype=np.float32,
        )

    def _decode_action(self, z: np.ndarray) -> np.ndarray:
        """Decode latent z to 18-dim muscle command in [-1, 1].

        Pipeline: z → decode → (1111,) → reshape (101, 11) → mean over time
        → (11,) EMG → EMGToMuscleMapper → (18,) in [0,1] → scale to [-1,1].
        """
        # Decode: (latent_dim,) → (1111,) in [0, 1]
        decoded = self._decoder_fn(z.reshape(1, -1))  # (1, 1111)
        stride = decoded.reshape(101, 11)  # (101, 11)

        # Average over stride to get single EMG snapshot
        emg_mean = stride.mean(axis=0)  # (11,)

        # Map 11 EMG channels → 18 muscles in [0, 1]
        muscles_01 = self._mapper.map_emg_to_muscles(emg_mean)  # (18,)

        # Convert to LocoMuJoCo range [-1, 1]
        return (2.0 * muscles_01 - 1.0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        key = jax.random.key(seed if seed is not None else 0)
        obs = self._env.reset(key=key)
        self._step_count = 0
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        muscle_action = self._decode_action(np.asarray(action, dtype=np.float32))
        obs, reward, terminated, truncated, info = self._env.step(muscle_action)
        self._step_count += 1
        if self._step_count >= self._max_steps:
            truncated = True
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        if self.render_mode is not None:
            return self._env.render(record=(self.render_mode == "rgb_array"))
        return None

    def close(self):
        self._env.stop()
