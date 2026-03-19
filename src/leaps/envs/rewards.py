"""Custom reward functions for the gait10dof18musc environment.

WalkingReward combines:
  - Forward velocity tracking: exp(-(v_x - target)²)
  - Survival bonus: +1 per step alive
  - Posture penalty: -w * tilt² (penalize pelvis leaning)
"""

from types import ModuleType
from typing import Any, Dict, Tuple, Union

import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid


class ForwardVelocityReward(Reward):
    """Simple velocity-only reward (kept for backward compat).

    reward = exp(-(v_x - target_velocity)²)
    """

    def __init__(self, env: Any, target_velocity: float = 1.25, **kwargs):
        super().__init__(env, **kwargs)
        self._target_vel = target_velocity
        root_joint = self._info_props["root_free_joint_xml_name"]
        self._x_vel_idx = mj_jntname2qvelid(root_joint, env._model)[0]

    def __call__(
        self,
        state: Union[np.ndarray, Any],
        action: Union[np.ndarray, Any],
        next_state: Union[np.ndarray, Any],
        absorbing: bool,
        info: Dict[str, Any],
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[float, Any]:
        x_vel = backend.squeeze(data.qvel[self._x_vel_idx])
        reward = backend.exp(-backend.square(x_vel - self._target_vel))
        return reward, carry


class WalkingReward(Reward):
    """Reward for walking: velocity tracking + survival + posture.

    reward = velocity_reward + alive_bonus - posture_penalty

    Components:
      - velocity:  exp(-(v_x - target)²)          ∈ (0, 1]
      - alive:     +1.0 per step                   constant
      - posture:   -posture_weight * tilt²          ≥ 0

    Default pose: pelvis_tilt ≈ 0 (upright).
    """

    def __init__(
        self,
        env: Any,
        target_velocity: float = 1.25,
        alive_bonus: float = 1.0,
        posture_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self._target_vel = target_velocity
        self._alive_bonus = alive_bonus
        self._posture_weight = posture_weight

        root_joint = self._info_props["root_free_joint_xml_name"]
        self._x_vel_idx = mj_jntname2qvelid(root_joint, env._model)[0]
        self._tilt_qpos_idx = int(
            np.array(mj_jntname2qposid("pelvis_tilt", env._model)).flat[0]
        )

    def __call__(
        self,
        state: Union[np.ndarray, Any],
        action: Union[np.ndarray, Any],
        next_state: Union[np.ndarray, Any],
        absorbing: bool,
        info: Dict[str, Any],
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[float, Any]:
        # Velocity tracking
        x_vel = backend.squeeze(data.qvel[self._x_vel_idx])
        vel_reward = backend.exp(-backend.square(x_vel - self._target_vel))

        # Posture: penalize tilt away from upright (0 rad)
        tilt = backend.squeeze(data.qpos[self._tilt_qpos_idx])
        posture_penalty = self._posture_weight * backend.square(tilt)

        reward = vel_reward + self._alive_bonus - posture_penalty
        return reward, carry
