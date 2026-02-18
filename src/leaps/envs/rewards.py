"""Custom reward functions for the gait10dof18musc environment.

ForwardVelocityReward computes exp(-||v_pelvis_tx - target||²) using the
pelvis_tx slide joint velocity. This avoids the free-joint assumption of
LocoMuJoCo's built-in TargetXVelocityReward.
"""

from types import ModuleType
from typing import Any, Dict, Tuple, Union

import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.utils import mj_jntname2qvelid


class ForwardVelocityReward(Reward):
    """Reward based on forward velocity of the pelvis_tx slide joint.

    reward = exp(-(v_x - target_velocity)²)

    Uses the root_free_joint_xml_name info property (set to "pelvis_tx")
    to find the correct qvel index.
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
