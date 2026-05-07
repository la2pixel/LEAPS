"""Custom reward functions for the gait10dof18musc environment.

WalkingReward (A):
  reward = alive_bonus + exp(-(v_x - 1.25)²)
  Always positive. Gaussian peak at 1.25 m/s. Strong gradient everywhere.

LinearWalkingReward (G):
  reward = 0.1 + 0.1·min(0, v_x) + max(0, min(v_x, 1.25)) / 1.25
  Piecewise linear. Standing barely rewarded (0.1). Backward penalised.
  Strong forward gradient (0.8/m/s). Always positive for v_x > -1 m/s.
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
    """Reward for walking: velocity tracking + effort penalty.

    Velocity term (from Haeufle et al. 2024, H0918 model — same as ours):
        r_vel = exp(-(v - target)²)   if v < target   (Gaussian ramp up)
        r_vel = 1.0                    if v >= target  (flat — don't chase speed)

    Flat above target stops the policy over-optimizing speed at the expense
    of effort/stability. Below target, Gaussian gives strong gradient toward 1.25.

    Effort penalty: -effort_weight * mean(a²)
    Penalizes large muscle activations → discourages co-contraction,
    encourages energy-efficient coordination similar to human EMG patterns.

    Total reward = alive_bonus + r_vel - effort_weight * mean(a²)
    """

    def __init__(
        self,
        env: Any,
        alive_bonus: float = 0.5,
        target_velocity: float = 1.25,
        effort_weight: float = 0.01,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self._alive_bonus = alive_bonus
        self._target_velocity = float(target_velocity)
        self._effort_weight = float(effort_weight)

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
        x_vel = backend.where(backend.isnan(x_vel), 0.0, x_vel)

        # Gaussian ramp to target, flat above — stops policy chasing speed
        # at the expense of stability (same pattern as Haeufle et al. 2024).
        vel_reward = backend.where(
            x_vel >= self._target_velocity,
            4.0,
            4.0 * backend.exp(-backend.square(x_vel - self._target_velocity)),
        )

        # Effort penalty: penalize mean squared muscle activation
        # action is in [-1, 1]; convert to [0, 1] muscle activation for penalty
        activation = 0.5 * (action + 1.0)
        effort = backend.mean(backend.square(activation))

        reward = self._alive_bonus + vel_reward - self._effort_weight * effort
        return reward, carry


class HaeufleReward(Reward):
    """Reward from Haeufle et al. (2024) — same H0918 model as ours.

    r = r_vel - alpha(t) * c_effort - w_nactive * N_active

    r_vel: flat at 1.0 above target speed, Gaussian ramp below.
        Flat ceiling stops the policy from chasing speed at the expense
        of effort and stability.

    c_effort: mean(|a|^3) — cubic penalises co-contraction more than
        quadratic (a=0.5 contributes 0.125 vs 0.25 for a^2).

    N_active: fraction of muscles with activation > nactive_threshold.
        Encourages sparse activation patterns (few muscles at once).

    alpha(t): adaptive effort weight. Starts at 0, ramps linearly to
        max_effort_weight over the first effort_ramp_steps steps of each
        episode. This avoids the performance collapse the paper describes:
        a strong effort penalty from step 1 prevents the policy from ever
        learning to walk in the first place.

    NOTE: the paper's smoothness term (u - u_prev)^2 requires storing the
    previous action. LocoMuJoCo sets carry.last_action = current_action
    before calling the reward, so prev_action is unavailable here.
    Omitted for now — add it by including last_action in the obs space.
    """

    def __init__(
        self,
        env: Any,
        target_velocity: float = 1.25,
        alive_bonus: float = 0.1,
        max_effort_weight: float = 0.1,
        effort_ramp_steps: int = 300,
        nactive_weight: float = 0.01,
        nactive_threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self._target_velocity = float(target_velocity)
        self._alive_bonus = float(alive_bonus)
        self._max_effort_weight = float(max_effort_weight)
        self._effort_ramp_steps = float(effort_ramp_steps)
        self._nactive_weight = float(nactive_weight)
        self._nactive_threshold = float(nactive_threshold)

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
        x_vel = backend.where(backend.isnan(x_vel), 0.0, x_vel)

        # Velocity reward: flat above target, Gaussian ramp below
        vel_reward = backend.where(
            x_vel >= self._target_velocity,
            1.0,
            backend.exp(-backend.square(x_vel - self._target_velocity)),
        )

        # Muscle activation in [0, 1]
        activation = 0.5 * (action + 1.0)

        # Cubic effort penalty
        effort = backend.mean(activation ** 3)

        # Adaptive weight: 0 → max_effort_weight over first effort_ramp_steps steps.
        # Integer / float division works in both numpy and jax.
        step_frac = backend.minimum(
            carry.cur_step_in_episode / self._effort_ramp_steps, 1.0
        )
        alpha = self._max_effort_weight * step_frac

        # N_active: fraction of muscles exceeding threshold
        n_active = backend.mean(
            backend.where(activation > self._nactive_threshold, 1.0, 0.0)
        )

        reward = self._alive_bonus + vel_reward - alpha * effort - self._nactive_weight * n_active
        return reward, carry


class LinearWalkingReward(Reward):
    """Piecewise-linear walking reward (Option G).

    reward = 0.1 + 0.1·min(0, v_x) + max(0, min(v_x, target)) / target

    - v_x = 1.25  →  1.10  (max)
    - v_x = 0.0   →  0.10  (standing barely rewarded — strong pull forward)
    - v_x = -0.5  →  0.05  (backward penalised, half of standing)
    - v_x < -1.0  →  ≤ 0   (extreme backward, essentially unreachable)

    Gradient at v=0 toward target: 0.8/ms — 8× stronger than backward penalty.
    """

    def __init__(self, env: Any, target_velocity: float = 1.25, **kwargs):
        super().__init__(env, **kwargs)
        self._target_velocity = float(target_velocity)
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
        x_vel = backend.where(backend.isnan(x_vel), 0.0, x_vel)
        fwd = backend.maximum(0.0, backend.minimum(x_vel, self._target_velocity)) / self._target_velocity
        backward_penalty = 0.1 * backend.minimum(0.0, x_vel)
        reward = 0.1 + fwd + backward_penalty
        return reward, carry
