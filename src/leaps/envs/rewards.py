"""Reward functions for muscle-actuated humanoid locomotion.

All rewards accept x_vel (m/s) and action (muscle activations in [0, 1]).
The env wrapper extracts these from obs/info.

WalkingReward:        alive_bonus + Gaussian(v_x, target) - effort_weight * mean(a²)
HaeufleReward:        alive_bonus + Gaussian(v_x, target) - alpha(t) * mean(|a|³) - w * N_active
LinearWalkingReward:  piecewise-linear in v_x
ForwardVelocityReward: pure Gaussian velocity tracking
"""

import numpy as np


class ForwardVelocityReward:
    def __init__(self, target_velocity: float = 1.25):
        self._target_vel = target_velocity

    def __call__(self, x_vel: float, action: np.ndarray) -> float:
        return float(np.exp(-((x_vel - self._target_vel) ** 2)))


class WalkingReward:
    """alive_bonus + Gaussian velocity reward (flat above target) - effort penalty.

    Flat ceiling above target stops the policy chasing speed at the expense of
    stability. Effort penalty discourages co-contraction.
    """

    def __init__(
        self,
        alive_bonus: float = 0.5,
        target_velocity: float = 1.25,
        effort_weight: float = 0.01,
    ):
        self._alive_bonus = alive_bonus
        self._target_velocity = target_velocity
        self._effort_weight = effort_weight

    def __call__(self, x_vel: float, action: np.ndarray) -> float:
        if x_vel >= self._target_velocity:
            vel_reward = 4.0
        else:
            vel_reward = 4.0 * float(np.exp(-((x_vel - self._target_velocity) ** 2)))
        effort = float(np.mean(action ** 2))
        return self._alive_bonus + vel_reward - self._effort_weight * effort


class HaeufleReward:
    """Reward from Haeufle et al. (2024) — same H0918 model.

    r = alive_bonus + r_vel - alpha(t) * mean(|a|³) - w_nactive * N_active

    alpha(t): ramps from 0 to max_effort_weight over effort_ramp_steps to avoid
    the policy collapsing before it learns to walk.
    """

    def __init__(
        self,
        target_velocity: float = 1.25,
        alive_bonus: float = 0.1,
        max_effort_weight: float = 0.1,
        effort_ramp_steps: int = 300,
        nactive_weight: float = 0.01,
        nactive_threshold: float = 0.1,
    ):
        self._target_velocity = target_velocity
        self._alive_bonus = alive_bonus
        self._max_effort_weight = max_effort_weight
        self._effort_ramp_steps = effort_ramp_steps
        self._nactive_weight = nactive_weight
        self._nactive_threshold = nactive_threshold

    def __call__(self, x_vel: float, action: np.ndarray, step: int = 0) -> float:
        if x_vel >= self._target_velocity:
            vel_reward = 1.0
        else:
            vel_reward = float(np.exp(-((x_vel - self._target_velocity) ** 2)))
        effort = float(np.mean(np.abs(action) ** 3))
        alpha = self._max_effort_weight * min(step / self._effort_ramp_steps, 1.0)
        n_active = float(np.mean(action > self._nactive_threshold))
        return self._alive_bonus + vel_reward - alpha * effort - self._nactive_weight * n_active


class LinearWalkingReward:
    """Piecewise-linear walking reward.

    reward = 0.1 + max(0, min(v_x, target)) / target + 0.1 * min(0, v_x)
    - Standing (v=0): 0.1 — strong pull forward.
    - Backward (v<0): penalised.
    - At target (v=1.25): 1.1 (max).
    """

    def __init__(self, target_velocity: float = 1.25):
        self._target_velocity = target_velocity

    def __call__(self, x_vel: float, action: np.ndarray) -> float:
        fwd = max(0.0, min(x_vel, self._target_velocity)) / self._target_velocity
        backward_penalty = 0.1 * min(0.0, x_vel)
        return 0.1 + fwd + backward_penalty
