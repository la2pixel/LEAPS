"""a_hat from DEP's self-organized correlation matrix instead of the EMG
decoder: a prior with coherent structure but no biological content.

Wraps deprl.dep_controller.DEP. Fiber lengths are normalized here with a
running min/max, because SconeWrapper (which provides muscle_states) wraps
around LatentActionPriorWrapper, not the other way round.
"""

from __future__ import annotations

import gym
import numpy as np
from deprl.dep_controller import DEP


class DepContentPrior:
    def __init__(self, mapped_mask: np.ndarray, n_actuators: int):
        self._max_len = np.zeros(n_actuators)
        self._min_len = np.ones(n_actuators) * 100.0
        # same actuator subset as the EMG prior
        self.mapped_mask = mapped_mask
        self.dep = DEP()
        # DEP-RL human-run settings (Schumacher et al. 2023, arXiv:2206.00484, Table 4c)
        self.dep.params.update(dict(
            kappa=1896.0, tau=26, buffer_size=200,
            bias_rate=0.004154, s4avg=0, time_dist=4,
        ))
        # DEP.initialize() only reads the shape
        space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n_actuators,))
        self.dep.initialize(observation_space=space, action_space=space)

    def step(self, fiber_lengths: np.ndarray) -> np.ndarray:
        # running min/max rescaling, as in AbstractWrapper.muscle_states
        self._max_len = np.maximum(fiber_lengths, self._max_len)
        self._min_len = np.minimum(fiber_lengths, self._min_len)
        muscle_states = (
            (fiber_lengths - self._min_len) / (self._max_len - self._min_len + 0.1) - 0.5
        ) * 2.0

        raw = self.dep.step(muscle_states)[0]
        a_hat = np.zeros_like(raw)
        a_hat[self.mapped_mask] = raw[self.mapped_mask]
        return a_hat
