"""a_hat sourced from DEP's self-organized correlation matrix instead of
the EMG decoder -- a content-source control, not a competing controller
(see latent_env.py's dep_content docstring note for why "DEP as a blended
RL controller" is a different, already-parked idea).

null_prior (a_hat=0) and untrained_decoder (random-init decoder, garbage
but same-shaped/same-scale content) bound "does real EMG content matter"
but can't isolate *why* untrained_decoder fails -- garbage content, or
specifically non-biological content. DEP self-organizes a coherent,
structured actuator-coordination matrix purely from the agent's own
sensorimotor experience, zero biological data -- feeding that through the
same a_hat slot tests "does it need to be biological, or just coherent."

Wraps deprl.dep_controller.DEP directly rather than reimplementing
self-organization -- that class is already used (as the exploration
policy inside dep_factory) throughout this codebase's DEP-MPO baselines.

Normalizes raw fiber lengths itself (running min/max, force_scale=0)
rather than reading AbstractWrapper.muscle_states (env_wrappers/wrappers.py)
-- that property lives on SconeWrapper, which apply_wrapper() attaches
*outside* the whole tonic.environment expression (confirmed via
custom_distributed.py's build_env_from_dict -> env_tonic_compat ->
apply_wrapper(eval(env)), where eval(env) already includes the fully-built
LatentActionPriorWrapper). So SconeWrapper wraps around
LatentActionPriorWrapper, not the other way around -- self.env inside this
wrapper is the raw, un-SconeWrapper'd gym.make() result, and .muscle_states
genuinely does not exist there. Confirmed no config in this project ever
overrides force_scale (grepped baselines_DEPRL/ and depRL-sconegym/),
so hardcoding force_scale=0 here reproduces the real formula every
existing dep_factory-based baseline actually uses.
"""

from __future__ import annotations

import gym
import numpy as np
from deprl.dep_controller import DEP


class DepContentPrior:
    def __init__(self, mapped_mask: np.ndarray, n_actuators: int):
        self._max_len = np.zeros(n_actuators)
        self._min_len = np.ones(n_actuators) * 100.0
        # mapped_mask: same self._mapped_mask the real/null/untrained
        # conditions use -- masking DEP's output to this exact subset keeps
        # coverage, action-space size, and blend weight identical across
        # the whole ladder, so only the content source changes.
        self.mapped_mask = mapped_mask
        self.dep = DEP()
        # DEP() with no params_path loads dep_controller.py's own shipped
        # defaults (kappa=100, tau=20, ...) -- overridden here with the
        # DEP-RL paper's own human-run settings (Schumacher, Haeufle,
        # Buchler, Schmitt, Martius, "DEP-RL: Embodied Exploration for RL
        # in Overactuated and Musculoskeletal Systems", ICLR 2023,
        # arXiv:2206.00484, Table 4c) -- re-verified against the arxiv PDF
        # directly, not from memory. Their human-run body has 18 leg
        # muscles and no arms (a in R^18, same actuator count as h0918)
        # at a 10ms timestep, 1000-step (10s) episodes -- both tau and
        # time_dist are measured in simulation steps, not seconds, and
        # that 10ms/10s setup is exactly what this project's simulator
        # actually runs at (gaitgym.py's hardcoded step_size=0.01, not the
        # configs' nominally-stated 0.025) -- not just the closest
        # available task, the timescale these values were tuned for.
        # Fields Table 4 never varies per-task (regularization,
        # q_norm_selector, normalization, sensor_delay, with_learning)
        # are left at the class's own shared defaults, matching the
        # paper's own convention of only tuning DEP once and reusing it.
        self.dep.params.update(dict(
            kappa=1896.0, tau=26, buffer_size=200,
            bias_rate=0.004154, s4avg=0, time_dist=4,
        ))
        # DEP.initialize() hardcodes its own internal Box(-1, 1, ...)
        # regardless of what's passed here (dep_controller.py:38-40) -- the
        # bounds below are for the .shape read, not the actual clamp.
        space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n_actuators,))
        self.dep.initialize(observation_space=space, action_space=space)

    def step(self, fiber_lengths: np.ndarray) -> np.ndarray:
        # Same running min/max rescaling as AbstractWrapper.muscle_states,
        # force term dropped (force_scale=0 everywhere in this project).
        self._max_len = np.maximum(fiber_lengths, self._max_len)
        self._min_len = np.minimum(fiber_lengths, self._min_len)
        muscle_states = (
            (fiber_lengths - self._min_len) / (self._max_len - self._min_len + 0.1) - 0.5
        ) * 2.0

        raw = self.dep.step(muscle_states)[0]
        a_hat = np.zeros_like(raw)
        a_hat[self.mapped_mask] = raw[self.mapped_mask]
        return a_hat
