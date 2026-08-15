"""Heel-strike-triggered gait mirroring, an alternative to
EMGToMuscleMapper's static synchronous mirror (mirror_left=True there gives
a_hat_l == a_hat_r every step, no phase offset -- see that module's
docstring). Real bipedal gait is ~anti-phase between legs; this class
instead gives the left leg a time-lagged copy of the right leg's own
recent EMG-derived signal, with the lag tracking an online estimate of the
current stride's half-period.

NOT the same mechanism as the literature's "PHASE" method (Abdolhosseini,
Ling, Xie, Peng, van de Panne, "On Learning Symmetric Locomotion", MIG
2019, Eq. 4 -- the paper Park et al.'s "phase-based mirroring... when the
model find the heel strike event" sentence traces to). That method mirrors
the *entire* state, runs it through the *same* policy, then mirrors the
*entire* action back, for one half of every gait cycle -- it requires a
full state/action mirror function over every joint. That doesn't have a
well-defined meaning for our action space's k-dim latent z (an abstract
AE code, not a spatially-organized joint vector with an obvious left/right
swap) -- the same kind of dimensional mismatch that blocks DEP from
applying to this action space (see dep_controller.py's C matrix). This
class is a bespoke adaptation of the same underlying goal (anti-phase
left-right coordination) to a residual+latent-prior architecture the
literature's exact PHASE method doesn't fit, not an implementation of it.

Kept separate from EMGToMuscleMapper on purpose: that class is a pure,
stateless, construction-time weight-matrix builder with no simulator
access. This one needs per-step state (an EMG history buffer, heel-strike
timestamps, a running period estimate) and per-step simulator input
(contact force) that EMGToMuscleMapper was never built to hold.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class PhaseMirror:
    def __init__(
        self,
        weights_right_only: np.ndarray,
        actuator_names: list[str],
        *,
        buffer_size: int = 200,
        default_half_period: float = 50.0,
        contact_threshold: float = 0.1,
        smoothing: float = 0.2,
        refractory_period: int = 20,
    ):
        # weights_right_only: (n_actuators, 11), real content on _r rows,
        # zero on _l rows -- i.e. an EMGToMuscleMapper built with
        # mirror_left=False. This class fills the _l rows in itself,
        # time-lagged, instead of EMGToMuscleMapper copying them statically.
        self.weights = weights_right_only
        self.actuator_names = actuator_names
        self.buffer_size = buffer_size
        self.default_half_period = default_half_period
        self.contact_threshold = contact_threshold
        self.smoothing = smoothing
        self.refractory_period = refractory_period

        # Which _r row (with real, nonzero content) feeds which _l row.
        # Computed once at construction, not per step.
        self._right_to_left: dict[int, int] = {}
        for i, name in enumerate(actuator_names):
            if not name.endswith("_r") or not self.weights[i].any():
                continue
            left_name = name[:-2] + "_l"
            if left_name in actuator_names:
                self._right_to_left[i] = actuator_names.index(left_name)

        self.reset()

    def reset(self) -> None:
        """Clear buffer/timing state. Must be called on every episode
        reset -- a stale buffer from the previous episode would produce a
        nonsense lagged lookup on the first steps of a new one."""
        self._emg_buffer: deque[np.ndarray] = deque(maxlen=self.buffer_size)
        self._t = 0
        self._last_heel_strike = {"r": None, "l": None}
        self._prev_contact = {"r": 0.0, "l": 0.0}
        self._half_period_estimate = self.default_half_period

    def _update_period_estimate(self, side: str, contact: float) -> None:
        rising_edge = self._prev_contact[side] < self.contact_threshold <= contact
        self._prev_contact[side] = contact
        if not rising_edge:
            return

        last = self._last_heel_strike[side]
        # Real stance phases dip mid-contact (e.g. observed 949 -> 153 -> 836
        # on a real rollout, never crossing back below threshold) but a
        # noisier one can briefly cross under contact_threshold and back --
        # without this, that registers as a second "heel-strike" a few steps
        # after the real one, corrupting the period estimate with spurious
        # short samples (confirmed empirically: a real rollout produced
        # 3-11 step "periods" from exactly this before this guard existed).
        # A stride's own half-period is ~40-70 steps at 100Hz, so 20 steps
        # is comfortably below any real strike-to-strike interval.
        if last is not None and self._t - last < self.refractory_period:
            return

        if last is not None:
            period = self._t - last
            sample = period / 2.0
            self._half_period_estimate = (
                (1 - self.smoothing) * self._half_period_estimate + self.smoothing * sample
            )
        self._last_heel_strike[side] = self._t

    def step(self, emg: np.ndarray, contact_r: float, contact_l: float) -> np.ndarray:
        """One step: detect heel-strike on each foot, update the running
        half-period estimate, buffer emg, and return the full
        (n_actuators,) a_hat -- _r rows from emg (now), _l rows from
        weights_r @ emg (now - half_period_estimate)."""
        self._update_period_estimate("r", contact_r)
        self._update_period_estimate("l", contact_l)

        self._emg_buffer.append(emg)
        self._t += 1

        lag = int(round(self._half_period_estimate))
        lag = min(lag, len(self._emg_buffer) - 1)
        emg_lagged = self._emg_buffer[-1 - lag]

        a_hat = self.weights @ emg
        for r_idx, l_idx in self._right_to_left.items():
            a_hat[l_idx] = self.weights[r_idx] @ emg_lagged
        return a_hat
