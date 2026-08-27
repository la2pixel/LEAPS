"""Adaptive cubic muscle-effort penalty, following Schumacher et al. 2025
(arXiv:2309.02976, Eq 3 + Algorithm 1).

Out of scope for reporting -- this project reports onlyvelrew and full
only. `AdaptiveEnergyBuffer` (custom_replay_buffers/action_cost_replay.py)
already applies this same alpha(t)*a**3 term to every config regardless;
this wrapper duplicates it at env.step() time instead of replay time and
isn't used in the two reported variants.
"""

from __future__ import annotations

import gym
import numpy as np


class AdaptiveEffortRewardWrapper(gym.Wrapper):
    """Adds an adaptively-weighted cubic muscle-effort penalty to the base
    env's reward, following Schumacher et al. 2025's Algorithm 1.

    threshold: performance level (in this env's own reward units) above
        which the agent is considered to be reliably solving the task.
    smoothing (beta): EMA decay for both the performance tracker (r_mean)
        and the "how long has performance been high" tracker (c_mean).
    delta_alpha_init: initial step size for alpha's per-episode increment.
    decay (lambda): shrinks delta_alpha once performance has been
        consistently high (stabilizes alpha instead of letting it oscillate
        forever once the task is solved).
    """

    def __init__(
        self,
        env,
        threshold: float,
        smoothing: float = 0.8,
        delta_alpha_init: float = 9e-4,
        decay: float = 0.9,
        alpha_init: float = 0.0,
    ):
        super().__init__(env)
        self.threshold = threshold
        self.beta = smoothing
        self.delta_alpha = delta_alpha_init
        self.decay = decay
        self.alpha = alpha_init
        self.r_mean = 0.0
        self.c_mean = 0.0
        self._episode_return = 0.0

    def reset(self, **kwargs):
        self._episode_return = 0.0
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._episode_return += reward  # pre-penalty task return, see module docstring point 1
        self._step_count += 1

        acts = self.unwrapped.model.muscle_activation_array()
        effort_penalty = self.alpha * float(np.mean(acts ** 3))
        shaped_reward = reward - effort_penalty

        # gaitgym.py's own `done` (_get_done()) fires ONLY on falling -- it
        # never checks step count, so a full-length successful walk episode
        # returns done=False for all 1000 steps. The horizon is enforced
        # OUTSIDE this wrapped env chain, by custom_distributed.py's
        # Sequential/Parallel classes (their own `_max_episode_steps`
        # counter), which this wrapper has no visibility into. Relying on
        # `done` alone would mean _update_alpha only ever fires on FALL
        # episodes (short, low return) and never on successful full-length
        # walks (long, high return) -- exactly backwards from what the
        # adaptation needs to detect "performance is reliably high". Track
        # the horizon ourselves instead, reading the same attribute
        # custom_distributed.py itself uses, so both layers agree on what
        # "episode over" means. Accessed directly (no getattr fallback) so
        # this fails loudly, not silently, if ever wrapped around a
        # non-gaitgym env that lacks the attribute.
        max_steps = self.unwrapped._max_episode_steps
        episode_over = done or self._step_count >= max_steps

        if episode_over:
            self._update_alpha(self._episode_return)

        info = dict(info)
        info["adaptive_alpha"] = self.alpha
        info["effort_penalty"] = effort_penalty
        return obs, shaped_reward, done, info

    def _update_alpha(self, episode_return: float) -> None:
        s_mean = self.c_mean  # slow "has performance been high for a while" indicator, from *before* this update
        self.r_mean = self.beta * self.r_mean + (1 - self.beta) * episode_return

        if self.r_mean > self.threshold and s_mean < 0.5:
            self.delta_alpha *= self.decay  # newly high -- slow down further adaptation
        elif self.r_mean > self.threshold and s_mean >= 0.5:
            self.alpha += self.delta_alpha  # high for a while -- push the constraint harder
        else:
            self.alpha = max(0.0, self.alpha - self.delta_alpha)  # not solving the task well enough -- relax

        c_target = 1.0 if self.r_mean > self.threshold else 0.0
        self.c_mean = self.beta * self.c_mean + (1 - self.beta) * c_target
