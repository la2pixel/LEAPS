"""Unit tests for DepContentPrior: proves masking is exact (not just
usually zero), DEP's own C matrix actually self-organizes over steps
(not frozen at its zero init), and the shape contract holds. No
simulator needed -- feeds synthetic fiber-length traces directly."""

import numpy as np
import pytest

from leaps.envs.dep_content_prior import DepContentPrior

N_ACTUATORS = 6
MAPPED_MASK = np.array([True, False, True, False, True, False])


def _fiber_lengths(t):
    """Varying, non-degenerate signal -- DEP's C matrix only grows away
    from zero when the sensor stream actually changes step to step
    (_compute_C's chi = x - xx is 0 forever on a constant input)."""
    phase = np.arange(N_ACTUATORS) * 0.3
    return 0.4 + 0.1 * np.sin(0.2 * t + phase)


def test_masked_actuators_stay_exactly_zero():
    prior = DepContentPrior(MAPPED_MASK, N_ACTUATORS)
    for t in range(80):
        a_hat = prior.step(_fiber_lengths(t))
        assert np.all(a_hat[~MAPPED_MASK] == 0.0)


def test_shape_contract():
    prior = DepContentPrior(MAPPED_MASK, N_ACTUATORS)
    a_hat = prior.step(_fiber_lengths(0))
    assert a_hat.shape == (N_ACTUATORS,)


def test_mapped_actuators_become_nonzero_past_cold_start():
    prior = DepContentPrior(MAPPED_MASK, N_ACTUATORS)
    a_hat = None
    for t in range(80):
        a_hat = prior.step(_fiber_lengths(t))
    # DEP's C matrix starts at exactly zero (dep_controller.py's _reset)
    # and only produces nonzero action once _learn_controller has run --
    # 80 steps is comfortably past that (needs > 2 + time_dist, default
    # time_dist=5).
    assert np.any(a_hat[MAPPED_MASK] != 0.0)


def test_c_matrix_actually_self_organizes():
    """Not just nonzero once -- genuinely still evolving, proving this is
    an online adaptive process, not a frozen-after-warmup one."""
    prior = DepContentPrior(MAPPED_MASK, N_ACTUATORS)
    for t in range(40):
        prior.step(_fiber_lengths(t))
    c_at_40 = prior.dep.C_norm.clone()
    for t in range(40, 80):
        prior.step(_fiber_lengths(t))
    c_at_80 = prior.dep.C_norm.clone()
    assert not np.allclose(c_at_40.numpy(), c_at_80.numpy())


def test_constant_input_keeps_c_at_zero():
    """Sanity check on the fixture itself: a genuinely unchanging sensor
    stream should never move C away from its zero init (chi = x - xx = 0
    every step) -- confirms _fiber_lengths' variation is what's actually
    driving the two tests above, not some other side effect."""
    prior = DepContentPrior(MAPPED_MASK, N_ACTUATORS)
    constant = _fiber_lengths(0)
    for _ in range(80):
        a_hat = prior.step(constant)
    assert np.all(a_hat == 0.0)


def test_no_reset_hook_needed_state_persists():
    """DEP's C matrix is meant to persist across the whole training run
    (no per-episode reset, matching every existing dep_factory agent's
    reset() being a no-op) -- confirms DepContentPrior has no reset()
    method to accidentally call."""
    assert not hasattr(DepContentPrior, "reset")
