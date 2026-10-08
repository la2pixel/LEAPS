"""DepContentPrior: masking is exact, DEP's C matrix self-organizes, shapes
hold. Uses synthetic fiber-length traces, no simulator."""

import numpy as np

from leaps.envs.dep_content_prior import DepContentPrior

N_ACTUATORS = 6
MAPPED_MASK = np.array([True, False, True, False, True, False])


def _fiber_lengths(t):
    """Varying signal: C only moves away from zero when the input changes."""
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
    # C starts at zero and is nonzero once learning has run
    assert np.any(a_hat[MAPPED_MASK] != 0.0)


def test_c_matrix_actually_self_organizes():
    """C keeps evolving after warm-up."""
    prior = DepContentPrior(MAPPED_MASK, N_ACTUATORS)
    for t in range(40):
        prior.step(_fiber_lengths(t))
    c_at_40 = prior.dep.C_norm.clone()
    for t in range(40, 80):
        prior.step(_fiber_lengths(t))
    c_at_80 = prior.dep.C_norm.clone()
    assert not np.allclose(c_at_40.numpy(), c_at_80.numpy())


def test_constant_input_keeps_c_at_zero():
    """A constant input never moves C from its zero init."""
    prior = DepContentPrior(MAPPED_MASK, N_ACTUATORS)
    constant = _fiber_lengths(0)
    for _ in range(80):
        a_hat = prior.step(constant)
    assert np.all(a_hat == 0.0)


def test_no_reset_hook_needed_state_persists():
    """C persists across episodes: there is no reset()."""
    assert not hasattr(DepContentPrior, "reset")
