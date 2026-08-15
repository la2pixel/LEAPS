"""Unit tests for PhaseMirror: proves the lag is real and correctly
targeted (not just "different from static"), the period estimate
converges, the cold-start fallback works, and reset clears state."""

import numpy as np
import pytest

from leaps.envs.phase_mirror import PhaseMirror

ACTUATOR_NAMES = ["muscle_r", "muscle_l", "unmapped_r", "unmapped_l"]
N_CHANNELS = 11


def _weights_right_only():
    # muscle_r reads EMG channel 0 with weight 1; unmapped_r/l stay zero,
    # same shape a mirror_left=False EMGToMuscleMapper would produce.
    w = np.zeros((4, N_CHANNELS))
    w[0, 0] = 1.0
    return w


def _emg(t):
    """A distinctive, non-repeating signal per step so a returned a_hat
    value can be traced back to exactly which timestep's emg produced it."""
    e = np.zeros(N_CHANNELS)
    e[0] = t
    return e


def test_right_to_left_mapping_built_correctly():
    pm = PhaseMirror(_weights_right_only(), ACTUATOR_NAMES)
    assert pm._right_to_left == {0: 1}  # muscle_r (idx 0) -> muscle_l (idx 1)


def test_cold_start_uses_default_half_period():
    pm = PhaseMirror(_weights_right_only(), ACTUATOR_NAMES, default_half_period=10.0)
    # No heel-strikes observed yet (contact stays at 0) -- lag should be
    # the default, clamped to however much buffer exists so far.
    a_hat = pm.step(_emg(0), contact_r=0.0, contact_l=0.0)
    assert a_hat[1] == 0.0  # only 1 step buffered, lag clamped to 0 -> emg[0][ch0]=0 -> muscle_l = 1*0
    for t in range(1, 5):
        a_hat = pm.step(_emg(t), contact_r=0.0, contact_l=0.0)
    # at t=4, lag = min(round(10), len(buffer)-1=4) = 4 -> emg_lagged = emg(0)
    assert a_hat[1] == pytest.approx(0.0)


def test_half_period_converges_to_true_value():
    """Square-wave contact trace, period 50 -> true half-period is 25."""
    pm = PhaseMirror(_weights_right_only(), ACTUATOR_NAMES, default_half_period=50.0)
    period = 50
    n_steps = period * 24  # ~24 cycles, enough for the smoothing=0.2 EMA to converge
    for t in range(n_steps):
        contact_r = 1.0 if (t % period) < period // 2 else 0.0
        contact_l = 1.0 if ((t + period // 2) % period) < period // 2 else 0.0
        pm.step(_emg(t), contact_r, contact_l)
    assert pm._half_period_estimate == pytest.approx(25.0, abs=2.0)


def test_lag_is_real_and_correctly_targeted():
    """The actual point of this mechanism: a_hat_l at step t must equal
    weights_r @ emg[t - half_period], not emg[t] (that would just be the
    old static mirror) and not some other arbitrary lag."""
    pm = PhaseMirror(_weights_right_only(), ACTUATOR_NAMES, default_half_period=10.0, buffer_size=200)
    emgs = []
    a_hats = []
    for t in range(60):
        e = _emg(t)
        emgs.append(e)
        # keep contact at 0 throughout -- no heel-strikes, so lag stays
        # pinned at the default (10), making the expected lagged index
        # fully predictable for this test.
        a_hats.append(pm.step(e, contact_r=0.0, contact_l=0.0))

    # after step 10 (buffer has >10 entries), lag is exactly 10
    t = 30
    expected_emg_lagged = emgs[t - 10]  # step() is 1-indexed internally by call order; emgs[t] is the t-th call (0-indexed)
    expected_a_hat_l = _weights_right_only()[0] @ expected_emg_lagged
    assert a_hats[t][1] == pytest.approx(expected_a_hat_l)
    # and it must NOT equal the static (same-step) value, confirming a
    # real lag is happening, not an accidental no-op
    same_step_value = _weights_right_only()[0] @ emgs[t]
    assert a_hats[t][1] != pytest.approx(same_step_value)


def test_right_side_unaffected_by_mirroring():
    pm = PhaseMirror(_weights_right_only(), ACTUATOR_NAMES)
    for t in range(20):
        a_hat = pm.step(_emg(t), contact_r=0.0, contact_l=0.0)
    assert a_hat[0] == pytest.approx(_weights_right_only()[0] @ _emg(19))


def test_unmapped_actuators_stay_zero():
    pm = PhaseMirror(_weights_right_only(), ACTUATOR_NAMES)
    a_hat = pm.step(_emg(0), contact_r=0.0, contact_l=0.0)
    assert a_hat[2] == 0.0 and a_hat[3] == 0.0


def test_reset_clears_state():
    pm = PhaseMirror(_weights_right_only(), ACTUATOR_NAMES, default_half_period=10.0)
    for t in range(60):
        pm.step(_emg(t), contact_r=float(t % 2), contact_l=float((t + 1) % 2))
    assert pm._t == 60
    assert len(pm._emg_buffer) > 0

    pm.reset()
    assert pm._t == 0
    assert len(pm._emg_buffer) == 0
    assert pm._half_period_estimate == 10.0
    assert pm._last_heel_strike == {"r": None, "l": None}
