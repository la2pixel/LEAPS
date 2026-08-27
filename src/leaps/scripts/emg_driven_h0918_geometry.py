"""Step 2 of the EMG-driven h0918 pipeline: real OpenSim moment arms + muscle
lengths for every muscle, at AB06's own phase-normalized joint-angle
trajectory (produced by emg_driven_h0918_phase_data.py).

Uses the actual H0918v2j.osim source model (confirmed to match the .hfd
used by sconewalk_h0918-v1) via the standalone OpenSim Python API -- the
installed Hyfydy sconepy binding has no moment-arm/PCSA accessor at all
(verified separately), so this MUST run in an env with the real `opensim`
package, not the `lalitha` conda env.

Left leg is set symmetrically to the right leg's own angles (no phase
offset) for this first pass -- a known simplification, not a claim that
real gait is bilaterally synchronous. Pelvis coordinates left at 0.

Run with: /home/nadinebadie/miniforge3/envs/myoconverter/bin/python \
    leaps/scripts/emg_driven_h0918_geometry.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import opensim as osim

MODEL_PATH = "/home/nadinebadie/lalitha/sconegym/sconegym/data-v1/H0918v2j.osim"
PHASE_DATA_PATH = "/home/nadinebadie/lalitha/LEAPS/results/emg_driven/h0918_ab06_phase_data.npz"
OUT_PATH = "/home/nadinebadie/lalitha/LEAPS/results/emg_driven/h0918_geometry.npz"

# right-leg DOF (from phase data) -> its mirrored left-leg DOF name
RIGHT_TO_LEFT_DOF = {
    "hip_flexion_r": "hip_flexion_l",
    "knee_angle_r": "knee_angle_l",
    "ankle_angle_r": "ankle_angle_l",
}


def main() -> None:
    data = np.load(PHASE_DATA_PATH, allow_pickle=True)
    dofs = list(data["dofs"])
    phase_angles = data["phase_angles"]  # (101, 3) radians, right leg
    n_phases = phase_angles.shape[0]
    stride_period_s = float(data["stride_period_s"])
    dt = stride_period_s / (n_phases - 1)

    model = osim.Model(MODEL_PATH)
    state = model.initSystem()
    coords = model.getCoordinateSet()
    muscles = model.getMuscles()
    n_muscles = muscles.getSize()
    muscle_names = [muscles.get(i).getName() for i in range(n_muscles)]

    all_dof_names = dofs + [RIGHT_TO_LEFT_DOF[d] for d in dofs]

    moment_arms = np.zeros((n_phases, n_muscles, len(all_dof_names)))
    lengths = np.zeros((n_phases, n_muscles))
    max_iso_force = np.array([muscles.get(i).getMaxIsometricForce() for i in range(n_muscles)])
    opt_fiber_len = np.array([muscles.get(i).getOptimalFiberLength() for i in range(n_muscles)])
    tendon_slack_len = np.array([muscles.get(i).getTendonSlackLength() for i in range(n_muscles)])
    pennation = np.array([muscles.get(i).getPennationAngleAtOptimalFiberLength() for i in range(n_muscles)])
    max_contraction_vel = np.array([muscles.get(i).getMaxContractionVelocity() for i in range(n_muscles)])

    # real Millard2012EquilibriumMuscle curves -- confirmed present (not the generic Gaussian/
    # f_V=1 approximation used in the first pass), one downcast + curve handle per muscle
    millard = [osim.Millard2012EquilibriumMuscle.safeDownCast(muscles.get(i)) for i in range(n_muscles)]
    assert all(mm is not None for mm in millard), "not all h0918 muscles are Millard2012EquilibriumMuscle"
    afl_curves = [mm.getActiveForceLengthCurve() for mm in millard]
    fv_curves = [mm.getForceVelocityCurve() for mm in millard]

    for p in range(n_phases):
        for j, d in enumerate(dofs):
            coords.get(d).setValue(state, float(phase_angles[p, j]))
            coords.get(RIGHT_TO_LEFT_DOF[d]).setValue(state, float(phase_angles[p, j]))
        model.assemble(state)
        model.realizePosition(state)
        for i in range(n_muscles):
            m = muscles.get(i)
            lengths[p, i] = m.getLength(state)
            for k, d in enumerate(all_dof_names):
                moment_arms[p, i, k] = m.computeMomentArm(state, coords.get(d))
        if p % 20 == 0:
            print(f"phase {p}/{n_phases} done")

    # rigid-tendon fiber length -> normalized length + velocity, then evaluate this model's
    # own real active-force-length and force-velocity curves (replaces the earlier Gaussian
    # f_L approximation and f_V=1 quasi-static placeholder)
    fiber_len = (lengths - tendon_slack_len[None, :]) / np.cos(pennation[None, :])
    l_tilde = fiber_len / opt_fiber_len[None, :]
    v_fiber = np.gradient(fiber_len, dt, axis=0)  # m/s; shortening (concentric) is negative
    v_max_abs = max_contraction_vel * opt_fiber_len  # m/s
    v_tilde = v_fiber / v_max_abs[None, :]

    active_force_length = np.zeros((n_phases, n_muscles))
    force_velocity = np.zeros((n_phases, n_muscles))
    for p in range(n_phases):
        for i in range(n_muscles):
            active_force_length[p, i] = afl_curves[i].calcValue(osim.Vector(1, float(l_tilde[p, i])))
            force_velocity[p, i] = fv_curves[i].calcValue(osim.Vector(1, float(v_tilde[p, i])))

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        muscle_names=np.array(muscle_names),
        dof_names=np.array(all_dof_names),
        moment_arms=moment_arms,       # (101, n_muscles, n_dofs)
        lengths=lengths,               # (101, n_muscles)
        max_iso_force=max_iso_force,
        opt_fiber_len=opt_fiber_len,
        tendon_slack_len=tendon_slack_len,
        pennation=pennation,
        max_contraction_vel=max_contraction_vel,
        l_tilde=l_tilde,
        v_tilde=v_tilde,
        active_force_length=active_force_length,   # f_L(phase, muscle), real Millard curve
        force_velocity=force_velocity,              # f_V(phase, muscle), real Millard curve
    )
    print(f"saved -> {OUT_PATH}")
    print(f"active_force_length range: [{active_force_length.min():.3f}, {active_force_length.max():.3f}]")
    print(f"force_velocity range: [{force_velocity.min():.3f}, {force_velocity.max():.3f}]")

    # sanity check: hamstrings_r moment arm on knee/hip should be a few cm, sign flexor-consistent
    hi = muscle_names.index("hamstrings_r")
    ki = all_dof_names.index("knee_angle_r")
    hpi = all_dof_names.index("hip_flexion_r")
    print(f"hamstrings_r moment arm: knee {moment_arms[:,hi,ki].mean():.4f} m, hip {moment_arms[:,hi,hpi].mean():.4f} m "
          f"(range over cycle: knee [{moment_arms[:,hi,ki].min():.4f},{moment_arms[:,hi,ki].max():.4f}])")


if __name__ == "__main__":
    main()
