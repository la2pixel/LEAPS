"""Step 3: combine AB06 phase data + real OpenSim geometry into a full-body,
mechanically-consistent muscle activation trajectory for h0918.

Mapped muscles (gastroc_r, tib_ant_r, soleus_r, rect_fem_r, vasti_r,
hamstrings_r) get their activation from real AB06 EMG via a Hill-type force
model. Unmapped muscles (bifemsh_r, glut_max_r, iliopsoas_r -- h0918 has no
possible EMG substitute for these) get theirs from a per-phase QP: minimum
Sum(a_k^2) subject to matching the residual joint moment (real AB06
inverse-dynamics minus the mapped muscles' own contribution) at hip/knee/
ankle, 0<=a<=1.

Remaining simplification, stated explicitly (not hidden): rigid-tendon
assumption to get fiber length from muscle-tendon length (no elastic tendon
compliance). f_L and f_V are no longer approximations -- both come from
h0918's actual Millard2012EquilibriumMuscle ActiveForceLengthCurve/
ForceVelocityCurve objects, evaluated at AB06's real phase-normalized
fiber length/velocity (velocity from a real measured stride period, not
assumed). EMG is rectified + normalized via the project's own
process_trial_emg/compute_emg_normalization (getEMGNormalization.m
replication), fixed after a first pass that used raw unrectified EMG by
mistake (verified: raw Camargo EMG is AC-coupled, mean ~0, 50% negative --
averaging it directly across strides silently collapsed to noise).
This is a first-pass diagnostic, not a finished model.

Run with the `lalitha` conda env: python -m leaps.scripts.emg_driven_h0918_resolve
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

PHASE_DATA_PATH = "/home/nadinebadie/lalitha/LEAPS/results/emg_driven/h0918_ab06_phase_data.npz"
GEOMETRY_PATH = "/home/nadinebadie/lalitha/LEAPS/results/emg_driven/h0918_geometry.npz"
OUT_PATH = "/home/nadinebadie/lalitha/LEAPS/results/emg_driven/h0918_resolved.npz"

# H0918_MAP grouping (leaps/envs/emg_mapping.py), right leg only
MAPPED_MUSCLE_TO_EMG = {
    "gastroc_r": ["gastrocmed"],
    "tib_ant_r": ["tibialisanterior"],
    "soleus_r": ["soleus"],
    "rect_fem_r": ["rectusfemoris"],
    "vasti_r": ["vastusmedialis", "vastuslateralis"],
    "hamstrings_r": ["bicepsfemoris", "semitendinosus"],
}
TARGET_DOFS = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"]


def main() -> None:
    pd = np.load(PHASE_DATA_PATH, allow_pickle=True)
    geo = np.load(GEOMETRY_PATH, allow_pickle=True)

    muscle_names = list(geo["muscle_names"])
    dof_names = list(geo["dof_names"])
    moment_arms = geo["moment_arms"]  # (101, 18, 6)
    max_iso = geo["max_iso_force"]
    penn = geo["pennation"]
    # real Millard2012EquilibriumMuscle curves (this model's own, not a generic approximation)
    fl = geo["active_force_length"]  # (101, 18)
    fv = geo["force_velocity"]       # (101, 18)
    n_phases = fl.shape[0]

    emg_channels = list(pd["emg_channels"])
    emg_norm = pd["phase_emg"]  # (101, 11), already rectified + per-channel normalized to [0,1]

    mapped_names = list(MAPPED_MUSCLE_TO_EMG.keys())
    unmapped_names = [m for m in muscle_names if m.endswith("_r") and m not in mapped_names]
    print("mapped:", mapped_names)
    print("unmapped (no EMG substitute):", unmapped_names)

    mi = [muscle_names.index(m) for m in mapped_names]
    ui = [muscle_names.index(m) for m in unmapped_names]
    dofi = [dof_names.index(d) for d in TARGET_DOFS]

    # mapped-muscle EMG per phase (raw normalized [0,1], avg across grouped channels) -- not
    # yet the true activation, see calibration below
    u_mapped = np.zeros((n_phases, len(mapped_names)))
    for j, m in enumerate(mapped_names):
        cols = [emg_channels.index(c) for c in MAPPED_MUSCLE_TO_EMG[m]]
        u_mapped[:, j] = emg_norm[:, cols].mean(axis=1)

    phase_moments = pd["phase_moments"]  # (101, 3), Nm/kg -- AB06 target, right leg only
    ab06_mass = float(pd["ab06_mass_kg"])
    # h0918 model mass is 74.53 kg (from biomech_fidelity.ipynb, model.mass()) -- essentially
    # identical to AB06's 74.8 kg, so no Hof rescaling needed for this body (flagged, not skipped blindly)
    M_target = phase_moments * ab06_mass  # back to Nm at AB06's own mass; model mass is a 0.36% difference

    # EMG-to-activation calibration (Lloyd & Besier 2003): a(u,A) = (exp(A*u)-1)/(exp(A)-1),
    # A -> 0 reduces to a=u (the previous, uncalibrated linear assumption). Fit per-muscle A
    # against the ankle DOF specifically -- it's the one target 100% explained by mapped muscles
    # (gastroc_r/soleus_r/tib_ant_r are the only mapped muscles crossing it), so it's a clean,
    # unconfounded calibration target with no unmapped-muscle contribution to disentangle.
    ankle_di = dof_names.index("ankle_angle_r")
    ankle_ma = moment_arms[:, mi, ankle_di]  # (101, 6)
    ankle_crossers = [j for j in range(len(mapped_names)) if np.abs(ankle_ma[:, j]).mean() > 1e-4]
    print(f"ankle-crossing mapped muscles (calibration set): {[mapped_names[j] for j in ankle_crossers]}")

    # gastroc_r is biarticular (crosses knee too) -- calibrating it against the ankle alone is
    # an incomplete objective (one (A,EMD) pair being asked to be right at a joint it's never
    # checked against). Build its knee residual target (real knee moment minus the OTHER mapped
    # knee-crossers' raw-EMG contribution) so its fit has to respect both joints simultaneously.
    knee_di = dof_names.index("knee_angle_r")
    knee_ma = moment_arms[:, mi, knee_di]
    knee_crossers = [j for j in range(len(mapped_names)) if np.abs(knee_ma[:, j]).mean() > 1e-4]
    biarticular = [j for j in ankle_crossers if j in knee_crossers]
    print(f"knee-crossing mapped muscles: {[mapped_names[j] for j in knee_crossers]}, "
          f"biarticular (needs joint ankle+knee fit): {[mapped_names[j] for j in biarticular]}")
    other_knee_crossers = [j for j in knee_crossers if j not in biarticular]
    F_other_knee = u_mapped[:, other_knee_crossers] * max_iso[mi][other_knee_crossers][None, :] \
        * fl[:, mi][:, other_knee_crossers] * fv[:, mi][:, other_knee_crossers] \
        * np.cos(penn[mi][other_knee_crossers])[None, :]
    M_knee_residual_for_biarticular = M_target[:, TARGET_DOFS.index("knee_angle_r")] - \
        np.einsum("pm,pm->p", F_other_knee, knee_ma[:, other_knee_crossers])

    def lloyd_besier(u: np.ndarray, A: float) -> np.ndarray:
        if abs(A) < 1e-3:
            return u
        return (np.exp(A * u) - 1) / (np.exp(A) - 1)

    stride_period_s = float(pd["stride_period_s"])
    phase = np.linspace(0.0, 1.0, n_phases, endpoint=True)

    def shift_periodic(u: np.ndarray, shift_s: float) -> np.ndarray:
        """Shift a phase-normalized signal forward in time by shift_s seconds, wrapping
        around the gait cycle (EMG leads force by the electromechanical delay, so evaluating
        u at an EARLIER phase than the target moment's phase is the correct direction)."""
        shift_frac = shift_s / stride_period_s
        phase_ext = np.concatenate([phase - 1.0, phase, phase + 1.0])
        u_ext = np.tile(u, 3)
        return np.interp(phase - shift_frac, phase_ext, u_ext)

    n_c = len(ankle_crossers)
    bi_pos = [ankle_crossers.index(j) for j in biarticular]  # position of biarticular muscles within ankle_crossers

    def calibrated_force(params: np.ndarray) -> np.ndarray:
        A_vec, shift_vec = params[:n_c], params[n_c:]
        a_cal = np.stack(
            [lloyd_besier(shift_periodic(u_mapped[:, j], shift_vec[k]), A_vec[k]) for k, j in enumerate(ankle_crossers)],
            axis=1,
        )
        return a_cal * max_iso[mi][ankle_crossers][None, :] * fl[:, mi][:, ankle_crossers] \
            * fv[:, mi][:, ankle_crossers] * np.cos(penn[mi][ankle_crossers])[None, :]

    def ankle_predicted(params: np.ndarray) -> np.ndarray:
        return np.einsum("pm,pm->p", calibrated_force(params), ankle_ma[:, ankle_crossers])

    def knee_predicted_biarticular(params: np.ndarray) -> np.ndarray:
        F_cal = calibrated_force(params)
        return np.einsum("pm,pm->p", F_cal[:, bi_pos], knee_ma[:, biarticular])

    def ankle_calibration_obj(params: np.ndarray) -> float:
        ankle_err = M_target[:, TARGET_DOFS.index("ankle_angle_r")] - ankle_predicted(params)
        knee_err = M_knee_residual_for_biarticular - knee_predicted_biarticular(params)
        return float(np.sum(ankle_err**2) + np.sum(knee_err**2))

    # joint fit: per-muscle activation shape (A) AND per-muscle electromechanical delay (EMD,
    # bounded 0-150ms, the standard physiological range). Biarticular muscles (gastroc_r) are
    # now fit against BOTH the ankle and knee residual simultaneously -- fixes an incomplete
    # objective (was only checked against ankle despite crossing knee too), not just adding
    # another free parameter.
    x0 = np.concatenate([np.full(n_c, 0.5), np.full(n_c, 0.05)])
    bounds = [(-3, 3)] * n_c + [(0.0, 0.15)] * n_c
    cal_res = minimize(ankle_calibration_obj, x0=x0, bounds=bounds, method="L-BFGS-B")
    A_fit, shift_fit = cal_res.x[:n_c], cal_res.x[n_c:]
    for k, j in enumerate(ankle_crossers):
        tag = " (biarticular, ankle+knee fit)" if j in biarticular else ""
        print(f"  {mapped_names[j]}: A={A_fit[k]:.3f}, EMD={shift_fit[k]*1000:.1f} ms{tag}")

    x0_uncal = np.concatenate([np.full(n_c, 1e-6), np.zeros(n_c)])
    uncal_ankle_mismatch = np.abs(M_target[:, TARGET_DOFS.index("ankle_angle_r")] - ankle_predicted(x0_uncal)).mean()
    cal_ankle_mismatch = np.abs(M_target[:, TARGET_DOFS.index("ankle_angle_r")] - ankle_predicted(cal_res.x)).mean()
    uncal_knee_mismatch = np.abs(M_knee_residual_for_biarticular - knee_predicted_biarticular(x0_uncal)).mean()
    cal_knee_mismatch = np.abs(M_knee_residual_for_biarticular - knee_predicted_biarticular(cal_res.x)).mean()
    print(f"ankle |mismatch| before: {uncal_ankle_mismatch:.2f} Nm, after: {cal_ankle_mismatch:.2f} Nm")
    print(f"gastroc's knee-residual |mismatch| before: {uncal_knee_mismatch:.2f} Nm, after: {cal_knee_mismatch:.2f} Nm")

    a_mapped = u_mapped.copy()
    for k, j in enumerate(ankle_crossers):
        a_mapped[:, j] = lloyd_besier(shift_periodic(u_mapped[:, j], shift_fit[k]), A_fit[k])

    # Second calibration group: rect_fem_r, vasti_r, hamstrings_r (hip/knee crossers, no ankle
    # role). Same joint-DOF logic as gastroc above -- calibrate against BOTH joints each muscle
    # actually crosses, not just one. Target excludes gastroc's contribution (already calibrated,
    # known) and the unmapped hip/knee muscles' contribution (iliopsoas_r/glut_max_r/bifemsh_r --
    # not yet solved at this point in the script, so treated as 0 here, same approximation this
    # script already made for gastroc's knee target above -- not a fully joint solve).
    hip_di = dof_names.index("hip_flexion_r")
    hip_ma = moment_arms[:, mi, hip_di]
    group2 = [mapped_names.index(m) for m in ("rect_fem_r", "vasti_r", "hamstrings_r")]
    group2_hip = [j for j in group2 if np.abs(hip_ma[:, j]).mean() > 1e-4]
    group2_knee = [j for j in group2 if np.abs(knee_ma[:, j]).mean() > 1e-4]
    print(f"group 2 (hip/knee, no ankle role): {[mapped_names[j] for j in group2]}, "
          f"hip crossers: {[mapped_names[j] for j in group2_hip]}, knee crossers: {[mapped_names[j] for j in group2_knee]}")

    gastroc_pos = ankle_crossers.index(mapped_names.index("gastroc_r"))
    F_gastroc_final = calibrated_force(cal_res.x)[:, gastroc_pos]
    M_knee_target_g2 = M_target[:, TARGET_DOFS.index("knee_angle_r")] - F_gastroc_final * knee_ma[:, mapped_names.index("gastroc_r")]
    M_hip_target_g2 = M_target[:, TARGET_DOFS.index("hip_flexion_r")]

    n_g2 = len(group2)

    def calibrated_force_g2(params: np.ndarray) -> np.ndarray:
        A_vec, shift_vec = params[:n_g2], params[n_g2:]
        a_cal = np.stack(
            [lloyd_besier(shift_periodic(u_mapped[:, j], shift_vec[k]), A_vec[k]) for k, j in enumerate(group2)],
            axis=1,
        )
        return a_cal * max_iso[mi][group2][None, :] * fl[:, mi][:, group2] \
            * fv[:, mi][:, group2] * np.cos(penn[mi][group2])[None, :]

    def g2_obj(params: np.ndarray) -> float:
        F_cal = calibrated_force_g2(params)
        hip_pos = [group2.index(j) for j in group2_hip]
        knee_pos = [group2.index(j) for j in group2_knee]
        hip_pred = np.einsum("pm,pm->p", F_cal[:, hip_pos], hip_ma[:, group2_hip])
        knee_pred = np.einsum("pm,pm->p", F_cal[:, knee_pos], knee_ma[:, group2_knee])
        return float(np.sum((M_hip_target_g2 - hip_pred) ** 2) + np.sum((M_knee_target_g2 - knee_pred) ** 2))

    x0_g2 = np.concatenate([np.full(n_g2, 0.5), np.full(n_g2, 0.05)])
    bounds_g2 = [(-3, 3)] * n_g2 + [(0.0, 0.15)] * n_g2
    cal_res_g2 = minimize(g2_obj, x0=x0_g2, bounds=bounds_g2, method="L-BFGS-B")
    A_fit_g2, shift_fit_g2 = cal_res_g2.x[:n_g2], cal_res_g2.x[n_g2:]
    for k, j in enumerate(group2):
        print(f"  {mapped_names[j]}: A={A_fit_g2[k]:.3f}, EMD={shift_fit_g2[k]*1000:.1f} ms")

    for k, j in enumerate(group2):
        a_mapped[:, j] = lloyd_besier(shift_periodic(u_mapped[:, j], shift_fit_g2[k]), A_fit_g2[k])
    print("all 6 mapped muscles now calibrated (shape + per-muscle EMD)")

    F_mapped = a_mapped * max_iso[mi][None, :] * fl[:, mi] * fv[:, mi] * np.cos(penn[mi])[None, :]  # (101, 6)
    M_mapped = np.einsum("pm,pmd->pd", F_mapped, moment_arms[:, mi][:, :, dofi])  # (101, 3)
    M_residual = M_target - M_mapped

    # Only constrain the QP on DOFs at least one unmapped muscle actually crosses -- ankle has
    # zero moment arm for bifemsh_r/glut_max_r/iliopsoas_r in this model (none of them cross the
    # ankle), so an equality constraint there is structurally infeasible by construction, not a
    # solver failure. Report it separately as a pure force-model diagnostic instead.
    unmapped_ma = moment_arms[:, ui][:, :, dofi]  # (101, n_unmapped, 3)
    solvable_dof_mask = np.abs(unmapped_ma).mean(axis=(0, 1)) > 1e-4
    solvable_dofs = [d for d, keep in zip(TARGET_DOFS, solvable_dof_mask) if keep]
    unresolvable_dofs = [d for d, keep in zip(TARGET_DOFS, solvable_dof_mask) if not keep]
    print(f"QP-solvable DOFs (>=1 unmapped muscle crosses them): {solvable_dofs}")
    print(f"structurally unresolvable DOFs (mapped-only by construction): {unresolvable_dofs}")

    solvable_idx = [i for i, keep in enumerate(solvable_dof_mask) if keep]

    a_unmapped = np.zeros((n_phases, len(unmapped_names)))
    residual_norms = np.zeros(n_phases)
    for p in range(n_phases):
        C = (max_iso[ui] * fl[p, ui] * fv[p, ui] * np.cos(penn[ui]))[None, :] * unmapped_ma[p][:, solvable_idx].T
        b = M_residual[p][solvable_idx]

        def obj(a):
            return float(np.sum(a**2))

        def obj_grad(a):
            return 2 * a

        cons = {"type": "eq", "fun": lambda a, C=C, b=b: C @ a - b}
        res = minimize(
            obj, x0=np.full(len(unmapped_names), 0.1), jac=obj_grad,
            bounds=[(0, 1)] * len(unmapped_names), constraints=[cons], method="SLSQP",
        )
        a_unmapped[p] = res.x
        residual_norms[p] = np.linalg.norm(C @ res.x - b)

    print(f"mean |constraint residual| after solve (solvable DOFs only): {residual_norms.mean():.4f} Nm (0 = exact match)")
    for d in unresolvable_dofs:
        di = TARGET_DOFS.index(d)
        print(f"  [{d}] mapped-only mismatch (target - mapped contribution): "
              f"mean {M_residual[:,di].mean():.2f} Nm, |mean| {np.abs(M_residual[:,di]).mean():.2f} Nm "
              f"(vs target scale |mean| {np.abs(M_target[:,di]).mean():.2f} Nm)")
    for j, m in enumerate(unmapped_names):
        print(f"  {m}: activation range [{a_unmapped[:,j].min():.3f}, {a_unmapped[:,j].max():.3f}], "
              f"mean {a_unmapped[:,j].mean():.3f}, frac saturated (>0.99 or <0.01): "
              f"{np.mean((a_unmapped[:,j]>0.99)|(a_unmapped[:,j]<0.01)):.2f}")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        mapped_names=np.array(mapped_names), a_mapped=a_mapped,
        unmapped_names=np.array(unmapped_names), a_unmapped=a_unmapped,
        M_target=M_target, M_mapped=M_mapped, M_residual=M_residual,
        residual_norms=residual_norms,
    )
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
