"""Phase-normalize AB06's real gon/id/emg data for the EMG-driven h0918 pipeline.

Step 1 of the physics-informed decoder framework: produce one clean,
101-point (N_GAIT_POINTS), phase-normalized gait-cycle trajectory for
- joint angles (hip/knee/ankle, right leg) -> feeds moment-arm lookup
- joint moments (same 3 DOFs), Nm/kg -> M_target for the redundancy solve
- the 11 raw EMG channels -> a_i(phi) for the 6 h0918-mapped muscles

Robustness pass: pools strides across ALL 8 available AB06 treadmill trial
blocks (treadmill_01..08), not just one trial file -- the first pass used
only trial 01 (31 strides), which is exactly the single-instance risk this
project's own standing lesson warns against. Also keeps the full per-stride
stack (not just the mean), so a downstream bootstrap/stability check on the
calibration fit is possible without re-parsing the raw data.

Reuses the exact same speed-matching convention as
LEAPS/notebooks/biomech_fidelity.ipynb (TARGET_SPEED=1.3 m/s, the closest
AB06 plateau to h0918 policies' self-selected speed) so results are
directly comparable to that notebook's existing kinematics/kinetics numbers.

Run with the `lalitha` conda env: python -m leaps.scripts.emg_driven_h0918_phase_data
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from leaps.data.loaders import load_mat_table
from leaps.data.processing import (
    N_GAIT_POINTS,
    compute_emg_normalization,
    find_stride_intervals,
    normalize_emg,
    process_trial_emg,
    time_normalize_stride,
)

CAMARGO_AB06 = Path("/home/nadinebadie/lalitha/datasets/camargo/AB06/10_09_18/treadmill")
TARGET_SPEED = 1.3  # m/s, matches biomech_fidelity.ipynb's h0918 comparison
AB06_MASS_KG = 74.8

# gon (goniometer) name -> SCONE/OpenSim DOF name, right leg only (gon has no _r/_l suffix)
GON_TO_DOF = {
    "hip_sagittal": "hip_flexion_r",
    "knee_sagittal": "knee_angle_r",
    "ankle_sagittal": "ankle_angle_r",
}
DOFS = list(GON_TO_DOF.values())


def _kept_strides_and_durations(header, data, intervals, speed_header, speed):
    kept, durations = [], []
    for t0, t1 in intervals:
        mask = (speed_header >= t0) & (speed_header < t1)
        s = speed[mask]
        if len(s) == 0 or len(np.unique(np.round(s, 2))) > 2:
            continue  # skip strides straddling a speed change
        if not np.isclose(np.mean(s), TARGET_SPEED, atol=0.05):
            continue
        seg = data[(header >= t0) & (header < t1)]
        if len(seg) > 10:
            kept.append(time_normalize_stride(seg))
            durations.append(t1 - t0)
    return kept, durations


def _available_trials() -> list[str]:
    # filenames are treadmill_<trial>_01.mat -- strip only the trailing "_01" block suffix,
    # not any "_01" substring (str.replace would also eat trial 01's own number)
    return sorted(p.stem.rsplit("_01", 1)[0] for p in (CAMARGO_AB06 / "emg").glob("treadmill_*_01.mat"))


def main() -> None:
    trials = _available_trials()
    print(f"pooling across {len(trials)} trial blocks: {trials}")

    all_angle_strides, all_moment_strides, all_emg_strides, all_durations = [], [], [], []
    per_trial_counts = {}

    for trial in trials:
        gon = load_mat_table(CAMARGO_AB06 / "gon" / f"{trial}_01.mat")
        idd = load_mat_table(CAMARGO_AB06 / "id" / f"{trial}_01.mat")
        emg = load_mat_table(CAMARGO_AB06 / "emg" / f"{trial}_01.mat")
        speed = load_mat_table(CAMARGO_AB06 / "conditions" / f"{trial}_01.mat")
        cycles = load_mat_table(CAMARGO_AB06 / "gcRight" / f"{trial}_01.mat")

        intervals = find_stride_intervals(cycles["Header"], cycles["HeelStrike"])

        angle_data = np.column_stack([gon[g] for g in GON_TO_DOF]) * np.pi / 180.0
        angle_strides, durations = _kept_strides_and_durations(
            gon["Header"], angle_data, intervals, speed["Header"], speed["Speed"])

        moment_cols = [f"{d}_moment" for d in DOFS]
        moment_data = np.column_stack([idd[c] for c in moment_cols]) / AB06_MASS_KG
        moment_strides, _ = _kept_strides_and_durations(
            idd["Header"], moment_data, intervals, speed["Header"], speed["Speed"])

        emg_channels = [c for c in emg if c != "Header"]
        rect_strides, emg_intervals = process_trial_emg(emg, cycles)
        emg_strides = []
        for stride, (t0, t1) in zip(rect_strides, emg_intervals):
            mask = (speed["Header"] >= t0) & (speed["Header"] < t1)
            s = speed["Speed"][mask]
            if len(s) == 0 or len(np.unique(np.round(s, 2))) > 2:
                continue
            if not np.isclose(np.mean(s), TARGET_SPEED, atol=0.05):
                continue
            emg_strides.append(time_normalize_stride(stride))

        per_trial_counts[trial] = (len(angle_strides), len(moment_strides), len(emg_strides))
        all_angle_strides += angle_strides
        all_moment_strides += moment_strides
        all_emg_strides += emg_strides
        all_durations += durations

    print("kept strides per trial (angles, moments, emg):")
    for t, c in per_trial_counts.items():
        print(f"  {t}: {c}")
    if not all_angle_strides or not all_moment_strides or not all_emg_strides:
        raise RuntimeError(f"no strides found at {TARGET_SPEED} m/s across any trial")

    stride_period_s = float(np.mean(all_durations))
    print(f"pooled: {len(all_angle_strides)} angle, {len(all_moment_strides)} moment, "
          f"{len(all_emg_strides)} emg strides @ {TARGET_SPEED} m/s "
          f"(vs 31/31/31 from trial 01 alone in the first pass)")
    print(f"mean stride period @ {TARGET_SPEED} m/s: {stride_period_s:.3f} s")

    angle_strides_arr = np.stack(all_angle_strides)   # (n, 101, 3)
    moment_strides_arr = np.stack(all_moment_strides)  # (n, 101, 3)
    emg_strides_arr = np.stack(all_emg_strides)        # (n, 101, 11)

    phase_angles = angle_strides_arr.mean(axis=0)
    phase_moments = moment_strides_arr.mean(axis=0)

    min_vals, max_vals = compute_emg_normalization(list(emg_strides_arr))
    phase_emg_raw = emg_strides_arr.mean(axis=0)
    phase_emg = normalize_emg(phase_emg_raw, min_vals, max_vals)
    emg_channels = [c for c in load_mat_table(CAMARGO_AB06 / "emg" / f"{trials[0]}_01.mat") if c != "Header"]

    for i, d in enumerate(DOFS):
        print(f"  {d}: angle range [{np.degrees(phase_angles[:,i].min()):.1f}, {np.degrees(phase_angles[:,i].max()):.1f}] deg, "
              f"moment range [{phase_moments[:,i].min():.3f}, {phase_moments[:,i].max():.3f}] Nm/kg")

    out_dir = Path("/home/nadinebadie/lalitha/LEAPS/results/emg_driven")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "h0918_ab06_phase_data.npz",
        dofs=np.array(DOFS),
        phase_angles=phase_angles,
        emg_channels=np.array(emg_channels),
        phase_moments=phase_moments,
        phase_emg=phase_emg,
        n_gait_points=N_GAIT_POINTS,
        target_speed=TARGET_SPEED,
        ab06_mass_kg=AB06_MASS_KG,
        stride_period_s=stride_period_s,
        n_strides_pooled=len(all_angle_strides),
        # full per-stride stacks, kept for a future bootstrap/stability check
        angle_strides=angle_strides_arr,
        moment_strides=moment_strides_arr,
        emg_strides=emg_strides_arr,
    )
    print(f"saved -> {out_dir / 'h0918_ab06_phase_data.npz'}")


if __name__ == "__main__":
    main()
