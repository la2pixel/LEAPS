"""Sanity check: raw EMG strides → muscle mapping → MuJoCo simulation.

Verifies the full data pipeline from Camargo EMG to MuJoCo simulation
before plugging in any AE or PPO. Two data sources are supported:

  levelground  Load from processed HDF5, filter to mode="levelground".
               All three self-selected speeds are included (~0.88–1.45 m/s).
               No --data-root needed.

  treadmill    Load from raw .mat files at a specific speed ± tolerance.
               Requires --data-root pointing to the Camargo raw dataset.
               Use --target-speed 1.2 for walk task match (LocoMuJoCo = 1.25 m/s).
               Use --target-speed 1.85 for highest available (fast-walk / run curiosity).

Pipeline verified:
  1. Load & filter strides           (N, 101, 11), values in [0, 1]
  2. Print EMG channel diagnostics   (mean, max, peak phase per channel)
  3. EMGToMuscleMapper               (11 ch → 18 bilateral muscles, 50% phase offset)
  4. Print muscle command diagnostics
  5. MuJoCo open-loop simulation     (MuscleHumanoidEnv, no LocoMuJoCo / JAX needed)
  6. Print simulation metrics        (steps, forward velocity, pelvis height)

Usage:
  # Level-ground (HDF5 only, fastest):
  python -m leaps.scripts.check_emg_sim \\
      --hdf5 /fast/lsivakumar/data/processed/emg_activations.h5 \\
      --source levelground --n-strides 30

  # Treadmill at 1.2 m/s — matches walk task (needs raw data):
  python -m leaps.scripts.check_emg_sim \\
      --hdf5 /fast/lsivakumar/data/processed/emg_activations.h5 \\
      --data-root /fast/lsivakumar/datasets/camargo \\
      --source treadmill --target-speed 1.2 --n-strides 30

  # Treadmill at 1.85 m/s — highest speed, curiosity / run task check:
  python -m leaps.scripts.check_emg_sim \\
      --hdf5 /fast/lsivakumar/data/processed/emg_activations.h5 \\
      --data-root /fast/lsivakumar/datasets/camargo \\
      --source treadmill --target-speed 1.85 --n-strides 30

  # Diagnostics only, no simulation:
  python -m leaps.scripts.check_emg_sim --hdf5 ... --source levelground --no-sim
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from leaps.data.loaders import discover_trials, load_mat_table
from leaps.data.metadata import EMG_CHANNELS
from leaps.data.processing import (
    find_stride_intervals,
    normalize_emg,
    rectify_emg,
    segment_strides,
    time_normalize_stride,
)
from leaps.envs.emg_mapping import EMGToMuscleMapper, MODEL_ACTUATORS

# Treadmill speeds in the Camargo dataset: 0.50–1.85 m/s in 0.05 m/s steps
_TREADMILL_SPEEDS = [round(0.5 + i * 0.05, 2) for i in range(28)]

# The gait10dof18musc.xml model XML path relative to the repo root
_MODEL_XML = (
    Path(__file__).resolve().parents[4] / "models" / "humanoid" / "gait10dof18musc.xml"
)


# ── Data loading ──────────────────────────────────────────────────────────────


def _subject_normalization(hdf5_path: str, subject: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (min_vals, max_vals) per channel for a subject, or None if missing."""
    with h5py.File(hdf5_path, "r") as f:
        grp = f.get(subject)
        if grp is None or "min_vals" not in grp or "max_vals" not in grp:
            return None
        return np.array(grp["min_vals"]), np.array(grp["max_vals"])


def load_strides_from_hdf5(
    hdf5_path: str,
    mode: str,
    subjects: list[str] | None = None,
    max_strides: int | None = None,
    rng_seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Load strides for a given mode from the processed HDF5.

    The HDF5 stores speeds as NaN for non-treadmill modes (levelground, ramp,
    stair) because those conditions files only have string labels which the
    numeric .mat parser cannot extract. All levelground strides are returned
    regardless of self-selected speed (slow ~0.88, normal ~1.17, fast ~1.45 m/s).

    Args:
        hdf5_path:   Path to emg_activations.h5.
        mode:        One of "levelground", "treadmill", "ramp", "stair".
        subjects:    Subjects to include (default: all AB* in file).
        max_strides: Cap on total strides (random sample if exceeded).
        rng_seed:    Seed for reproducible downsampling.

    Returns:
        strides:     (N, 101, 11) float array, values in [0, 1].
        subject_ids: Length-N list of subject ID strings.
    """
    all_strides: list[np.ndarray] = []
    all_subjects: list[str] = []

    with h5py.File(hdf5_path, "r") as f:
        if subjects is None:
            subjects = sorted(k for k in f.keys() if k.startswith("AB"))

        for subj in subjects:
            if subj not in f:
                continue
            grp = f[subj]
            if "strides" not in grp or "modes" not in grp:
                continue

            strides = np.array(grp["strides"])           # (n, 101, 11)
            modes = np.array(grp["modes"]).astype(str)   # (n,)

            mask = modes == mode
            if not mask.any():
                continue

            all_strides.append(strides[mask])
            all_subjects.extend([subj] * int(mask.sum()))

    if not all_strides:
        raise RuntimeError(f"No {mode} strides found in {hdf5_path}")

    strides_out = np.concatenate(all_strides, axis=0)

    if max_strides is not None and len(strides_out) > max_strides:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(strides_out), max_strides, replace=False)
        strides_out = strides_out[idx]
        all_subjects = [all_subjects[i] for i in idx]

    return strides_out, all_subjects


def load_treadmill_strides_at_speed(
    hdf5_path: str,
    data_root: str,
    target_speed: float,
    speed_tol: float = 0.05,
    subjects: list[str] | None = None,
    max_strides: int | None = None,
    fs: int = 1000,
    rng_seed: int = 42,
) -> tuple[np.ndarray, list[str], list[float]]:
    """Load treadmill strides at a specific speed from raw .mat files.

    Because the processed HDF5 does not store per-stride speed metadata,
    this function loads directly from the raw .mat files and filters strides
    to those where the conditions Speed column is within [target_speed ± tol]
    for the entire stride (i.e. steady-state, not mid-acceleration).

    Normalization (min_vals / max_vals) is loaded from the HDF5 per subject
    so that values are on the same [0, 1] scale as all other data.

    Args:
        hdf5_path:    Path to emg_activations.h5 (for per-subject normalization).
        data_root:    Root of the raw Camargo dataset directory.
        target_speed: Treadmill speed to select in m/s.
        speed_tol:    Tolerance around target_speed (default ±0.05 m/s).
        subjects:     Subjects to load (default: all found in data_root).
        max_strides:  Cap on total strides (random sample if exceeded).
        fs:           EMG sampling rate in Hz (1000 for Camargo).
        rng_seed:     Seed for reproducible downsampling.

    Returns:
        strides:        (N, 101, 11) float array, values in [0, 1].
        subject_ids:    Length-N list of subject ID strings.
        actual_speeds:  Length-N list of mean stride speeds (m/s).
    """
    all_strides: list[np.ndarray] = []
    all_subjects: list[str] = []
    all_speeds: list[float] = []

    # Discover treadmill trials that have all three required sensors
    trials = discover_trials(
        data_root,
        subjects=subjects,
        modes=["treadmill"],
        sensors=["emg", "gcRight", "conditions"],
    )

    if not trials:
        raise RuntimeError(
            f"No treadmill trials found in {data_root}. "
            "Check --data-root points to the Camargo raw dataset."
        )

    for subj, modes_dict in sorted(trials.items()):
        # Load per-subject normalization from HDF5
        norm = _subject_normalization(hdf5_path, subj)
        if norm is None:
            print(f"  [SKIP] {subj} — no normalization in HDF5")
            continue
        min_vals, max_vals = norm

        treadmill = modes_dict.get("treadmill", {})
        emg_files = treadmill.get("emg", [])
        gc_files = treadmill.get("gcRight", [])
        cond_files = treadmill.get("conditions", [])
        n_trials = min(len(emg_files), len(gc_files), len(cond_files))
        if n_trials == 0:
            continue

        subj_strides: list[np.ndarray] = []
        subj_speeds: list[float] = []

        for t in range(n_trials):
            emg_table = load_mat_table(emg_files[t])
            gc_table = load_mat_table(gc_files[t])
            cond_table = load_mat_table(cond_files[t])

            if not emg_table or not gc_table or "HeelStrike" not in gc_table:
                continue

            # EMG: separate header from 11 muscle channels
            channel_names = [k for k in emg_table if k != "Header"]
            if len(channel_names) != 11:
                continue
            raw_emg = np.column_stack([emg_table[ch] for ch in channel_names])
            emg_header = emg_table["Header"]

            rect_emg = rectify_emg(raw_emg, fs=fs)

            # Stride intervals from gait cycle right-leg heel strike
            intervals = find_stride_intervals(
                gc_table["Header"], gc_table["HeelStrike"]
            )
            rect_strides_raw = segment_strides(emg_header, rect_emg, intervals)

            cond_header = cond_table.get("Header")
            speed_col = cond_table.get("Speed")
            has_speed = cond_header is not None and speed_col is not None

            if not has_speed:
                # Conditions file parsed but Speed column not found (shouldn't happen
                # for treadmill, but guard against corrupt/missing files)
                continue

            for stride_raw, (t_start, t_end) in zip(rect_strides_raw, intervals):
                # Get speed values for this stride's time window
                mask = (cond_header >= t_start) & (cond_header < t_end)
                stride_speeds = speed_col[mask]
                if len(stride_speeds) == 0:
                    continue

                # Discard acceleration strides (treadmill ramps between speeds)
                # A steady-state stride has at most 2 unique rounded speed values
                unique_speeds = np.unique(np.round(stride_speeds, 2))
                if len(unique_speeds) > 2:
                    continue

                mean_speed = float(np.mean(stride_speeds))
                if abs(mean_speed - target_speed) > speed_tol:
                    continue

                # Normalize to [0, 1] using per-subject reference normalization
                norm_stride = normalize_emg(stride_raw, min_vals, max_vals)
                tn = time_normalize_stride(norm_stride)  # (101, 11)
                subj_strides.append(tn)
                subj_speeds.append(mean_speed)

        if subj_strides:
            print(
                f"  {subj}: {len(subj_strides)} strides  "
                f"speed {np.mean(subj_speeds):.3f} ± {np.std(subj_speeds):.3f} m/s"
            )
            all_strides.extend(subj_strides)
            all_subjects.extend([subj] * len(subj_strides))
            all_speeds.extend(subj_speeds)

    if not all_strides:
        raise RuntimeError(
            f"No treadmill strides found at {target_speed} ± {speed_tol} m/s.\n"
            f"Available speeds: {_TREADMILL_SPEEDS}\n"
            "Try a speed from that list or increase --speed-tol."
        )

    strides_out = np.stack(all_strides, axis=0)  # (N, 101, 11)

    if max_strides is not None and len(strides_out) > max_strides:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(strides_out), max_strides, replace=False)
        strides_out = strides_out[idx]
        all_subjects = [all_subjects[i] for i in idx]
        all_speeds = [all_speeds[i] for i in idx]

    return strides_out, all_subjects, all_speeds


# ── Diagnostics ───────────────────────────────────────────────────────────────


def print_emg_diagnostics(
    strides: np.ndarray,
    label: str,
    actual_speeds: list[float] | None = None,
) -> None:
    """Print per-channel EMG statistics and mean stride profile."""
    n_strides, T, n_ch = strides.shape
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  EMG Diagnostics — {label}")
    print(sep)
    print(f"  Strides : {n_strides}   Shape : ({n_strides}, {T}, {n_ch})")
    print(f"  Values  : min={strides.min():.3f}  max={strides.max():.3f}  "
          f"mean={strides.mean():.3f}  fraction > 1.0 = {np.mean(strides > 1.0):.2%}")

    if actual_speeds:
        print(
            f"  Speed   : {np.mean(actual_speeds):.3f} ± {np.std(actual_speeds):.3f} m/s  "
            f"(range {min(actual_speeds):.2f}–{max(actual_speeds):.2f})"
        )

    flat = strides.reshape(-1, n_ch)  # (N*T, 11)

    print(f"\n  {'Channel':<24} {'Mean':>6} {'Max':>6} {'Min':>6} {'Std':>6}")
    print(f"  {'─'*24} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for i, ch in enumerate(EMG_CHANNELS):
        col = flat[:, i]
        print(
            f"  {ch:<24} {np.mean(col):6.3f} {np.max(col):6.3f} "
            f"{np.min(col):6.3f} {np.std(col):6.3f}"
        )

    # Mean stride profile — show peak phase and amplitude per channel
    mean_profile = strides.mean(axis=0)  # (101, 11)
    peak_phases = np.argmax(mean_profile, axis=0)  # (11,)
    print(f"\n  Mean stride profile (peak phase = % of gait cycle):")
    print(f"  {'Channel':<24} {'Peak%':>6} {'PeakAmp':>8}  profile")
    print(f"  {'─'*24} {'─'*6} {'─'*8}  {'─'*30}")
    for i, ch in enumerate(EMG_CHANNELS):
        profile = mean_profile[:, i]
        peak_amp = profile.max()
        n_bars = int(peak_amp * 30)
        bar = "█" * min(n_bars, 30)
        print(f"  {ch:<24} {peak_phases[i]:6d} {peak_amp:8.3f}  {bar}")


def print_muscle_diagnostics(muscle_cmds_01: np.ndarray) -> None:
    """Print per-actuator statistics after EMGToMuscleMapper."""
    T, n_act = muscle_cmds_01.shape
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Muscle Command Diagnostics (18 actuators, [0, 1])")
    print(sep)
    print(
        f"  Shape   : {muscle_cmds_01.shape}   "
        f"(will be converted to [-1, 1] for LocoMuJoCo)"
    )
    print(
        f"  Overall : min={muscle_cmds_01.min():.3f}  "
        f"max={muscle_cmds_01.max():.3f}  mean={muscle_cmds_01.mean():.3f}"
    )
    print(f"\n  {'Actuator':<20} {'Mean':>6} {'Max':>6} {'Min':>6}  side")
    print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*6}  {'─'*4}")
    for i, act in enumerate(MODEL_ACTUATORS):
        col = muscle_cmds_01[:, i]
        side = "R" if act.endswith("_r") else "L"
        print(
            f"  {act:<20} {np.mean(col):6.3f} {np.max(col):6.3f} "
            f"{np.min(col):6.3f}  {side}"
        )


def run_simulation(
    actions_loco: np.ndarray,
    xml_path: str,
    max_steps: int,
) -> dict:
    """Open-loop simulation with MuscleHumanoidEnv (raw MuJoCo, no JAX).

    Actions cycle through the sequence until max_steps is reached or
    the pelvis falls below 0.5 m (early termination).

    Args:
        actions_loco: (T, 18) muscle commands in [-1, 1] (LocoMuJoCo convention).
                      Will be converted internally to [0, 1] for the raw env.
        xml_path:     Path to gait10dof18musc.xml.
        max_steps:    Maximum simulation steps.

    Returns:
        dict with keys: qpos (steps, n_q), qvel (steps, n_v), actions (steps, 18).
    """
    from leaps.envs.muscle_env import MuscleHumanoidEnv

    env = MuscleHumanoidEnv(xml_path=xml_path)
    env.reset()

    # MuscleHumanoidEnv.step expects [0, 1]; convert from LocoMuJoCo [-1, 1]
    actions_01 = (actions_loco + 1.0) / 2.0

    qpos_list: list[np.ndarray] = []
    qvel_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []

    n_actions = len(actions_01)
    fell_step = None

    for step in range(max_steps):
        action = actions_01[step % n_actions]
        env.step(action)

        qpos_list.append(env.data.qpos.copy())
        qvel_list.append(env.data.qvel.copy())
        act_list.append(action.copy())

        # Pelvis height = qpos[1] (pelvis_ty slide joint)
        if env.data.qpos[1] < 0.5:
            fell_step = step + 1
            break

    env.close()

    results = {
        "qpos": np.array(qpos_list),
        "qvel": np.array(qvel_list),
        "actions": np.array(act_list),
        "fell_step": fell_step,
    }
    return results


def print_sim_metrics(results: dict, dt: float, label: str = "") -> None:
    """Print summary metrics from simulation results."""
    qpos = results["qpos"]
    qvel = results["qvel"]
    n_steps = len(qpos)
    fell_step = results.get("fell_step")

    pelvis_x = qpos[:, 0]
    pelvis_y = qpos[:, 1]
    pelvis_tilt_deg = np.degrees(qpos[:, 2])  # pelvis_tilt hinge
    fwd_vel = qvel[:, 0]

    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Simulation Results — {label}")
    print(sep)

    if fell_step is not None:
        print(f"  Fell at step {fell_step} ({fell_step * dt:.2f} s) — pelvis < 0.5 m")
    else:
        print(f"  Completed {n_steps} steps ({n_steps * dt:.1f} s) without falling")

    print(f"  Forward distance : {pelvis_x[-1] - pelvis_x[0]:.3f} m")
    print(
        f"  Forward velocity : {np.mean(fwd_vel):.3f} ± {np.std(fwd_vel):.3f} m/s  "
        f"(peak {np.max(fwd_vel):.3f} m/s)"
    )
    print(
        f"  Pelvis height    : {np.mean(pelvis_y):.3f} ± {np.std(pelvis_y):.3f} m  "
        f"(min {pelvis_y.min():.3f} m, start {pelvis_y[0]:.3f} m)"
    )
    print(f"  Pelvis tilt      : {np.mean(pelvis_tilt_deg):.1f}° mean  "
          f"(range {pelvis_tilt_deg.min():.1f}°–{pelvis_tilt_deg.max():.1f}°)")

    # Quick verdict
    print()
    if fell_step is not None and fell_step < 50:
        print("  VERDICT: Falls almost immediately — check muscle mapping or initial pose.")
    elif fell_step is not None:
        print(f"  VERDICT: Falls after {fell_step * dt:.1f} s of simulation.")
    elif np.mean(fwd_vel) > 0.3:
        print(f"  VERDICT: Humanoid moves forward at ~{np.mean(fwd_vel):.2f} m/s — pipeline OK.")
    else:
        print("  VERDICT: Humanoid does not fall but also does not move forward — check mapping.")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity check: raw EMG strides → muscle mapping → MuJoCo simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hdf5",
        default="/fast/lsivakumar/data/processed/emg_activations.h5",
        help="Processed HDF5 from leaps-preprocess. Required for normalization.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Raw Camargo dataset root. Required for --source treadmill.",
    )
    parser.add_argument(
        "--source",
        choices=["treadmill", "levelground"],
        default="levelground",
        help=(
            "Data source. 'levelground' loads from HDF5 (no --data-root). "
            "'treadmill' loads from raw .mat at --target-speed (needs --data-root)."
        ),
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=1.2,
        help=(
            "Treadmill speed to select in m/s. Only used with --source treadmill. "
            "1.2 matches the LocoMuJoCo walk task (1.25 m/s). "
            "1.85 is the highest available — use to check run-task EMG patterns. "
            "Note: 1.85 m/s is still fast walking, not true running (>~2 m/s). "
            "Default: 1.2"
        ),
    )
    parser.add_argument(
        "--speed-tol",
        type=float,
        default=0.05,
        help="Speed tolerance for treadmill filtering (±m/s). Default: 0.05.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Limit to specific subjects (e.g. AB06 AB09). Default: all.",
    )
    parser.add_argument(
        "--n-strides",
        type=int,
        default=50,
        help="Max strides to load and use for simulation. Default: 50.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Simulation duration in seconds. "
            "Default: one pass through all loaded strides."
        ),
    )
    parser.add_argument(
        "--xml",
        default=None,
        help="MuJoCo XML path. Default: models/humanoid/gait10dof18musc.xml.",
    )
    parser.add_argument(
        "--no-sim",
        action="store_true",
        help="Skip simulation — print EMG and muscle diagnostics only.",
    )
    args = parser.parse_args()

    # ── Resolve XML path ────────────────────────────────────────────────────
    xml_path = str(args.xml if args.xml else _MODEL_XML)
    if not Path(xml_path).exists():
        print(f"ERROR: MuJoCo XML not found: {xml_path}", file=sys.stderr)
        sys.exit(1)

    # ── Header ──────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  LEAPS — EMG Simulation Sanity Check")
    print("=" * 62)
    print(f"  Source  : {args.source}")
    print(f"  HDF5    : {args.hdf5}")
    if args.source == "treadmill":
        print(f"  Speed   : {args.target_speed} ± {args.speed_tol} m/s")
        print(f"  Data    : {args.data_root}")

    # ── Step 1: Load strides ────────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print("  Step 1 — Load EMG strides")
    print(f"{'─' * 62}")

    actual_speeds: list[float] | None = None

    if args.source == "levelground":
        print("  Loading levelground strides from HDF5 ...")
        print("  NOTE: Speed labels (slow/normal/fast) are stored as strings in the raw")
        print("  conditions files. The HDF5 speeds column is NaN for levelground —")
        print("  all three self-selected speeds (~0.88–1.45 m/s) are included.")
        strides, subject_ids = load_strides_from_hdf5(
            args.hdf5,
            mode="levelground",
            subjects=args.subjects,
            max_strides=args.n_strides,
        )
        source_label = "Level-ground walking (all speeds ~0.88–1.45 m/s)"

    else:  # treadmill
        # Snap target_speed to nearest Camargo treadmill speed
        target = args.target_speed
        closest = min(_TREADMILL_SPEEDS, key=lambda s: abs(s - target))
        if abs(closest - target) > 1e-9:
            print(
                f"  NOTE: {target} m/s not exactly in Camargo treadmill speeds — "
                f"using closest: {closest} m/s"
            )
            target = closest

        is_run_check = target >= 1.5
        if is_run_check:
            print(
                f"  NOTE: {target} m/s is fast walking, NOT true running (~>2 m/s). "
                f"LocoMuJoCo run task target = 2.5 m/s. "
                f"These EMG patterns are the closest available approximation."
            )

        if args.data_root is not None:
            # Load directly from raw .mat files (works even before HDF5 regeneration)
            print(f"  Loading treadmill strides at {target} m/s from raw .mat files ...")
            strides, subject_ids, actual_speeds = load_treadmill_strides_at_speed(
                args.hdf5,
                args.data_root,
                target_speed=target,
                speed_tol=args.speed_tol,
                subjects=args.subjects,
                max_strides=args.n_strides,
            )
        else:
            # Load from HDF5 speed column (requires regenerated HDF5 with speed metadata)
            print(f"  Loading treadmill strides at {target} m/s from HDF5 ...")
            print(f"  (Requires regenerated HDF5 with speed column. Use --data-root to load from raw.)")
            all_strides_list: list[np.ndarray] = []
            subject_ids = []
            actual_speeds = []

            with h5py.File(args.hdf5, "r") as f:
                subjs = args.subjects or sorted(k for k in f.keys() if k.startswith("AB"))
                for subj in subjs:
                    if subj not in f:
                        continue
                    grp = f[subj]
                    if "strides" not in grp or "modes" not in grp or "speeds" not in grp:
                        continue
                    strides_s = np.array(grp["strides"])
                    modes_s = np.array(grp["modes"]).astype(str)
                    speeds_s = np.array(grp["speeds"])

                    mask = (
                        (modes_s == "treadmill")
                        & np.isfinite(speeds_s)
                        & (np.abs(speeds_s - target) <= args.speed_tol)
                    )
                    if not mask.any():
                        continue
                    all_strides_list.append(strides_s[mask])
                    subject_ids.extend([subj] * int(mask.sum()))
                    actual_speeds.extend(speeds_s[mask].tolist())
                    print(
                        f"  {subj}: {int(mask.sum())} strides  "
                        f"speed {speeds_s[mask].mean():.3f} m/s"
                    )

            if not all_strides_list:
                print(
                    f"ERROR: No treadmill strides at {target} ± {args.speed_tol} m/s in HDF5.\n"
                    "Did you regenerate the HDF5? Use --data-root to load from raw .mat files.",
                    file=sys.stderr,
                )
                sys.exit(1)

            strides = np.concatenate(all_strides_list, axis=0)
            if args.n_strides and len(strides) > args.n_strides:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(strides), args.n_strides, replace=False)
                strides = strides[idx]
                subject_ids = [subject_ids[i] for i in idx]
                actual_speeds = [actual_speeds[i] for i in idx]

        task_note = "(walk task ≈ 1.25 m/s)" if not is_run_check else "(fast walk / run curiosity)"
        source_label = f"Treadmill {target} m/s {task_note}"

    unique_subjects = sorted(set(subject_ids))
    print(
        f"\n  Loaded : {len(strides)} strides  "
        f"from {len(unique_subjects)} subjects: {', '.join(unique_subjects)}"
    )
    print(f"  Shape  : {strides.shape}   dtype={strides.dtype}")

    # ── Step 2: EMG diagnostics ─────────────────────────────────────────────
    print_emg_diagnostics(strides, source_label, actual_speeds)

    # ── Step 3: EMG → 18 muscles ────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print("  Step 2 — EMGToMuscleMapper (11 ch → 18 bilateral muscles)")
    print(f"{'─' * 62}")
    print("  Applying map_stride_with_phase_offset (50% left-leg delay) ...")

    mapper = EMGToMuscleMapper()
    muscle_strides: list[np.ndarray] = []
    for i in range(len(strides)):
        # (101, 11) → (101, 18) in [0, 1]
        muscle_strides.append(mapper.map_stride_with_phase_offset(strides[i]))

    # Concatenate all strides into one long action sequence
    muscle_cmds_01 = np.concatenate(muscle_strides, axis=0)  # (N*101, 18) in [0, 1]
    actions_loco = 2.0 * muscle_cmds_01 - 1.0               # (N*101, 18) in [-1, 1]

    print(f"  Action sequence: {actions_loco.shape}  range [{actions_loco.min():.2f}, {actions_loco.max():.2f}]")

    print_muscle_diagnostics(muscle_cmds_01)

    # ── Step 4: Simulation ──────────────────────────────────────────────────
    if args.no_sim:
        print("\n  [Simulation skipped — --no-sim flag set]")
        return

    import mujoco
    m_model = mujoco.MjModel.from_xml_path(xml_path)
    dt = m_model.opt.timestep

    if args.duration:
        max_steps = int(args.duration / dt)
    else:
        max_steps = len(actions_loco)  # one pass through all strides

    print(f"\n{'─' * 62}")
    print("  Step 3 — MuJoCo Open-Loop Simulation (MuscleHumanoidEnv)")
    print(f"{'─' * 62}")
    print(f"  XML      : {xml_path}")
    print(f"  dt       : {dt:.4f} s  ({1 / dt:.0f} Hz)")
    print(f"  Actions  : {actions_loco.shape}  (cycles if max_steps > len)")
    print(f"  Max steps: {max_steps}  ({max_steps * dt:.1f} s)")
    print("  Running ...")

    results = run_simulation(actions_loco, xml_path, max_steps)
    print_sim_metrics(results, dt, label=source_label)

    print(f"\n{'=' * 62}")
    print("  Done.")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
