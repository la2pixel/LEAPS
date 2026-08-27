"""Open-loop EMG replay in sconegym.

Loads one subject's walking strides from the HDF5, maps 11-channel EMG → 18
SCONE muscles (exact anatomical matches only, see envs/emg_mapping.py),
interpolates to sconegym timesteps, and feeds them directly to the
simulation without any policy or RL.

Purpose: sanity-check the EMG-to-muscle mapping. If the model walks for a few
strides before falling, the mapping is plausible and Phase 2 (RL + latent prior)
is worth pursuing.

Caveat: the left leg gets 0 activation throughout (no policy here to fill it
in, and we don't fabricate it -- see envs/emg_mapping.py). That means this
replay tests whether the *mapped right-leg subset* looks timed/signed
correctly, not whether the mapping alone can produce full bipedal walking --
a one-legged model will fall regardless of mapping quality.

Usage (on cluster):
    cd /home/nadinebadie/lalitha/LEAPS
    source env.sh
    python -m leaps.scripts.emg_replay --subject AB06 --n_strides 8

    # Try a different subject or speed:
    python -m leaps.scripts.emg_replay --subject AB10 --target_speed 1.35

    # Skip saving .sto file (faster):
    python -m leaps.scripts.emg_replay --no_store
"""

import argparse
import os
import sys

import h5py
import numpy as np

from leaps.envs.emg_mapping import EMGToMuscleMapper

try:
    import gym
    import sconegym  # noqa: registers env IDs
    SCONEGYM_AVAILABLE = True
except ImportError:
    SCONEGYM_AVAILABLE = False

STEP_SIZE = 0.01  # sconegym default timestep (seconds)
ENV_ID = "sconewalk_h0918-v1"
MODEL_NAME = "h0918"  # key into leaps.envs.emg_mapping.MODEL_MAPS, must match ENV_ID
H5_DEFAULT = "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_subject_strides(
    h5_path: str,
    subject: str,
    target_speed: float,
    speed_tol: float,
    n_strides: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (strides, durations) for level-ground walking at target_speed."""
    with h5py.File(h5_path, "r") as f:
        if subject not in f:
            raise ValueError(
                f"Subject {subject!r} not in HDF5. Available: {sorted(f.keys())}"
            )
        grp = f[subject]
        strides   = grp["strides"][:]    # (N, 101, 11)
        durations = grp["durations"][:]  # (N,) seconds
        speeds    = grp["speeds"][:]     # (N,) m/s
        labels    = np.array([
            x.decode() if isinstance(x, bytes) else x for x in grp["labels"][:]
        ])
        modes     = np.array([
            x.decode() if isinstance(x, bytes) else x for x in grp["modes"][:]
        ])

    walk_mask  = labels == "walk"
    speed_mask = np.abs(speeds - target_speed) <= speed_tol
    valid_mask = ~np.isnan(speeds)
    mask = walk_mask & speed_mask & valid_mask

    n_found = int(mask.sum())
    print(
        f"[data]  {subject}: {n_found} strides matching "
        f"label='walk', speed={target_speed}±{speed_tol} m/s"
    )
    if n_found == 0:
        raise ValueError(
            f"No strides found. Try widening --speed_tol or picking another subject."
        )

    strides   = strides[mask]
    durations = durations[mask]
    modes_sel = modes[mask]

    # Prefer treadmill strides (more consistent speed control)
    tmill = modes_sel == "treadmill"
    if tmill.sum() >= n_strides:
        strides   = strides[tmill]
        durations = durations[tmill]
        print(f"[data]  Using treadmill strides ({tmill.sum()} available)")

    n_take = min(n_strides, len(strides))
    print(
        f"[data]  Taking {n_take} strides, "
        f"mean duration={durations[:n_take].mean():.3f}s"
    )
    return strides[:n_take], durations[:n_take]


# ── Interpolation ─────────────────────────────────────────────────────────────


def interpolate_stride(
    muscle_stride: np.ndarray,  # (101, 18) in [0, 1]
    duration: float,
) -> np.ndarray:
    """Resample 101 gait-cycle points → n_steps sconegym frames."""
    n_steps = max(1, round(duration / STEP_SIZE))
    gc_axis    = np.linspace(0, 100, 101)
    query_axis = np.linspace(0, 100, n_steps)
    out = np.empty((n_steps, muscle_stride.shape[1]), dtype=np.float32)
    for ch in range(muscle_stride.shape[1]):
        out[:, ch] = np.interp(query_axis, gc_axis, muscle_stride[:, ch])
    return out


# ── Main replay ───────────────────────────────────────────────────────────────


def run_replay(
    h5_path: str,
    subject: str,
    n_strides: int,
    target_speed: float,
    speed_tol: float,
    store_results: bool,
) -> None:
    if not SCONEGYM_AVAILABLE:
        print("ERROR: sconegym not importable. Run this on the cluster with SCONE installed.")
        sys.exit(1)

    # ── Load EMG ──
    strides_emg, durations = load_subject_strides(
        h5_path, subject, target_speed, speed_tol, n_strides
    )

    # ── Create env first -- the mapper needs the model's actual actuator
    # order, since that's what env.step() expects the action vector in. ──
    env = gym.make(ENV_ID)
    # Disable action clipping so the full [0, 1] EMG range reaches the muscles.
    # sconewalk-v1 defaults to clip_actions=True which would halve all activations.
    env.unwrapped.clip_actions = False
    actuator_names = [a.name() for a in env.unwrapped.model.actuators()]
    # Save under your own lalitha/ results folder instead of the live/ root,
    # same DATE_TIME.model_name pattern gaitgym.py uses by default.
    env.unwrapped.set_output_dir(f"lalitha/emg_replay/DATE_TIME.{subject}")

    # ── Map 11-channel EMG → SCONE muscles (right leg only, see emg_mapping.py) ──
    mapper = EMGToMuscleMapper(MODEL_NAME, actuator_names)
    n_actuators = len(actuator_names)
    strides_muscle = mapper.map_batch(
        strides_emg.reshape(-1, strides_emg.shape[-1])
    ).reshape(len(strides_emg), -1, n_actuators)  # (n_strides, 101, n_actuators)

    # ── Interpolate each stride to sconegym timesteps ──
    frames_list = []
    for i, (ms, dur) in enumerate(zip(strides_muscle, durations)):
        f = interpolate_stride(ms, dur)
        frames_list.append(f)
        print(f"[interp] Stride {i + 1}: {dur:.3f}s → {len(f)} steps")
    all_frames = np.concatenate(frames_list, axis=0)  # (total_steps, n_actuators)

    total_steps = len(all_frames)
    print(
        f"\n[replay] Total: {total_steps} steps "
        f"({total_steps * STEP_SIZE:.2f}s of simulation)\n"
    )

    # Prime result saving before reset so the .sto file captures the episode
    env.unwrapped.store_next = store_results
    obs = env.reset()

    total_reward = 0.0
    fell_at = None

    print(f"{'Step':>5}  {'com_y':>7}  {'vel_x':>7}  {'reward':>8}")
    print("-" * 36)

    for t, muscle_act in enumerate(all_frames):
        obs, reward, done, _ = env.step(muscle_act)
        total_reward += reward

        if t % 25 == 0 or done:
            com_y = env.unwrapped.model.com_pos().y
            vel_x = env.unwrapped.model.com_vel().x
            print(f"{t:5d}  {com_y:7.3f}  {vel_x:7.3f}  {reward:8.3f}")

        if done:
            fell_at = t
            break

    if store_results:
        env.write_now()
    env.close()

    # ── Summary ──
    print("\n" + "=" * 40)
    print("EMG REPLAY SUMMARY")
    print("=" * 40)
    print(f"  Subject:      {subject}")
    print(f"  Strides:      {len(strides_emg)}")
    print(f"  Total steps:  {total_steps}  ({total_steps * STEP_SIZE:.2f}s)")
    if fell_at is not None:
        pct = fell_at / total_steps * 100
        print(f"  Fell at step: {fell_at}  ({fell_at * STEP_SIZE:.2f}s, {pct:.1f}% through)")
    else:
        print(f"  Completed all {total_steps} steps without falling!")
    print(f"  Total reward: {total_reward:.3f}")
    if store_results:
        print(f"  Results saved to SCONE results dir (open in SCONE GUI to visualise)")
    print("=" * 40)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open-loop EMG replay in sconegym"
    )
    parser.add_argument(
        "--h5", default=None,
        help="Path to emg_activations_v2.h5 (auto-detected if not given)",
    )
    parser.add_argument("--subject",      default="AB06",  help="Subject ID (default: AB06)")
    parser.add_argument("--n_strides",    type=int,   default=8,    help="Number of strides to replay")
    parser.add_argument("--target_speed", type=float, default=1.2,  help="Target speed m/s (default: 1.2)")
    parser.add_argument("--speed_tol",    type=float, default=0.15, help="Speed filter tolerance m/s")
    parser.add_argument("--no_store",     action="store_true",       help="Skip saving .sto result file")
    args = parser.parse_args()

    h5_path = args.h5 or os.environ.get("LEAPS_EMG_H5", "") or H5_DEFAULT
    if not os.path.exists(h5_path):
        print(f"ERROR: HDF5 not found at {h5_path!r}. Set LEAPS_EMG_H5 or pass --h5.")
        sys.exit(1)

    run_replay(
        h5_path=h5_path,
        subject=args.subject,
        n_strides=args.n_strides,
        target_speed=args.target_speed,
        speed_tol=args.speed_tol,
        store_results=not args.no_store,
    )


if __name__ == "__main__":
    main()
