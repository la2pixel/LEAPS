"""Render EMG replay simulation using Gait10dof18Musc (LocoMuJoCo env).

Runs entirely on the cluster using EGL offscreen rendering — no display needed.
Generates one video per condition (treadmill / levelground) filtered near 1.25 m/s.

Usage:
    python -m leaps.scripts.render_emg \
        --data /fast/lsivakumar/data/processed/emg_activations_v2.h5 \
        --output-dir outputs/emg_conditions \
        --duration 5
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np

import leaps.envs  # registers Gait10dof18Musc + HeightJointTerminalStateHandler
from leaps.data.metadata import LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides
from leaps.envs.emg_mapping import EMGToMuscleMapper
from leaps.envs.gait10dof_env import Gait10dof18Musc, _INIT_POSES

# FPS is set from sim_dt in main() and passed to render_video — do not hardcode here.
FPS: int = 100  # placeholder; overridden at runtime to match 1/sim_dt

# Level-ground self-selected speeds average ~0.88 / 1.17 / 1.45 m/s.
# Use a wider tolerance (0.30) to catch the nearest level-ground condition.
_SPEED_TOL = {"treadmill": 0.10, "levelground": 0.30}


def _resample_stride(stride: np.ndarray, n_out: int) -> np.ndarray:
    t_in = np.linspace(0, 1, stride.shape[0])
    t_out = np.linspace(0, 1, n_out)
    return np.stack([np.interp(t_out, t_in, stride[:, c]) for c in range(stride.shape[1])], axis=1)


def _mode_match(modes: np.ndarray, name: str) -> np.ndarray:
    """Match mode column regardless of bytes vs str encoding."""
    return (modes == name.encode()) | (modes == name)


def build_actions(
    strides: np.ndarray,
    metadata: dict,
    mode: str,
    n_steps: int,
    sim_dt: float,
    target_speed: float = 1.25,
    trial_idx: int = 0,
) -> np.ndarray:
    """Select strides for a given mode near target_speed and build muscle action sequence."""
    mapper = EMGToMuscleMapper()
    speeds = metadata["speeds"]

    mode_mask = _mode_match(metadata["modes"], mode)
    speed_tol = _SPEED_TOL.get(mode, 0.15)
    near_speed = np.abs(speeds - target_speed) <= speed_tol
    mask = mode_mask & near_speed

    if not mask.any():
        print(f"  WARNING: no {mode} strides within ±{speed_tol} m/s of {target_speed:.2f}, "
              f"using all {mode} strides")
        mask = mode_mask

    if not mask.any():
        raise ValueError(f"No strides found for mode='{mode}'")

    filtered = strides[mask]
    filtered_speeds = speeds[mask]
    n_avail = len(filtered)
    print(f"  [{mode}] {n_avail} strides | speed range "
          f"[{filtered_speeds.min():.2f}, {filtered_speeds.max():.2f}] m/s")

    mean_speed = float(np.nanmean(filtered_speeds))
    if np.isnan(mean_speed):
        mean_speed = target_speed  # levelground has no recorded speeds; use requested speed
    stride_len_m = 2 * 0.71 * (mean_speed ** 0.48)
    stride_duration_s = stride_len_m / mean_speed
    n_per_stride = max(101, round(stride_duration_s / sim_dt))
    print(f"  stride duration: {stride_duration_s:.3f} s → {n_per_stride} sim steps")

    n_needed = (n_steps // n_per_stride) + 2
    # Each trial_idx picks a different contiguous block
    start = (trial_idx * n_needed) % max(1, n_avail - n_needed)
    selected = filtered[start: start + n_needed]
    if len(selected) < n_needed:
        selected = np.concatenate([selected, filtered[:n_needed - len(selected)]])

    resampled = []
    for s in selected:
        mapped = mapper.map_stride_with_phase_offset(s)
        resampled.append(_resample_stride(mapped, n_per_stride))

    muscle_act = np.concatenate(resampled, axis=0)[:n_steps]
    return mapper.to_loco_action(muscle_act).astype(np.float32)


def render_video(env, actions: np.ndarray, output_path: str, label: str = "", fps: int = 100) -> None:
    model = env._model
    data = env._data
    env.reset()

    # The EMG action sequence always starts from frame 0 (right heel strike).
    # Override the random RSI pose to pose 0 so the muscles match the joint angles.
    # A mid-stance or push-off reset with heel-strike muscle activation causes
    # immediate knee collapse because the quads aren't loaded to hold a bent knee.
    for jnt_name, angle in _INIT_POSES[0].items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle
    env._apply_eq_constraints_cpu(data)
    mujoco.mj_kinematics(model, data)
    heel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
    heel_z = float(data.xpos[heel_id, 2])
    # Get heel geom radius directly from model — sphere geom: size[0] = radius
    heel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "calcn_r_geom_1")
    heel_geom_radius = float(model.geom_size[heel_geom_id, 0])
    pty_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_ty")
    data.qpos[model.jnt_qposadr[pty_jid]] -= (heel_z - heel_geom_radius)
    data.qvel[:] = 0.0
    data.qvel[env._qvel_pelvis_tx] = 1.25
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance = 3.5
    cam.elevation = -15.0
    cam.azimuth = 90.0

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []
    n = len(actions)
    for t in range(n):
        env.step(actions[t])
        if t % max(1, n // 10) == 0:
            print(f"  {label}: {t}/{n} ({100*t//n}%)")
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    renderer.close()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=fps)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=LEAPS_H5_PATH)
    parser.add_argument("--output-dir", default="outputs/emg_conditions")
    parser.add_argument("--speed", type=float, default=1.25)
    parser.add_argument("--n-trials", type=int, default=1,
                        help="Number of different trial windows per condition")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    print("Creating environment...")
    Gait10dof18Musc.mjx_enabled = False
    try:
        env = Gait10dof18Musc.generate()
    finally:
        Gait10dof18Musc.mjx_enabled = True

    # Control step = n_substeps × physics timestep. env.step() advances this much per call.
    # Using the raw physics timestep (0.001s) would give 10× too many steps and play
    # the action sequence 10× too slowly.
    sim_dt = float(env._model.opt.timestep * env._n_substeps)  # 0.001 × 10 = 0.01s
    fps = round(1.0 / sim_dt)                                   # 100 Hz → 100 FPS (1 frame per step)
    n_steps = int(args.duration / sim_dt)
    print(f"sim_dt={sim_dt*1000:.1f} ms | fps={fps} | duration={args.duration}s → {n_steps} steps\n")

    print("Loading strides...")
    strides, metadata = load_strides(args.data, with_metadata=True)
    print(f"Total strides: {len(strides)}\n")

    for mode in ("treadmill", "levelground"):
        for trial in range(args.n_trials):
            label = f"{mode}_trial{trial}"
            out = os.path.join(args.output_dir, f"emg_{label}.mp4")
            print(f"── {label} ──")
            try:
                actions = build_actions(strides, metadata, mode, n_steps, sim_dt,
                                        target_speed=args.speed, trial_idx=trial)
                render_video(env, actions, out, label=label, fps=fps)
            except ValueError as e:
                print(f"  SKIP: {e}")
            print()

    print(f"Done. Videos in {args.output_dir}/")


if __name__ == "__main__":
    main()
