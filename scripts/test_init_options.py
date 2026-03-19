"""Try different gait-phase starting frames and record a short video for each.

Samples frames at every 10% of the stride cycle from the hang trajectory,
drops pelvis to ground height, runs free on the floor.
Watch the videos to find which frame gives the most stable start.

Usage:
    python3 /lustre/fast/fast/lsivakumar/LEAPS/scripts/test_init_options.py \
        --data /fast/lsivakumar/data/processed/emg_activations.h5 \
        --hang-traj results/hang_test.npz \
        --output-dir results/init_test
"""

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import mujoco
import numpy as np
from pathlib import Path


def _load_actions(data_path):
    from leaps.data.stride_dataset import load_strides
    from leaps.envs.emg_mapping import EMGToMuscleMapper

    strides = load_strides(data_path)
    mapper = EMGToMuscleMapper()
    n = min(strides.shape[0], 50)
    actions = []
    for i in range(n):
        s18 = mapper.map_stride_with_phase_offset(strides[i])
        actions.append(2.0 * s18 - 1.0)
    return np.concatenate(actions, axis=0)  # (T, 18) in [-1, 1]


def _make_env(video_dir, video_name):
    from leaps.envs.gait10dof_env import Gait10dof18Musc
    from leaps.envs.rewards import ForwardVelocityReward
    from leaps.envs.terminal_state import HeightJointTerminalStateHandler

    HeightJointTerminalStateHandler.register()
    ForwardVelocityReward.register()
    Gait10dof18Musc.register()

    Path(video_dir).mkdir(parents=True, exist_ok=True)
    return Gait10dof18Musc(
        terminal_state_type="HeightJointTerminalStateHandler",
        goal_type="NoGoal",
        reward_type="WalkingReward",
        reward_params={"target_velocity": 1.25},
        headless=True,
        recorder_params={
            "path": video_dir,
            "tag": "",
            "video_name": video_name,
            "fps": 100,
            "compress": True,
        },
    )


def _run_one(env, actions_seq, frame_idx, max_steps, label):
    env.render(record=True)
    survived = max_steps
    for step in range(max_steps):
        action = actions_seq[(frame_idx + step) % len(actions_seq)]
        _, reward, terminated, truncated, _ = env.step(np.asarray(action))
        env.render(record=True)
        if terminated or truncated:
            survived = step + 1
            break
    print(f"  frame {frame_idx:3d} ({label}): survived {survived} steps "
          f"({survived * env.info.dt:.2f}s)")
    return survived


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--hang-traj", required=True)
    parser.add_argument("--output-dir", default="results/init_test")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Max duration per trial (seconds)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    actions_seq = _load_actions(args.data)
    hang = np.load(args.hang_traj)
    hang_qpos = hang["qpos"]   # (T, nq)
    hang_qvel = hang["qvel"]   # (T, nv)

    # Try every 10% of the stride (101 points = one stride)
    stride_len = 101
    frames = {f"frame{i:03d}_{i*100//stride_len}pct": i
              for i in range(0, stride_len, 10)}

    results = {}
    dt = None
    for label, frame_idx in frames.items():
        env = _make_env(str(output_dir / "videos"), label)
        if dt is None:
            dt = env.info.dt
            max_steps = int(args.duration / dt)

        env.reset(key=jax.random.key(0))

        qpos = hang_qpos[frame_idx].copy()
        qvel = hang_qvel[frame_idx].copy()
        qpos[0] = 0.0    # reset forward position
        qpos[1] = 0.95   # ground height
        qvel[0] = 0.0
        qvel[1] = 0.0
        env._data.qpos[:] = qpos
        env._data.qvel[:] = qvel
        mujoco.mj_forward(env._model, env._data)

        survived = _run_one(env, actions_seq, frame_idx, max_steps, label)
        results[label] = survived
        env.stop()

    print(f"\nBest starting frames (longest survival):")
    for label, s in sorted(results.items(), key=lambda x: -x[1])[:3]:
        print(f"  {label}: {s} steps ({s*dt:.2f}s)")
    print(f"\nVideos: {output_dir}/videos/")


if __name__ == "__main__":
    main()
