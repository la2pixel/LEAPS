"""Render EMG replay simulation using Gait10dof18Musc (LocoMuJoCo env).

Runs entirely on the cluster using EGL offscreen rendering — no display needed.

Usage:
    python -m leaps.scripts.render_emg \
        --data /fast/lsivakumar/data/processed/emg_activations.h5 \
        --output outputs/emg_replay.mp4 \
        --duration 5
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np

import leaps.envs  # registers Gait10dof18Musc + HeightJointTerminalStateHandler
from leaps.data.stride_dataset import load_strides
from leaps.envs.emg_mapping import EMGToMuscleMapper
from leaps.envs.gait10dof_env import Gait10dof18Musc

DT = 0.005  # model timestep
FPS = 50


def load_emg_actions(data_path: str, n_steps: int, seed: int = 42) -> np.ndarray:
    """Load EMG strides and map to 18 muscle activations in [-1, 1] (LocoMuJoCo scale)."""
    mapper = EMGToMuscleMapper()
    strides, metadata = load_strides(data_path, with_metadata=True)

    # Prefer treadmill (steady-state walking)
    mask = metadata["modes"] == "treadmill"
    if mask.any():
        strides = strides[mask]

    rng = np.random.default_rng(seed)
    n_needed = (n_steps // 101) + 2
    start = rng.integers(0, max(1, len(strides) - n_needed))
    selected = strides[start : start + n_needed]

    actions = np.concatenate(
        [mapper.map_stride_with_phase_offset(s) for s in selected], axis=0
    )[:n_steps]

    actions = np.clip(actions, 0, 1).astype(np.float32)
    # LocoMuJoCo expects [-1, 1]; it maps to [0, 1] for muscles internally
    return 2.0 * actions - 1.0


def simulate_and_render(data_path: str, output_path: str, duration: float, seed: int):
    n_steps = int(duration / DT)
    print(f"Duration: {duration}s → {n_steps} steps")

    print("Loading EMG actions...")
    actions = load_emg_actions(data_path, n_steps, seed=seed)

    print("Creating environment...")
    env = Gait10dof18Musc.generate()
    env.reset()

    model = env._model
    data = env._data

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []

    print("Simulating + rendering...")
    for t in range(n_steps):
        env.step(actions[t])
        if t % (n_steps // 10) == 0:
            print(f"  {t}/{n_steps} steps ({100*t//n_steps}%)")
        renderer.update_scene(data)
        frames.append(renderer.render().copy())

    renderer.close()
    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=FPS)
    print(f"Saved {len(frames)} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to HDF5 EMG data.")
    parser.add_argument("--output", default="outputs/emg_replay.mp4")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to simulate.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    simulate_and_render(args.data, args.output, args.duration, args.seed)


if __name__ == "__main__":
    main()
