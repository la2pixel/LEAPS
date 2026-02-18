"""Generate simulation trajectories on cluster, render locally.

Two-step workflow:
  Step 1 (cluster): Generate action sequences and simulate physics.
      Saves trajectories (qpos per timestep) as .npz files — no rendering needed.
      python -m leaps.scripts.render_comparison generate \
          --data /fast/lsivakumar/data/processed/emg_activations.h5 \
          --output renders/trajectories.npz --duration 5

  Step 2 (local Windows): Load trajectories, render video with MuJoCo viewer.
      python -m leaps.scripts.render_comparison render \
          --trajectories renders/trajectories.npz \
          --output renders/comparison.mp4
"""

import argparse
from pathlib import Path

import numpy as np

from leaps.envs.emg_mapping import EMGToMuscleMapper
from leaps.envs.muscle_env import MuscleHumanoidEnv


# ── Action generation ─────────────────────────────────────────────────


def generate_random_actions(n_steps: int, n_actuators: int = 18, seed: int = 42) -> np.ndarray:
    """Uniform random muscle activations in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(n_steps, n_actuators)).astype(np.float32)


def generate_emg_replay_actions(
    n_steps: int,
    data_path: str,
    mapper: EMGToMuscleMapper,
    seed: int = 42,
) -> np.ndarray:
    """Replay real EMG strides through the muscle mapping.

    Picks consecutive strides from a random treadmill walking subject
    for a natural-looking replay, with left leg phase-shifted by 50%.
    """
    from leaps.data.stride_dataset import load_strides

    strides, metadata = load_strides(data_path, with_metadata=True)

    # Prefer treadmill strides (steady-state walking)
    treadmill_mask = metadata["modes"] == "treadmill"
    if treadmill_mask.any():
        strides = strides[treadmill_mask]

    rng = np.random.default_rng(seed)
    n_strides_needed = (n_steps // 101) + 2

    # Pick a random starting point and take consecutive strides
    start = rng.integers(0, max(1, len(strides) - n_strides_needed))
    selected = strides[start : start + n_strides_needed]

    actions_list = []
    for stride in selected:
        mapped = mapper.map_stride_with_phase_offset(stride)  # (101, 18)
        actions_list.append(mapped)

    actions = np.concatenate(actions_list, axis=0)[:n_steps]
    return np.clip(actions, 0, 1).astype(np.float32)


def generate_latent_actions(
    n_steps: int,
    model,
    mapper: EMGToMuscleMapper,
    data_path: str,
    seed: int = 42,
) -> np.ndarray:
    """Encode real strides → latent → decode → map to 18 muscles.

    Instead of random latent walk, this encodes actual EMG strides,
    passes them through the bottleneck, and applies the decoded output.
    Shows what the autoencoder's reconstruction looks like in simulation.
    """
    from leaps.data.stride_dataset import load_strides

    strides, metadata = load_strides(data_path, with_metadata=True)

    # Same treadmill strides as replay for fair comparison
    treadmill_mask = metadata["modes"] == "treadmill"
    if treadmill_mask.any():
        strides = strides[treadmill_mask]

    rng = np.random.default_rng(seed)
    n_strides_needed = (n_steps // 101) + 2
    start = rng.integers(0, max(1, len(strides) - n_strides_needed))
    selected = strides[start : start + n_strides_needed]

    # Encode → decode (pass through bottleneck)
    reconstructed = model.reconstruct(selected.astype(np.float32))  # (n, 101, 11)

    actions_list = []
    for stride in reconstructed:
        mapped = mapper.map_stride_with_phase_offset(stride)
        actions_list.append(mapped)

    actions = np.concatenate(actions_list, axis=0)[:n_steps]
    return np.clip(actions, 0, 1).astype(np.float32)


# ── Simulation (cluster) ─────────────────────────────────────────────


def simulate(env, actions: np.ndarray) -> np.ndarray:
    """Run physics simulation, return qpos trajectory for replay.

    Works with both MuscleHumanoidEnv (raw MuJoCo) and LocoMuJoCo environments.
    """
    env.reset()
    is_loco = hasattr(env, "_model")  # LocoMuJoCo uses _model, raw uses model
    model = env._model if is_loco else env.model
    data = env._data if is_loco else env.data
    qpos_traj = np.zeros((len(actions), model.nq))
    for t in range(len(actions)):
        env.step(actions[t])
        qpos_traj[t] = data.qpos.copy()
    return qpos_traj


# ── Rendering (local) ────────────────────────────────────────────────


def render_video(
    qpos_random: np.ndarray,
    qpos_emg: np.ndarray,
    xml_path: str,
    output_path: str,
    fps: int = 50,
    subsample: int = 4,
    width: int = 640,
    height: int = 480,
):
    """Render side-by-side video from saved qpos trajectories."""
    import imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)

    n = min(len(qpos_random), len(qpos_emg))
    writer = imageio.get_writer(output_path, fps=fps)

    for t in range(0, n, subsample):
        # Random side
        data.qpos[:] = qpos_random[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frame_left = renderer.render().copy()

        # EMG side
        data.qpos[:] = qpos_emg[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frame_right = renderer.render().copy()

        combined = np.concatenate([frame_left, frame_right], axis=1)
        writer.append_data(combined)

    writer.close()
    renderer.close()
    print(f"Saved {n // subsample} frames to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate simulation trajectories (cluster) or render video (local)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate (runs on cluster, no rendering) ---
    gen = sub.add_parser("generate", help="Simulate physics and save qpos trajectories.")
    gen.add_argument("--data", required=True, help="Path to HDF5 EMG data.")
    gen.add_argument("--checkpoint", default=None, help="Trained model checkpoint (.pt).")
    gen.add_argument("--model", default="StrideFlatAE", help="Model name for --checkpoint.")
    gen.add_argument("--latent-dim", type=int, default=16, help="Latent dim for --checkpoint.")
    gen.add_argument("--xml", default=None, help="Path to humanoid XML (default: built-in).")
    gen.add_argument("--output", default="renders/trajectories.npz", help="Output .npz file.")
    gen.add_argument("--duration", type=float, default=5.0, help="Simulation duration (seconds).")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument(
        "--use-loco", action="store_true",
        help="Use LocoMuJoCo Gait10dof18Musc env instead of raw MuscleHumanoidEnv.",
    )

    # --- render (runs locally with display) ---
    ren = sub.add_parser("render", help="Render video from saved trajectories.")
    ren.add_argument("--trajectories", required=True, help="Path to .npz from generate step.")
    ren.add_argument("--xml", default=None, help="Path to humanoid XML.")
    ren.add_argument("--output", default="renders/comparison.mp4", help="Output video path.")
    ren.add_argument("--fps", type=int, default=50)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.command == "generate":
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mapper = EMGToMuscleMapper()
        dt = 0.005
        n_steps = int(args.duration / dt)
        print(f"Duration: {args.duration}s → {n_steps} steps (dt={dt})")

        # Generate action sequences
        print("Generating random actions...")
        random_actions = generate_random_actions(n_steps, seed=args.seed)

        print("Generating EMG replay actions...")
        emg_replay_actions = generate_emg_replay_actions(n_steps, args.data, mapper, seed=args.seed)

        # Optionally generate AE-decoded actions
        ae_actions = None
        if args.checkpoint:
            print(f"Loading {args.model} d={args.latent_dim} from {args.checkpoint}...")
            from leaps.training.stride_trainer import build_stride_model
            model = build_stride_model(args.model, args.latent_dim)
            model.load(args.checkpoint)
            print("Generating AE-decoded actions...")
            ae_actions = generate_latent_actions(n_steps, model, mapper, args.data, seed=args.seed)

        # Simulate all
        if args.use_loco:
            from leaps.envs import Gait10dof18Musc

            env = Gait10dof18Musc.generate()
            # LocoMuJoCo expects actions in [-1, 1], maps to [0, 1] for muscles
            random_actions = 2.0 * random_actions - 1.0
            emg_replay_actions = 2.0 * emg_replay_actions - 1.0
            if ae_actions is not None:
                ae_actions = 2.0 * ae_actions - 1.0
            print("Using LocoMuJoCo Gait10dof18Musc env")
        else:
            env = MuscleHumanoidEnv(xml_path=args.xml)

        print("Simulating random actions...")
        qpos_random = simulate(env, random_actions)

        print("Simulating EMG replay actions...")
        qpos_emg = simulate(env, emg_replay_actions)

        save_dict = {
            "qpos_random": qpos_random,
            "qpos_emg_replay": qpos_emg,
            "actions_random": random_actions,
            "actions_emg_replay": emg_replay_actions,
            "dt": dt,
            "duration": args.duration,
        }

        if ae_actions is not None:
            print("Simulating AE-decoded actions...")
            qpos_ae = simulate(env, ae_actions)
            save_dict["qpos_ae"] = qpos_ae
            save_dict["actions_ae"] = ae_actions

        env.close()

        np.savez_compressed(str(output_path), **save_dict)
        print(f"\nTrajectories saved to {output_path}")
        print(f"  Keys: {list(save_dict.keys())}")
        print(f"  Copy to local machine and run:")
        print(f"    python -m leaps.scripts.render_comparison render \\")
        print(f"      --trajectories {output_path} --output renders/comparison.mp4")

    elif args.command == "render":
        from leaps.envs.muscle_env import DEFAULT_XML

        traj = np.load(args.trajectories)
        xml_path = str(args.xml or DEFAULT_XML)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Available trajectories: {list(traj.keys())}")

        # Render random vs EMG replay
        print("Rendering random vs EMG replay...")
        render_video(
            traj["qpos_random"], traj["qpos_emg_replay"],
            xml_path, str(output_path), fps=args.fps,
        )

        # If AE trajectory exists, render random vs AE too
        if "qpos_ae" in traj:
            ae_path = str(output_path.with_stem(output_path.stem + "_ae"))
            print("Rendering random vs AE-decoded...")
            render_video(
                traj["qpos_random"], traj["qpos_ae"],
                xml_path, ae_path, fps=args.fps,
            )

        print("Done!")


if __name__ == "__main__":
    main()
