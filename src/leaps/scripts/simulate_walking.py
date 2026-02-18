"""Walking simulation scenarios for the gait10dof18musc humanoid.

Three subcommands:
  emg-replay      Open-loop playback of real EMG data through the simulator.
  representation  Encode→decode EMG through trained models, then simulate.
  ppo             Closed-loop RL training/eval with SB3 PPO.

All simulation commands support --record to save MP4 video via LocoMuJoCo's
headless renderer. Videos are saved alongside the .npz trajectory files.

Usage:
    python -m leaps.scripts.simulate_walking emg-replay \
        --data DATA --output results/emg_replay.npz --record

    python -m leaps.scripts.simulate_walking representation \
        --data DATA --checkpoint-dir DIR --record

    python -m leaps.scripts.simulate_walking ppo \
        --variant direct --train --total-timesteps 1000000
"""

import argparse
import os
import time
from pathlib import Path

# Force JAX to CPU — MuJoCo sim doesn't need GPU, avoids CUDA_ERROR_NO_DEVICE
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Environment creation (uses LocoMuJoCo task factory pattern)
# ---------------------------------------------------------------------------


def _ensure_registered():
    """Register custom LocoMuJoCo components (idempotent)."""
    from leaps.envs.gait10dof_env import Gait10dof18Musc
    from leaps.envs.rewards import ForwardVelocityReward
    from leaps.envs.terminal_state import HeightJointTerminalStateHandler

    HeightJointTerminalStateHandler.register()
    ForwardVelocityReward.register()
    Gait10dof18Musc.register()


def make_env(
    record: bool = False,
    video_path: str = "./recordings",
    video_name: str = "recording",
    no_terminal: bool = False,
    target_velocity: float = 1.25,
):
    """Create Gait10dof18Musc env with LocoMuJoCo viewer for recording.

    Args:
        record: If True, record video via LocoMuJoCo's headless renderer.
        video_path: Directory for recorded video.
        video_name: Video filename (without extension).
        no_terminal: Disable early termination (let it fall).
        target_velocity: Target forward velocity for reward.

    Returns:
        Gait10dof18Musc environment instance.
    """
    _ensure_registered()
    from leaps.envs.gait10dof_env import Gait10dof18Musc

    terminal_type = "NoTerminalStateHandler" if no_terminal else "HeightJointTerminalStateHandler"

    viewer_kwargs = {"headless": True}
    if record:
        Path(video_path).mkdir(parents=True, exist_ok=True)
        # recorder_params passed as viewer kwarg; record=True passed to env.render()
        viewer_kwargs["recorder_params"] = {
            "path": video_path,
            "tag": "",  # no timestamp subfolder
            "video_name": video_name,
            "fps": 100,  # 1/dt = 1/0.01 = 100
            "compress": True,
        }

    env = Gait10dof18Musc(
        terminal_state_type=terminal_type,
        goal_type="NoGoal",
        reward_type="ForwardVelocityReward",
        reward_params={"target_velocity": target_velocity},
        **viewer_kwargs,
    )
    return env


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def simulate_open_loop(env, actions_sequence, max_steps=None, record=False):
    """Run open-loop simulation, cycling through a fixed action sequence.

    Args:
        env: LocoMuJoCo Gait10dof18Musc environment.
        actions_sequence: (T, 18) array of muscle commands in [-1, 1].
        max_steps: Maximum simulation steps (default: len(actions_sequence)).
        record: If True, call env.render(record=True) each step.

    Returns:
        Dict with qpos, qvel, obs, rewards, actions arrays.
    """
    if max_steps is None:
        max_steps = len(actions_sequence)

    obs = env.reset(key=jax.random.key(0))
    if record:
        env.render(record=True)

    all_obs = [np.asarray(obs)]
    all_qpos = [np.array(env._data.qpos)]
    all_qvel = [np.array(env._data.qvel)]
    all_rewards = []
    all_actions = []

    for step in range(max_steps):
        action = actions_sequence[step % len(actions_sequence)]
        obs, reward, terminated, truncated, info = env.step(np.asarray(action))

        if record:
            env.render(record=True)

        all_obs.append(np.asarray(obs))
        all_qpos.append(np.array(env._data.qpos))
        all_qvel.append(np.array(env._data.qvel))
        all_rewards.append(float(reward))
        all_actions.append(np.asarray(action))

        if terminated or truncated:
            print(f"  Episode ended at step {step + 1} "
                  f"(terminated={terminated}, truncated={truncated})")
            break

    return {
        "obs": np.array(all_obs),
        "qpos": np.array(all_qpos),
        "qvel": np.array(all_qvel),
        "rewards": np.array(all_rewards),
        "actions": np.array(all_actions),
    }


def compute_sim_metrics(results):
    """Compute summary metrics from simulation results."""
    qvel = results["qvel"]
    qpos = results["qpos"]
    rewards = results["rewards"]

    forward_vel = qvel[:, 0]
    pelvis_height = qpos[:, 1]

    return {
        "forward_vel_mean": float(np.mean(forward_vel)),
        "forward_vel_std": float(np.std(forward_vel)),
        "pelvis_height_mean": float(np.mean(pelvis_height)),
        "pelvis_height_std": float(np.std(pelvis_height)),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
        "episode_length": len(rewards),
    }


def _print_metrics(metrics, label=""):
    """Print simulation metrics."""
    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}Episode length: {metrics['episode_length']}")
    print(f"  {prefix}Forward velocity: {metrics['forward_vel_mean']:.3f} "
          f"± {metrics['forward_vel_std']:.3f} m/s")
    print(f"  {prefix}Pelvis height: {metrics['pelvis_height_mean']:.3f} "
          f"± {metrics['pelvis_height_std']:.3f} m")
    print(f"  {prefix}Total reward: {metrics['total_reward']:.2f} "
          f"(mean: {metrics['mean_reward']:.4f})")


def _prepare_emg_actions(strides, n_strides=50):
    """Map EMG strides → (T, 18) LocoMuJoCo actions in [-1, 1]."""
    from leaps.envs.emg_mapping import EMGToMuscleMapper

    mapper = EMGToMuscleMapper()
    n = min(strides.shape[0], n_strides)
    all_actions = []
    for i in range(n):
        stride_18 = mapper.map_stride_with_phase_offset(strides[i])  # (101, 18) in [0,1]
        all_actions.append(2.0 * stride_18 - 1.0)  # → [-1, 1]
    return np.concatenate(all_actions, axis=0)


# ---------------------------------------------------------------------------
# EMG Replay
# ---------------------------------------------------------------------------


def cmd_emg_replay(args):
    """Open-loop EMG replay through the simulator."""
    from leaps.data.stride_dataset import load_strides

    print("=== EMG Replay ===")
    print(f"Loading strides from {args.data}")
    strides = load_strides(args.data)
    print(f"  {strides.shape[0]} strides, shape {strides.shape}")

    actions_seq = _prepare_emg_actions(strides)
    print(f"  Action sequence: {actions_seq.shape}")

    output = Path(args.output)
    video_dir = str(output.parent / "videos")
    video_name = output.stem

    env = make_env(
        record=args.record,
        video_path=video_dir,
        video_name=video_name,
        no_terminal=args.no_terminal,
    )
    dt = env.info.dt
    max_steps = int(args.duration / dt) if args.duration else len(actions_seq)
    terminal = "disabled" if args.no_terminal else "enabled"
    print(f"  dt={dt:.4f}s, {max_steps} steps (~{max_steps * dt:.1f}s), terminal={terminal}")
    if args.record:
        print(f"  Recording video to {video_dir}/{video_name}.mp4")

    results = simulate_open_loop(env, actions_seq, max_steps=max_steps, record=args.record)
    metrics = compute_sim_metrics(results)
    _print_metrics(metrics, "EMG Replay")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **results, **{f"metric_{k}": v for k, v in metrics.items()})
    print(f"  Trajectory saved to {output}")

    env.stop()


# ---------------------------------------------------------------------------
# Representation comparison
# ---------------------------------------------------------------------------


def cmd_representation(args):
    """Simulate with model-reconstructed EMG (encode → decode → simulate)."""
    from leaps.data.stride_dataset import load_strides
    from leaps.envs.emg_mapping import EMGToMuscleMapper
    from leaps.training.stride_trainer import build_stride_model

    _ensure_registered()

    print("=== Representation Comparison ===")
    print(f"Loading strides from {args.data}")
    strides = load_strides(args.data)
    print(f"  {strides.shape[0]} strides, shape {strides.shape}")

    mapper = EMGToMuscleMapper()
    ckpt_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_strides = min(strides.shape[0], 50)
    test_strides = strides[:n_strides]

    # Checkpoint naming: trainer saves with model.name (class __name__)
    _class_names = {
        "StridePCA": "StridePCAModel",
        "StrideNMF": "StrideNMFModel",
        "StrideCNMF": "StrideCNMFModel",
        "StrideAE": "StrideAutoencoder",
        "StrideVAE": "StrideVAE",
        "StrideMAE": "StrideMaskedAutoencoder",
        "StrideWAE": "StrideWAE_MMD",
        "StrideFlatAE": "StrideFlatAE",
    }
    _pytorch_models = {"StrideAE", "StrideVAE", "StrideMAE", "StrideWAE", "StrideFlatAE"}

    for model_name in args.models:
        print(f"\n--- {model_name} (d={args.latent_dim}) ---")

        model = build_stride_model(model_name, args.latent_dim)
        ext = ".pt" if model_name in _pytorch_models else ".pkl"

        # Try both display name and class name for checkpoint lookup
        candidates = [
            ckpt_dir / f"{model_name}_d{args.latent_dim}{ext}",
            ckpt_dir / f"{_class_names.get(model_name, model_name)}_d{args.latent_dim}{ext}",
        ]
        ckpt_path = None
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break
        if ckpt_path is None:
            print(f"  WARNING: No checkpoint found, tried: {[str(c) for c in candidates]}")
            continue

        print(f"  Loading: {ckpt_path}")
        model.load(ckpt_path)

        # Reconstruct → map → simulate
        recon = np.clip(model.reconstruct(test_strides), 0.0, 1.0)
        all_actions = []
        for i in range(n_strides):
            stride_18 = mapper.map_stride_with_phase_offset(recon[i])
            all_actions.append(2.0 * stride_18 - 1.0)
        actions_seq = np.concatenate(all_actions, axis=0)

        video_name = f"{model_name}_d{args.latent_dim}"
        env = make_env(
            record=args.record,
            video_path=str(output_dir / "videos"),
            video_name=video_name,
            no_terminal=args.no_terminal,
        )
        dt = env.info.dt
        max_steps = int(args.duration / dt) if args.duration else len(actions_seq)
        print(f"  Simulating {max_steps} steps")

        results = simulate_open_loop(env, actions_seq, max_steps=max_steps, record=args.record)
        metrics = compute_sim_metrics(results)
        _print_metrics(metrics, model_name)

        out_path = output_dir / f"{video_name}.npz"
        np.savez(out_path, **results, **{f"metric_{k}": v for k, v in metrics.items()})
        print(f"  Saved to {out_path}")

        env.stop()


# ---------------------------------------------------------------------------
# PPO training / evaluation
# ---------------------------------------------------------------------------


def cmd_ppo(args):
    """Train or evaluate SB3 PPO on the gait10dof18musc env."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    from leaps.envs.gym_wrapper import Gait10dof18MuscGymEnv, LatentActionGymEnv

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== PPO ({args.variant}) ===")

    if args.variant == "direct":
        env = Gait10dof18MuscGymEnv(
            target_velocity=1.25,
            max_episode_steps=1000,
        )
    elif args.variant == "latent":
        if args.decoder_checkpoint is None:
            raise ValueError("--decoder-checkpoint required for latent variant")

        from leaps.training.stride_trainer import build_stride_model

        ae_model = build_stride_model("StrideFlatAE", args.latent_dim)
        print(f"  Loading decoder from {args.decoder_checkpoint}")
        ae_model.load(args.decoder_checkpoint)

        for param in ae_model.decoder.parameters():
            param.requires_grad = False
        ae_model._module.eval()

        env = LatentActionGymEnv(
            decoder_fn=ae_model.decode,
            latent_dim=args.latent_dim,
            target_velocity=1.25,
            max_episode_steps=1000,
        )
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    if args.train:
        print(f"  Training for {args.total_timesteps} timesteps")

        wandb_callback = None
        if args.wandb:
            try:
                import wandb
                from wandb.integration.sb3 import WandbCallback

                wandb.init(
                    project=args.wandb_project,
                    name=f"ppo_{args.variant}_{'latent' + str(args.latent_dim) if args.variant == 'latent' else 'direct18'}",
                    config={
                        "variant": args.variant,
                        "total_timesteps": args.total_timesteps,
                        "latent_dim": args.latent_dim if args.variant == "latent" else 18,
                    },
                    sync_tensorboard=True,
                )
                wandb_callback = WandbCallback(verbose=2)
            except ImportError:
                print("  WARNING: wandb not installed, skipping.")

        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=str(output_dir / "checkpoints"),
            name_prefix=f"ppo_{args.variant}",
        )
        callbacks = [checkpoint_cb]
        if wandb_callback is not None:
            callbacks.append(wandb_callback)

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            tensorboard_log=str(output_dir / "tb_logs"),
            device="auto",
        )

        t0 = time.time()
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        elapsed = time.time() - t0
        print(f"  Training completed in {elapsed:.1f}s")

        final_path = output_dir / f"ppo_{args.variant}_final"
        model.save(str(final_path))
        print(f"  Saved final model to {final_path}")

        if args.wandb:
            try:
                wandb.finish()
            except Exception:
                pass

    if args.eval:
        model_path = output_dir / f"ppo_{args.variant}_final.zip"
        if not model_path.exists():
            ckpt_dir = output_dir / "checkpoints"
            if ckpt_dir.exists():
                ckpts = sorted(ckpt_dir.glob(f"ppo_{args.variant}_*.zip"))
                if ckpts:
                    model_path = ckpts[-1]
                else:
                    print(f"  ERROR: No model found at {model_path}")
                    env.close()
                    return
            else:
                print(f"  ERROR: No model found at {model_path}")
                env.close()
                return

        print(f"  Evaluating model from {model_path}")
        model = PPO.load(str(model_path), env=env)

        n_eval_episodes = 10
        all_rewards = []
        all_lengths = []

        for ep in range(n_eval_episodes):
            obs, _ = env.reset(seed=ep)
            ep_reward = 0.0
            ep_len = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_len += 1
                done = terminated or truncated
            all_rewards.append(ep_reward)
            all_lengths.append(ep_len)
            print(f"    Episode {ep + 1}: reward={ep_reward:.2f}, length={ep_len}")

        print(f"\n  Mean reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
        print(f"  Mean length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")

        eval_path = output_dir / f"eval_{args.variant}.npz"
        np.savez(eval_path, rewards=np.array(all_rewards), lengths=np.array(all_lengths))
        print(f"  Saved eval results to {eval_path}")

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Walking simulation scenarios for gait10dof18musc.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- emg-replay ---
    p_emg = subparsers.add_parser("emg-replay", help="Open-loop EMG replay simulation.")
    p_emg.add_argument("--data", required=True, help="Path to emg_activations.h5")
    p_emg.add_argument("--output", default="results/emg_replay.npz", help="Output .npz path")
    p_emg.add_argument("--duration", type=float, default=10.0, help="Simulation duration (seconds)")
    p_emg.add_argument("--seed", type=int, default=42)
    p_emg.add_argument("--no-terminal", action="store_true", help="Disable early termination")
    p_emg.add_argument("--record", action="store_true", help="Record MP4 video via LocoMuJoCo")

    # --- representation ---
    p_rep = subparsers.add_parser("representation", help="Reconstruct EMG through models, then simulate.")
    p_rep.add_argument("--data", required=True, help="Path to emg_activations.h5")
    p_rep.add_argument("--checkpoint-dir", required=True, help="Directory with model checkpoints")
    p_rep.add_argument("--latent-dim", type=int, default=9, help="Latent dimension")
    p_rep.add_argument(
        "--models", nargs="+", default=["StridePCA", "StrideNMF", "StrideFlatAE"],
        help="Models to compare",
    )
    p_rep.add_argument("--output-dir", default="results/representation", help="Output directory")
    p_rep.add_argument("--duration", type=float, default=None, help="Sim duration (default: all strides)")
    p_rep.add_argument("--no-terminal", action="store_true", help="Disable early termination")
    p_rep.add_argument("--record", action="store_true", help="Record MP4 video via LocoMuJoCo")

    # --- ppo ---
    p_ppo = subparsers.add_parser("ppo", help="PPO training / evaluation.")
    p_ppo.add_argument("--variant", choices=["direct", "latent"], required=True)
    p_ppo.add_argument("--train", action="store_true", help="Train PPO")
    p_ppo.add_argument("--eval", action="store_true", help="Evaluate PPO")
    p_ppo.add_argument("--total-timesteps", type=int, default=1_000_000)
    p_ppo.add_argument("--output-dir", default="experiments/ppo")
    p_ppo.add_argument("--decoder-checkpoint", default=None, help="StrideFlatAE checkpoint (for latent)")
    p_ppo.add_argument("--latent-dim", type=int, default=9)
    p_ppo.add_argument("--wandb", action="store_true")
    p_ppo.add_argument("--wandb-project", default="leaps")

    args = parser.parse_args(argv)

    if args.command == "emg-replay":
        cmd_emg_replay(args)
    elif args.command == "representation":
        cmd_representation(args)
    elif args.command == "ppo":
        if not args.train and not args.eval:
            parser.error("At least one of --train or --eval required for ppo command.")
        cmd_ppo(args)


if __name__ == "__main__":
    main()
