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

from leaps.data.metadata import LEAPS_H5_PATH

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
        reward_type="WalkingReward",
        reward_params={"target_velocity": target_velocity},
        **viewer_kwargs,
    )
    return env


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def simulate_open_loop(env, actions_sequence, max_steps=None, record=False, hang_mode=None):
    """Run open-loop simulation, cycling through a fixed action sequence.

    Args:
        env: LocoMuJoCo Gait10dof18Musc environment.
        actions_sequence: (T, 18) array of muscle commands in [-1, 1].
        max_steps: Maximum simulation steps (default: len(actions_sequence)).
        record: If True, call env.render(record=True) each step.
        hang_mode: None | "air" | "place"
            None  — normal simulation, skeleton can fall.
            "air" — pelvis pinned at 1.5m, ground disabled, joints damped.
                    Observe pure leg swing without ground contact.
            "place" — pelvis tx/ty pinned at natural standing height (0.95m),
                    ground active. Walking in place with proper foot contact.

    Returns:
        Dict with qpos, qvel, obs, rewards, actions arrays.
    """
    import mujoco

    if max_steps is None:
        max_steps = len(actions_sequence)

    obs = env.reset(key=jax.random.key(0))

    # Reset to keyframe for a proper standing pose in both hang modes
    if hang_mode is not None:
        keyframe_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_KEY, "default-pose")
        if keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(env._model, env._data, keyframe_id)

    if hang_mode == "air":
        # Suspend at 1.5m — feet hang ~0.55m above ground
        pelvis_tx0 = 0.0
        pelvis_ty0 = 1.5

        # Actually move the skeleton to hang height before first step
        env._data.qpos[0] = pelvis_tx0
        env._data.qpos[1] = pelvis_ty0
        env._data.qvel[:] = 0.0
        mujoco.mj_forward(env._model, env._data)

        # Disable ground: no contact, not visible
        ground_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "ground-plane")
        if ground_id >= 0:
            env._model.geom_contype[ground_id] = 0
            env._model.geom_conaffinity[ground_id] = 0
            env._model.geom_rgba[ground_id, 3] = 0.0


    elif hang_mode == "place":
        # Pin at natural standing height, ground stays active
        pelvis_tx0 = 0.0
        pelvis_ty0 = float(env._data.qpos[1])  # 0.95m from keyframe

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

        if hang_mode is not None:
            env._data.qpos[0] = pelvis_tx0
            env._data.qpos[1] = pelvis_ty0
            env._data.qvel[0] = 0.0
            env._data.qvel[1] = 0.0
            mujoco.mj_forward(env._model, env._data)

        if record:
            env.render(record=True)

        all_obs.append(np.asarray(obs))
        all_qpos.append(np.array(env._data.qpos))
        all_qvel.append(np.array(env._data.qvel))
        all_rewards.append(float(reward))
        all_actions.append(np.asarray(action))

        if terminated or truncated:
            if hang_mode is None:
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
    if args.hang_mode == "air":
        print("  Hang mode: air — pelvis at 1.5m, no ground, joints damped")
    elif args.hang_mode == "place":
        print("  Hang mode: place — walking in place at 0.95m, ground active")
    if args.record:
        print(f"  Recording video to {video_dir}/{video_name}.mp4")

    results = simulate_open_loop(env, actions_seq, max_steps=max_steps, record=args.record, hang_mode=args.hang_mode)
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
# PPO helpers
# ---------------------------------------------------------------------------


class _VecNormCheckpointCallback:
    """Mixin-style callback: saves VecNormalize stats alongside every checkpoint.

    Used together with CheckpointCallback. SB3's CheckpointCallback only saves
    the policy .zip file — this callback saves the matching _vecnorm.pkl so that
    every checkpoint is self-contained and loadable for evaluation even if the
    job is killed mid-run.

    Usage:
        checkpoint_cb = CheckpointCallback(save_freq=..., save_path=..., name_prefix=...)
        vecnorm_cb = _VecNormCheckpointCallback(train_env, checkpoint_cb)
    """

    def __new__(cls, train_env_vecnorm, checkpoint_cb):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Impl(BaseCallback):
            def __init__(self):
                super().__init__()
                self._vec_env = train_env_vecnorm
                self._ckpt_cb = checkpoint_cb

            def _on_step(self) -> bool:
                # n_calls is incremented before _on_step, same as CheckpointCallback,
                # so this triggers at exactly the same steps.
                if self.n_calls % self._ckpt_cb.save_freq == 0:
                    path = (
                        f"{self._ckpt_cb.save_path}/"
                        f"{self._ckpt_cb.name_prefix}_{self.num_timesteps}_steps_vecnorm.pkl"
                    )
                    self._vec_env.save(path)
                return True

        return _Impl()


def _make_env_fn(variant: str, env_kwargs: dict, seed: int):
    """Return a callable that creates the env for SubprocVecEnv.

    Uses functools.partial so the returned callable is picklable.
    All env_kwargs values must also be picklable (numpy arrays are fine;
    PyTorch models are NOT — use decoder_weights dict instead).
    """
    import functools
    return functools.partial(_subprocess_env_init, variant, env_kwargs, seed)


def _subprocess_env_init(variant: str, env_kwargs: dict, seed: int):
    """Module-level env factory for SubprocVecEnv workers.

    Must be at module level (not a closure) so it is picklable by Python's
    multiprocessing 'spawn' start method. Closures / local functions are NOT
    picklable and would crash SubprocVecEnv immediately.

    Each worker process calls this once at startup to build its env.
    """
    from stable_baselines3.common.monitor import Monitor

    from leaps.envs.gym_wrapper import (
        Gait10dof18MuscGymEnv,
        LatentActionGymEnv,
        ResidualLatentGymEnv,
    )

    if variant == "direct":
        env = Gait10dof18MuscGymEnv(**env_kwargs)
    elif variant == "latent":
        env = LatentActionGymEnv(**env_kwargs)
    elif variant == "residual":
        env = ResidualLatentGymEnv(**env_kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    env = Monitor(env)
    env.reset(seed=seed)
    return env


def cmd_ppo(args):
    """Train or evaluate SB3 PPO on the gait10dof18musc env."""
    import random

    import numpy as np
    import torch

    # Reproducibility: fix all random sources before anything else.
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from leaps.envs.gym_wrapper import (
        Gait10dof18MuscGymEnv,
        LatentActionGymEnv,
        ResidualLatentGymEnv,
        load_flatae_decoder_weights,
        load_hausdorfer_decoder_weights,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_envs = max(1, args.n_envs)
    print(f"=== PPO ({args.variant}, {n_envs} envs) ===")

    # Build env_kwargs (all values must be picklable for SubprocVecEnv)
    common_kwargs = {"max_episode_steps": 1000, "headless": True, "effort_weight": args.effort_weight, "upright_weight": args.upright_weight, "tilt_limit": 0.8}

    if args.variant == "direct":
        env_kwargs = {"target_velocity": 1.25, "obs_type": args.obs_type, **common_kwargs}

    elif args.variant in ("latent", "residual"):
        if args.decoder_checkpoint is None:
            raise ValueError(f"--decoder-checkpoint required for {args.variant} variant")
        print(f"  Loading {args.decoder_type} decoder from {args.decoder_checkpoint}")
        if args.decoder_type == "snapshot":
            decoder_weights = load_hausdorfer_decoder_weights(args.decoder_checkpoint)
        else:
            decoder_weights = load_flatae_decoder_weights(args.decoder_checkpoint)

        env_kwargs = {
            "decoder_weights": decoder_weights,
            "latent_dim": args.latent_dim,
            "target_velocity": 1.25,
            "stride_duration": args.stride_duration,
            "obs_type": args.obs_type,
            **common_kwargs,
        }
        if args.variant == "residual":
            env_kwargs["residual_weight"] = args.residual_weight

    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize

    # ── Hyperparameters (paper Table I — Humanoid-v4 settings) ───────────────
    # n_steps × n_envs = 2048 × 4 = 8192 total rollout per update (paper-exact)
    N_STEPS_PER_ENV = 2048          # per-env rollout steps (paper: 2048 for humanoid)
    BATCH_SIZE      = args.batch_size  # SB3 default 64; paper does not override for humanoid
    NET_ARCH        = [512, 512]    # paper: 512×512 for humanoid

    # ── Build env factory helper ──────────────────────────────────────────
    def _make_single_env():
        if args.variant == "direct":
            e = Gait10dof18MuscGymEnv(**env_kwargs)
        elif args.variant == "latent":
            e = LatentActionGymEnv(**env_kwargs)
        else:
            e = ResidualLatentGymEnv(**env_kwargs)
        return Monitor(e)

    # ── Eval env (always needed: for EvalCallback during train, or for --eval)
    # DummyVecEnv wraps a single Monitor env. VecNormalize with training=False
    # means the running stats are only READ here, never updated. EvalCallback
    # calls sync_envs_normalization(train_env, eval_env) before each eval run
    # to copy the latest stats from training, so obs are normalised identically.
    eval_vec = DummyVecEnv([_make_single_env])
    eval_env = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    _act = eval_vec.action_space
    _obs = eval_vec.observation_space
    print(f"  Action space:      {_act}")
    print(f"  Observation space: {_obs}")
    print(f"  n_steps/env={N_STEPS_PER_ENV}, batch={BATCH_SIZE}, "
          f"total/update={N_STEPS_PER_ENV * n_envs}, net={NET_ARCH}")

    # ── Training env: only spawned when --train is requested ─────────────
    # SubprocVecEnv spawns N independent worker processes (real parallelism).
    # "spawn" is safe on cluster — workers import fresh, no risk of inheriting
    # partially-initialised JAX/CUDA state.
    # VecNormalize: norm_obs=True matches paper exactly.
    train_env = None
    if args.train:
        if n_envs == 1:
            train_vec = DummyVecEnv([_make_single_env])
        else:
            fns = [_make_env_fn(args.variant, env_kwargs, seed=seed + i) for i in range(n_envs)]
            try:
                train_vec = SubprocVecEnv(fns, start_method="spawn")
            except Exception as exc:
                print(f"  SubprocVecEnv failed ({exc}), falling back to DummyVecEnv")
                train_vec = DummyVecEnv(fns)

        train_env = VecNormalize(train_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    if args.train:  # train_env is guaranteed non-None here (created above)
        print(f"  Training for {args.total_timesteps} timesteps")

        wandb_callback = None
        if args.wandb:
            try:
                import wandb
                from wandb.integration.sb3 import WandbCallback

                run_name = f"ppo_{args.variant}"
                if args.variant in ("latent", "residual"):
                    run_name += f"_d{args.latent_dim}"
                if args.variant == "residual":
                    run_name += f"_w{args.residual_weight}"
                if args.ent_coef != 0.0:
                    run_name += f"_e{args.ent_coef}"

                wandb.init(
                    project=args.wandb_project,
                    group="ppo_baselines",
                    name=run_name,
                    config={
                        "variant": args.variant,
                        "decoder_type": args.decoder_type if args.variant != "direct" else None,
                        "obs_type": args.obs_type,
                        "total_timesteps": args.total_timesteps,
                        "latent_dim": args.latent_dim if args.variant != "direct" else 18,
                        "residual_weight": args.residual_weight if args.variant == "residual" else None,
                        "n_envs": n_envs,
                        "n_steps": N_STEPS_PER_ENV,
                        "batch_size": BATCH_SIZE,
                        "net_arch": str(NET_ARCH),
                        "ent_coef": args.ent_coef,
                        "lr": args.lr,
                        "effort_weight": args.effort_weight,
                    },
                    sync_tensorboard=True,
                )
                wandb_callback = WandbCallback(verbose=2)
                print(f"  W&B run: {wandb.run.url}")
            except Exception as e:
                print(f"  WARNING: W&B init failed ({e}), training continues without logging.")

        # Checkpoint every 500K env steps.
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cb = CheckpointCallback(
            save_freq=max(1, 500_000 // n_envs),
            save_path=str(ckpt_dir),
            name_prefix=f"ppo_{args.variant}",
        )
        # Saves VecNormalize stats alongside every checkpoint (.pkl matches .zip).
        # Critical: if the job is killed mid-run, intermediate checkpoints are
        # still loadable for eval without needing to retrain.
        vecnorm_ckpt_cb = _VecNormCheckpointCallback(train_env, checkpoint_cb)

        # EvalCallback syncs VecNormalize stats (train→eval) before each eval run,
        # then runs 10 deterministic episodes and logs eval/mean_reward,
        # eval/mean_ep_length. Saves best model to best_model/.
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(output_dir / "best_model"),
            log_path=str(output_dir / "eval_logs"),
            eval_freq=max(1, 25_000 // n_envs),
            n_eval_episodes=10,
            deterministic=True,
            render=False,
            verbose=1,
        )

        callbacks = [checkpoint_cb, vecnorm_ckpt_cb, eval_cb]
        if wandb_callback is not None:
            callbacks.append(wandb_callback)

        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=seed,
            n_steps=N_STEPS_PER_ENV,
            batch_size=BATCH_SIZE,
            n_epochs=10,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=NET_ARCH),
            tensorboard_log=str(output_dir / "tb_logs"),
            device="auto",
        )

        t0 = time.time()
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        elapsed = time.time() - t0
        print(f"  Training completed in {elapsed:.1f}s "
              f"({args.total_timesteps / elapsed:.0f} steps/s)")

        # Save final model + VecNormalize stats (needed for loading/eval later)
        final_path = output_dir / f"ppo_{args.variant}_final"
        model.save(str(final_path))
        train_env.save(str(final_path) + "_vecnorm.pkl")
        print(f"  Saved model to {final_path}.zip")
        print(f"  Saved VecNormalize stats to {final_path}_vecnorm.pkl")

        if args.wandb:
            try:
                wandb.finish()
            except Exception:
                pass

    if args.eval:
        model_path = output_dir / f"ppo_{args.variant}_final.zip"
        vecnorm_path = output_dir / f"ppo_{args.variant}_final_vecnorm.pkl"
        if not model_path.exists():
            ckpt_dir = output_dir / "checkpoints"
            if ckpt_dir.exists():
                ckpts = sorted(ckpt_dir.glob(f"ppo_{args.variant}_*.zip"))
                if ckpts:
                    model_path = ckpts[-1]
                    vecnorm_path = Path(str(ckpts[-1]).replace(".zip", "_vecnorm.pkl"))
                else:
                    print(f"  ERROR: No model found at {model_path}")
                    eval_env.close()
                    if train_env is not None:
                        train_env.close()
                    return
            else:
                print(f"  ERROR: No model found at {model_path}")
                eval_env.close()
                train_env.close()
                return

        # Load VecNormalize stats so eval obs are normalised identically to training
        if vecnorm_path.exists():
            eval_env = VecNormalize.load(str(vecnorm_path), eval_vec)
            eval_env.training = False
            eval_env.norm_reward = False
            print(f"  Loaded VecNormalize stats from {vecnorm_path}")
        else:
            print(f"  WARNING: VecNormalize stats not found at {vecnorm_path}, eval obs unnormalised")

        print(f"  Evaluating model from {model_path}")
        model = PPO.load(str(model_path), env=eval_env)

        n_eval_episodes = 10
        all_rewards = []
        all_lengths = []

        for ep in range(n_eval_episodes):
            obs = eval_env.reset()
            ep_reward = 0.0
            ep_len = 0
            done = [False]
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                ep_reward += float(reward[0])
                ep_len += 1
            all_rewards.append(ep_reward)
            all_lengths.append(ep_len)
            print(f"    Episode {ep + 1}: reward={ep_reward:.2f}, length={ep_len}")

        print(f"\n  Mean reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
        print(f"  Mean length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")

        eval_path = output_dir / f"eval_{args.variant}.npz"
        np.savez(eval_path, rewards=np.array(all_rewards), lengths=np.array(all_lengths))
        print(f"  Saved eval results to {eval_path}")

    eval_env.close()
    if train_env is not None:
        train_env.close()


# ---------------------------------------------------------------------------
# Record PPO policy video via LocoMuJoCo
# ---------------------------------------------------------------------------


def cmd_record_ppo(args):
    """Record a trained PPO policy as video using LocoMuJoCo's renderer."""
    from stable_baselines3 import PPO

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"

    print(f"=== Record PPO ({args.variant}) ===")

    # Find model
    model_path = output_dir / f"ppo_{args.variant}_final.zip"
    if not model_path.exists():
        ckpt_dir = output_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob(f"ppo_{args.variant}_*.zip"))
            if ckpts:
                model_path = ckpts[-1]
        if not model_path.exists():
            print(f"  ERROR: No model found at {model_path}")
            return
    print(f"  Model: {model_path}")

    # Create LocoMuJoCo env with recorder
    video_name = f"ppo_{args.variant}"
    env = make_env(
        record=True,
        video_path=str(video_dir),
        video_name=video_name,
        no_terminal=False,
        target_velocity=1.25,
    )

    # Load decoder for latent variant
    decoder_fn = None
    if args.variant == "latent":
        if args.decoder_checkpoint is None:
            raise ValueError("--decoder-checkpoint required for latent variant")
        from leaps.training.stride_trainer import build_stride_model

        ae_model = build_stride_model("StrideFlatAE", args.latent_dim)
        ae_model.load(args.decoder_checkpoint)
        ae_model._module.eval()
        decoder_fn = ae_model.decode

    # Load PPO — we need a dummy gym env just for PPO.load
    from leaps.envs.gym_wrapper import Gait10dof18MuscGymEnv, LatentActionGymEnv

    if args.variant == "direct":
        dummy_env = Gait10dof18MuscGymEnv(max_episode_steps=1000)
    else:
        dummy_env = LatentActionGymEnv(
            decoder_fn=decoder_fn, latent_dim=args.latent_dim, max_episode_steps=1000,
        )
    ppo_model = PPO.load(str(model_path), env=dummy_env)

    from leaps.envs.emg_mapping import EMGToMuscleMapper

    mapper = EMGToMuscleMapper()

    # Run episodes and record
    n_episodes = args.n_episodes
    for ep in range(n_episodes):
        obs = env.reset(key=jax.random.key(ep))
        env.render(record=True)
        ep_reward = 0.0

        for step in range(args.max_steps):
            # Get action from PPO
            obs_32 = np.asarray(obs, dtype=np.float32)
            action, _ = ppo_model.predict(obs_32, deterministic=True)

            # For latent variant: decode z → phase-indexed frame → muscles → LocoMuJoCo [-1,1]
            if args.variant == "latent" and decoder_fn is not None:
                z = np.asarray(action, dtype=np.float32)
                decoded = np.clip(decoder_fn(z.reshape(1, -1)), 0.0, 1.0)  # (1, 1111)
                stride = decoded.reshape(101, 11)
                # Phase-based extraction — matches LatentActionGymEnv._decode_action
                t = float(env._data.time)
                phase = (t % 1.0) / 1.0
                t_r = int(round(phase * 100)) % 101
                t_l = (t_r + 50) % 101
                muscles_r = mapper.map_emg_to_muscles(stride[t_r])[:9]
                muscles_l = mapper.map_emg_to_muscles(stride[t_l])[:9]
                muscles_01 = np.concatenate([muscles_r, muscles_l])
                loco_action = (2.0 * muscles_01 - 1.0).astype(np.float32)
            else:
                loco_action = np.asarray(action, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(loco_action)
            env.render(record=True)
            ep_reward += float(reward)

            if terminated or truncated:
                break

        print(f"  Episode {ep + 1}: reward={ep_reward:.2f}, steps={step + 1}")

    env.stop()
    print(f"  Video saved to {video_dir}/{video_name}.mp4")


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
    p_emg.add_argument("--data", default=LEAPS_H5_PATH, help="Path to emg_activations.h5")
    p_emg.add_argument("--output", default="results/emg_replay.npz", help="Output .npz path")
    p_emg.add_argument("--duration", type=float, default=10.0, help="Simulation duration (seconds)")
    p_emg.add_argument("--seed", type=int, default=42)
    p_emg.add_argument("--no-terminal", action="store_true", help="Disable early termination")
    p_emg.add_argument("--hang-mode", choices=["air", "place"], default=None,
                       help="air: suspend at 1.5m no ground (observe leg swing); "
                            "place: fix pelvis at 0.95m with ground (walking in place)")
    p_emg.add_argument("--record", action="store_true", help="Record MP4 video via LocoMuJoCo")

    # --- representation ---
    p_rep = subparsers.add_parser("representation", help="Reconstruct EMG through models, then simulate.")
    p_rep.add_argument("--data", default=LEAPS_H5_PATH, help="Path to emg_activations.h5")
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
    p_ppo.add_argument(
        "--variant", choices=["direct", "latent", "residual"], required=True,
        help="direct: 18-dim muscle actions. "
             "latent: latent_dim actions decoded to muscles. "
             "residual: (latent_dim+18) actions — decoded prior blended with residual correction.",
    )
    p_ppo.add_argument("--train", action="store_true", help="Train PPO")
    p_ppo.add_argument("--eval", action="store_true", help="Evaluate PPO")
    p_ppo.add_argument("--total-timesteps", type=int, default=1_000_000)
    p_ppo.add_argument("--output-dir", default="experiments/ppo")
    p_ppo.add_argument("--decoder-checkpoint", default=None,
                       help="AE checkpoint .pt (required for latent/residual variants)")
    p_ppo.add_argument("--decoder-type", choices=["snapshot", "stride"], default="snapshot",
                       help="'snapshot' for HausdorferAE (default), 'stride' for StrideFlatAE")
    p_ppo.add_argument("--obs-type", choices=["reduced", "full"], default="reduced",
                       help="'reduced' (19 dims, default) or 'full' (75 dims)")
    p_ppo.add_argument("--latent-dim", type=int, default=9)
    p_ppo.add_argument("--residual-weight", type=float, default=0.2,
                       help="Residual blend weight w: a = (1-w)*prior + w*residual. "
                            "Paper Table I: Humanoid=0.5, UnitreeA1=0.1. "
                            "Sweep: 0.2, 0.5. Only used for --variant residual.")
    p_ppo.add_argument("--stride-duration", type=float, default=1.0,
                       help="Gait cycle duration in seconds. Only used for stride decoder.")
    p_ppo.add_argument("--n-envs", type=int, default=4,
                       help="Parallel envs. Default 4 (paper setting, 4×2048=8192/update).")
    p_ppo.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p_ppo.add_argument("--effort-weight", type=float, default=0.01,
                       help="Effort penalty weight on mean(ctrl²). 0.0 = pure velocity reward (Hausdörfer).")
    p_ppo.add_argument("--upright-weight", type=float, default=0.0,
                       help="Upright posture reward weight. Scales linearly from 1.0 (upright) to 0.0 at tilt_limit.")
    p_ppo.add_argument("--ent-coef", type=float, default=0.0,
                       help="PPO entropy coefficient. SB3 default / paper default: 0.0.")
    p_ppo.add_argument("--lr", type=float, default=3e-4,
                       help="PPO learning rate. SB3 default: 3e-4.")
    p_ppo.add_argument("--batch-size", type=int, default=64,
                       help="PPO minibatch size. SB3 default / paper default: 64.")
    p_ppo.add_argument("--wandb", action="store_true")
    p_ppo.add_argument("--wandb-project", default="leaps")

    # --- record-ppo ---
    p_rec = subparsers.add_parser("record-ppo", help="Record trained PPO policy as video.")
    p_rec.add_argument("--variant", choices=["direct", "latent"], required=True)
    p_rec.add_argument("--output-dir", required=True, help="PPO experiment directory")
    p_rec.add_argument("--decoder-checkpoint", default=None, help="StrideFlatAE checkpoint (for latent)")
    p_rec.add_argument("--latent-dim", type=int, default=9)
    p_rec.add_argument("--n-episodes", type=int, default=1)
    p_rec.add_argument("--max-steps", type=int, default=1000)

    args = parser.parse_args(argv)

    if args.command == "emg-replay":
        cmd_emg_replay(args)
    elif args.command == "representation":
        cmd_representation(args)
    elif args.command == "ppo":
        if not args.train and not args.eval:
            parser.error("At least one of --train or --eval required for ppo command.")
        cmd_ppo(args)
    elif args.command == "record-ppo":
        cmd_record_ppo(args)


if __name__ == "__main__":
    main()
