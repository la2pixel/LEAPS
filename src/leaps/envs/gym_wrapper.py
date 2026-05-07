"""Gymnasium wrappers for the Gait10dof18Musc LocoMuJoCo environment.

Provides SB3-compatible Gymnasium envs:
  - Gait10dof18MuscGymEnv: direct 18-dim muscle action space
  - LatentActionGymEnv: latent_dim action space, frozen decoder maps z → muscles
  - ResidualLatentGymEnv: (latent_dim + 18) action space, blends decoded prior with residual
    a_final = (1 - residual_weight) * decode(z) + residual_weight * a_res
    (Hausdörfer et al. 2024 residual action prior pattern)

Two decoder backends are supported:
  - "snapshot"  HausdorferAE: z (latent_dim,) → (18,) muscles directly, no phase needed.
                Use with HausdorferAE checkpoints (.pt saved by HausdorferAE.save()).
  - "stride"    StrideFlatAE: z (latent_dim,) → (101,11) stride template, phase-indexed.
                Use with StrideFlatAE checkpoints.

All decoders are plain numpy — picklable for SubprocVecEnv, no PyTorch at step time.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from leaps.envs.emg_mapping import EMGToMuscleMapper
from leaps.envs.fast_gym_env import FastMuscleGymEnv


# ---------------------------------------------------------------------------
# Decoder loaders
# ---------------------------------------------------------------------------


def load_hausdorfer_decoder_weights(checkpoint_path: str) -> dict:
    """Load HausdorferAE decoder weights from a .pt checkpoint into numpy arrays.

    The checkpoint is saved by HausdorferAE.save() — a dict with key "state_dict"
    containing the nn.ModuleDict state. Decoder is a 2-layer MLP:
        Linear(latent→hidden) → Tanh → Linear(hidden→action_dim)

    Returns dict with keys: w1, b1, w2, b2, type="snapshot".
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    return {
        "w1": sd["decoder.0.weight"].numpy().astype(np.float32),  # (hidden, latent)
        "b1": sd["decoder.0.bias"].numpy().astype(np.float32),    # (hidden,)
        "w2": sd["decoder.2.weight"].numpy().astype(np.float32),  # (action_dim, hidden)
        "b2": sd["decoder.2.bias"].numpy().astype(np.float32),    # (action_dim,)
        "type": "snapshot",
    }


def load_flatae_decoder_weights(checkpoint_path: str) -> dict:
    """Load StrideFlatAE decoder weights from a .pt checkpoint into numpy arrays.

    The decoder is a 3-layer MLP: Linear(d→256)→Tanh→Linear(256→512)→Tanh→Linear(512→1111).

    Returns dict with keys: w1, b1, w2, b2, w3, b3, type="stride".
    """
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return {
        "w1": state["decoder.0.weight"].numpy().astype(np.float32),  # (256, d)
        "b1": state["decoder.0.bias"].numpy().astype(np.float32),    # (256,)
        "w2": state["decoder.2.weight"].numpy().astype(np.float32),  # (512, 256)
        "b2": state["decoder.2.bias"].numpy().astype(np.float32),    # (512,)
        "w3": state["decoder.4.weight"].numpy().astype(np.float32),  # (1111, 512)
        "b3": state["decoder.4.bias"].numpy().astype(np.float32),    # (1111,)
        "type": "stride",
    }


# ---------------------------------------------------------------------------
# Decoder backends — unified interface: decode_to_muscle_action(z, sim_time, stride_duration)
# Returns (18,) muscle action in [-1, 1], ready for FastMuscleGymEnv.step().
# ---------------------------------------------------------------------------


class _HausdorferDecoder:
    """Snapshot-level decoder: z (latent_dim,) → (18,) muscles in [-1,1].

    HausdorferAE architecture: Linear(latent→hidden)→Tanh→Linear(hidden→18).
    No phase indexing — outputs 18 bilateral muscle activations directly.
    sim_time and stride_duration are ignored.

    Args:
        weights: dict from load_hausdorfer_decoder_weights().
    """

    def __init__(self, weights: dict):
        self._w1 = weights["w1"]  # (hidden, latent)
        self._b1 = weights["b1"]  # (hidden,)
        self._w2 = weights["w2"]  # (action_dim, hidden)
        self._b2 = weights["b2"]  # (action_dim,)

    def decode_to_muscle_action(
        self, z: np.ndarray, sim_time: float = 0.0, stride_duration: float = 1.0
    ) -> np.ndarray:
        """z (latent_dim,) → (18,) muscle action in [-1, 1]."""
        z = z.ravel().astype(np.float32)
        h = np.tanh(self._w1 @ z + self._b1)
        out = np.clip(self._w2 @ h + self._b2, 0.0, 1.0)  # (18,) in [0,1]
        return (2.0 * out - 1.0).astype(np.float32)        # scale to [-1,1]


class _NumpyDecoder:
    """Stride-level decoder: z (latent_dim,) → (18,) muscles in [-1,1] via phase indexing.

    StrideFlatAE architecture: Linear(d→256)→Tanh→Linear(256→512)→Tanh→Linear(512→1111).
    Decodes a full (101,11) stride template, extracts the frame for the current gait phase,
    maps 11-channel EMG → 18 bilateral muscles, scales to [-1,1].

    Args:
        weights: dict from load_flatae_decoder_weights().
        mapper:  EMGToMuscleMapper instance.
    """

    def __init__(self, weights: dict, mapper: EMGToMuscleMapper):
        self._w1 = weights["w1"]
        self._b1 = weights["b1"]
        self._w2 = weights["w2"]
        self._b2 = weights["b2"]
        self._w3 = weights["w3"]
        self._b3 = weights["b3"]
        self._mapper = mapper

    def _decode_stride(self, z: np.ndarray) -> np.ndarray:
        """z (latent_dim,) → (101, 11) EMG stride template in [0,1]."""
        z = z.ravel().astype(np.float32)
        h1 = np.tanh(self._w1 @ z + self._b1)
        h2 = np.tanh(self._w2 @ h1 + self._b2)
        out = self._w3 @ h2 + self._b3
        return np.clip(out.reshape(101, 11), 0.0, 1.0)

    def decode_to_muscle_action(
        self, z: np.ndarray, sim_time: float = 0.0, stride_duration: float = 1.0
    ) -> np.ndarray:
        """z (latent_dim,) → (18,) muscle action in [-1, 1] via phase indexing."""
        stride = self._decode_stride(z)  # (101, 11)

        phase = (sim_time % stride_duration) / stride_duration
        t_idx_r = int(round(phase * 100)) % 101
        t_idx_l = (t_idx_r + 50) % 101

        muscles_r = self._mapper.map_emg_to_muscles(stride[t_idx_r])[:9]
        muscles_l = self._mapper.map_emg_to_muscles(stride[t_idx_l])[:9]
        muscles_01 = np.concatenate([muscles_r, muscles_l])  # (18,) in [0,1]
        return (2.0 * muscles_01 - 1.0).astype(np.float32)


def _build_decoder(weights: dict) -> "_HausdorferDecoder | _NumpyDecoder":
    """Build the right decoder from a weights dict (auto-detects type from 'type' key)."""
    dtype = weights.get("type", "stride")
    if dtype == "snapshot":
        return _HausdorferDecoder(weights)
    elif dtype == "stride":
        return _NumpyDecoder(weights, EMGToMuscleMapper())
    else:
        raise ValueError(f"Unknown decoder type: {dtype!r}. Expected 'snapshot' or 'stride'.")


# ---------------------------------------------------------------------------
# Environment factory (used by SubprocVecEnv workers)
# ---------------------------------------------------------------------------


def _make_fast_env(target_velocity: float = 1.25, obs_type: str = "reduced", **kwargs) -> FastMuscleGymEnv:
    """Create a FastMuscleGymEnv — pure MuJoCo, no JAX overhead."""
    for key in ("headless", "terminal_state_type", "goal_type", "reward_type", "reward_params"):
        kwargs.pop(key, None)
    return FastMuscleGymEnv(target_velocity=target_velocity, obs_type=obs_type, **kwargs)


# ---------------------------------------------------------------------------
# Direct 18-dim env
# ---------------------------------------------------------------------------


class Gait10dof18MuscGymEnv(gym.Env):
    """Gymnasium wrapper for Gait10dof18Musc with direct 18-dim muscle actions.

    Action space: Box(-1, 1, shape=(18,)) — LocoMuJoCo convention.
    Observation space: Box(-inf, inf, shape=(obs_dim,)).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        target_velocity: float = 1.25,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        tilt_limit: float = 0.8,
        **loco_kwargs,
    ):
        super().__init__()
        if render_mode == "human":
            loco_kwargs["headless"] = False
        else:
            loco_kwargs.setdefault("headless", True)

        self._env = _make_fast_env(target_velocity=target_velocity, render_mode=render_mode, tilt_limit=tilt_limit, **loco_kwargs)
        self._max_steps = max_episode_steps
        self._step_count = 0
        self.render_mode = render_mode

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, _ = self._env.reset(seed=seed)
        self._step_count = 0
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(action))
        self._step_count += 1
        if self._step_count >= self._max_steps:
            truncated = True
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        if self.render_mode is not None:
            return self._env.render()
        return None

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Latent action env (z → decode → muscles)
# ---------------------------------------------------------------------------


class LatentActionGymEnv(gym.Env):
    """Gymnasium wrapper with latent action space and frozen numpy decoder.

    Action space: Box(-1, 1, shape=(latent_dim,)).

    Supports two decoder backends (set via decoder_weights["type"]):
      "snapshot"  HausdorferAE: z → (18,) muscles directly, no phase needed.
      "stride"    StrideFlatAE: z → (101,11) stride template, phase-indexed.

    Args:
        decoder_weights: dict from load_hausdorfer_decoder_weights() or
            load_flatae_decoder_weights(). The "type" key selects the backend.
        latent_dim: Dimension of the latent action space.
        stride_duration: Gait cycle duration (seconds). Used by stride decoder only.
        obs_type: "reduced" (19 dims, default) or "full" (75 dims).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        decoder_weights: dict,
        latent_dim: int,
        target_velocity: float = 1.25,
        stride_duration: float = 1.0,
        obs_type: str = "reduced",
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        tilt_limit: float = 0.8,
        **loco_kwargs,
    ):
        super().__init__()
        if render_mode == "human":
            loco_kwargs["headless"] = False
        else:
            loco_kwargs.setdefault("headless", True)

        self._env = _make_fast_env(target_velocity=target_velocity, obs_type=obs_type, render_mode=render_mode, tilt_limit=tilt_limit, **loco_kwargs)
        self._decoder = _build_decoder(decoder_weights)
        self._latent_dim = latent_dim
        self._stride_duration = float(stride_duration)
        self._max_steps = max_episode_steps
        self._step_count = 0
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(latent_dim,), dtype=np.float32,
        )
        self.observation_space = self._env.observation_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, _ = self._env.reset(seed=seed)
        self._step_count = 0
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        muscle_action = self._decoder.decode_to_muscle_action(
            np.asarray(action, dtype=np.float32),
            sim_time=float(self._env.data.time),
            stride_duration=self._stride_duration,
        )
        obs, reward, terminated, truncated, info = self._env.step(muscle_action)
        self._step_count += 1
        if self._step_count >= self._max_steps:
            truncated = True
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        if self.render_mode is not None:
            return self._env.render()
        return None

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Residual latent env  (Hausdörfer et al. 2024 pattern)
# ---------------------------------------------------------------------------


class ResidualLatentGymEnv(gym.Env):
    """PPO latent prior with residual correction (Hausdörfer et al. 2024).

    Action space: Box(-1, 1, shape=(latent_dim + 18,)).
      - First `latent_dim` dims: z — decoded by frozen AE to muscle prior
      - Last 18 dims: a_res — additive residual correction

    Final muscle command:
        a_prior = decode(z)               # EMG-informed muscle template
        a_final = (1 - w) * a_prior + w * a_res

    The weight w = residual_weight. Paper (Table I) recommendation:
        w = 0.1 for stable quadrupeds (UnitreeA1), w = 0.5 for humanoids.
    We keep w FIXED (paper's decaying w caused policy collapse at ~2M steps).

    Supports both decoder backends ("snapshot" and "stride") via decoder_weights["type"].

    Args:
        decoder_weights: dict from load_hausdorfer_decoder_weights() or load_flatae_decoder_weights().
        latent_dim: Latent space dimension (must match checkpoint).
        residual_weight: Blend weight w ∈ [0, 1]. 0 = pure prior, 1 = pure residual.
        stride_duration: Gait cycle duration (seconds). Used by stride decoder only.
        obs_type: "reduced" (19 dims, default) or "full" (75 dims).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        decoder_weights: dict,
        latent_dim: int,
        residual_weight: float = 0.2,
        target_velocity: float = 1.25,
        stride_duration: float = 1.0,
        obs_type: str = "reduced",
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        tilt_limit: float = 0.8,
        **loco_kwargs,
    ):
        super().__init__()
        if render_mode == "human":
            loco_kwargs["headless"] = False
        else:
            loco_kwargs.setdefault("headless", True)

        self._env = _make_fast_env(target_velocity=target_velocity, obs_type=obs_type, render_mode=render_mode, tilt_limit=tilt_limit, **loco_kwargs)
        self._decoder = _build_decoder(decoder_weights)
        self._latent_dim = latent_dim
        self._residual_weight = float(residual_weight)
        self._stride_duration = float(stride_duration)
        self._max_steps = max_episode_steps
        self._step_count = 0
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(latent_dim + 18,), dtype=np.float32,
        )
        self.observation_space = self._env.observation_space

    def _get_action(self, action: np.ndarray) -> np.ndarray:
        z = action[: self._latent_dim]
        a_res = action[self._latent_dim :]
        a_prior = self._decoder.decode_to_muscle_action(
            z, sim_time=float(self._env.data.time), stride_duration=self._stride_duration,
        )
        a_final = (1.0 - self._residual_weight) * a_prior + self._residual_weight * a_res
        return np.clip(a_final, -1.0, 1.0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, _ = self._env.reset(seed=seed)
        self._step_count = 0
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        muscle_action = self._get_action(np.asarray(action, dtype=np.float32))
        obs, reward, terminated, truncated, info = self._env.step(muscle_action)
        self._step_count += 1
        if self._step_count >= self._max_steps:
            truncated = True
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        if self.render_mode is not None:
            return self._env.render()
        return None

    def close(self):
        self._env.close()
