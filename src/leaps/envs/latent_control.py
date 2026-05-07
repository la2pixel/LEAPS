"""Latent-action decoder control function for LocoMuJoCo PPOJax.

This module plugs the frozen FlatAE decoder into LocoMuJoCo's action pipeline.

How it works (the full flow):
-------------------------------
PPO policy
  → outputs  z ∈ R^d  (latent code, one per timestep, in [-1, 1])
  ↓
LatentDecoderControl.generate_action()
  → decodes   z  →  (101, 11)  EMG stride  (FlatAE decoder, frozen weights)
  → reads current simulation time, computes gait phase ∈ [0%, 100%]
  → extracts  emg_frame  =  emg_stride[phase_idx]         (11 channels)
  → maps      emg_frame  →  muscles_18   (EMGToMuscleMapper, bilateral)
  → converts  [0, 1]  →  [-1, 1]        (LocoMuJoCo action convention)
  ↓
MuJoCo env: data.ctrl ← muscles_18   (18 muscles, bilateral walking)

Why use a stride-level decoder per timestep?
--------------------------------------------
The FlatAE decoder maps a single latent code z to a full 101-frame gait
cycle. We use it as a "muscle template" that the policy can deform by
choosing different z values. At timestep t, we look up which frame in the
template corresponds to the current phase of the gait cycle (using
simulation time mod stride_duration) and output only that frame's 11
muscle values.

This gives the policy a smooth, biomechanically structured action space:
similar z values produce similar gait patterns, and the decoder's latent
space was trained on real human EMG, so valid z values produce movements
that resemble real human muscle coordination.

The FlatAE decoder weights are loaded from the PyTorch checkpoint and
stored as JAX arrays so the decode can run inside jax.jit (required by
PPOJax's VecEnv which vmaps everything).
"""

import numpy as np
import jax.numpy as jnp
import torch
from typing import Any, Dict, Tuple, Union
from types import ModuleType

from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.control_functions import ControlFunction


class LatentDecoderControl(ControlFunction):
    """Control function that decodes a latent z → 18 bilateral muscle commands.

    The policy outputs z ∈ R^d (d = latent_dim). This class:
        1. Decodes z using the frozen FlatAE decoder → (101, 11) EMG stride
        2. Extracts one frame based on current gait phase (from sim time)
        3. Maps 11 EMG channels → 18 bilateral muscle commands
        4. Converts muscle activations [0, 1] → LocoMuJoCo action [-1, 1]

    The decoder weights are stored as JAX arrays (converted from PyTorch
    at construction time), so the entire generate_action() is JAX-JIT-able.

    Args:
        env: The LocoMuJoCo environment instance.
        checkpoint_path (str): Path to the StrideFlatAE .pt checkpoint file.
        latent_dim (int): Dimension of the latent space (must match checkpoint).
        stride_duration (float): Duration of one gait cycle in seconds.
            At 1.25 m/s level-ground walking, a typical stride is ~1.0 s.
            This sets how fast the template cycles through: every
            stride_duration seconds, the phase wraps from 0% back to 0%.
    """

    def __init__(
        self,
        env: Any,
        checkpoint_path: str,
        latent_dim: int,
        stride_duration: float = 1.0,
        **kwargs: Dict,
    ):
        # ── Load FlatAE decoder weights from PyTorch checkpoint ──────────
        # The checkpoint was saved by StrideFlatAE.save() which calls
        # torch.save(self._module.state_dict(), path). The state_dict has:
        #   decoder.0.weight  (256, d)   — first linear layer weights
        #   decoder.0.bias    (256,)     — first linear layer biases
        #   decoder.2.weight  (512, 256) — second linear layer weights
        #   decoder.2.bias    (512,)     — second linear layer biases
        #   decoder.4.weight  (1111, 512)— third linear layer weights
        #   decoder.4.bias    (1111,)    — third linear layer biases
        # (Indices 0, 2, 4 because nn.Sequential wraps Tanh at 1, 3.)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Convert to JAX arrays. These become compile-time constants inside
        # jax.jit — JAX traces through them as static values.
        self._w1 = jnp.array(state["decoder.0.weight"].numpy())  # (256, d)
        self._b1 = jnp.array(state["decoder.0.bias"].numpy())    # (256,)
        self._w2 = jnp.array(state["decoder.2.weight"].numpy())  # (512, 256)
        self._b2 = jnp.array(state["decoder.2.bias"].numpy())    # (512,)
        self._w3 = jnp.array(state["decoder.4.weight"].numpy())  # (1111, 512)
        self._b3 = jnp.array(state["decoder.4.bias"].numpy())    # (1111,)

        # ── EMG → muscle mapping constants (precomputed, JAX-friendly) ───
        # Default activation per leg (9-dim) for muscles with no EMG channel.
        # Only iliopsoas (index 3) keeps this default; all others are overwritten.
        # 0.05 avoids permanent hip-flexion torque that overwhelms glut_max at rest.
        self._default_act_leg = jnp.full(9, 0.05)

        # Stride configuration
        self._stride_duration = float(stride_duration)

        # ── Set policy action space to [-1, 1]^latent_dim ───────────────
        # PPO samples z from this box. The decoder maps any z ∈ [-1, 1]^d
        # to a biomechanically plausible muscle pattern (learned from EMG).
        low = -np.ones(latent_dim, dtype=np.float32)
        high = np.ones(latent_dim, dtype=np.float32)
        super().__init__(env, low, high, **kwargs)

    # ── Decoder (JAX-JIT-able) ────────────────────────────────────────────

    def _decode_emg_stride(self, z: jnp.ndarray) -> jnp.ndarray:
        """FlatAE decoder: z ∈ R^d → emg_stride ∈ [0,1]^(101×11).

        The decoder is a 3-layer MLP:
            z → Linear(d→256) → Tanh → Linear(256→512) → Tanh
              → Linear(512→1111) → reshape(101, 11) → clip[0,1]

        The linear output (no sigmoid) is clipped to [0,1] because the
        model was trained against [0,1] EMG targets so the output naturally
        stays near that range, but without the flat-gradient problem of
        sigmoid at the extremes.

        Math:
            h1 = Tanh(W1 @ z + b1)          # (256,)
            h2 = Tanh(W2 @ h1 + b2)         # (512,)
            out = W3 @ h2 + b3              # (1111,) = 101 × 11
            emg_stride = clip(out.reshape(101, 11), 0, 1)
        """
        h1 = jnp.dot(self._w1, z) + self._b1     # (256,): W1 is (256,d), z is (d,)
        h1 = jnp.tanh(h1)                        # Tanh — matches StrideFlatAE decoder
        h2 = jnp.dot(self._w2, h1) + self._b2   # (512,)
        h2 = jnp.tanh(h2)                        # Tanh
        out = jnp.dot(self._w3, h2) + self._b3  # (1111,)
        emg_stride = out.reshape(101, 11)        # (101, 11) — one frame per % gait cycle
        return jnp.clip(emg_stride, 0.0, 1.0)

    def _map_emg_to_9muscles(self, emg: jnp.ndarray) -> jnp.ndarray:
        """Map 11-channel EMG frame → 9 muscle commands for one leg.

        Mapping rules:
            gastrocmed  [0]  → gastroc   [6]            (1:1 gastrocnemius)
            tibialis    [1]  → tib_ant   [8]            (1:1 tibialis anterior)
            soleus      [2]  → soleus    [7]            (1:1 soleus)
            vastusmed   [3] }→ vasti     [5]            (average of two quad heads)
            vastuslat   [4] }
            rectfem     [5]  → rect_fem  [4]            (1:1 biarticular quad)
            bicepsfem   [6] }→ hamstrings [0]           (average of biarticular hamstrings)
            semitend    [7] }
            bicepsfem   [6]  → bifemsh   [1] × 0.5     (monoarticular knee flexor only)
            gluteusmed  [9]  → glut_max  [2] × 0.3     (partial proxy — see note)
            gracilis    [8]  → DROPPED (no match in 18-muscle model)
            ext.oblique [10] → DROPPED (no match in 18-muscle model)
            iliopsoas        → default 0.20             (no EMG channel available)

        Note on glut_max scaling (0.3):
            gluteusmedius is a hip ABDUCTOR (frontal plane stabiliser).
            glut_max is a hip EXTENSOR (sagittal plane). They are different
            muscles. Using the abductor signal at full strength would drive
            excessive hip extension and tip the torso backward every step.
            0.3 is a partial proxy: gluteal muscles co-activate during stance,
            so some signal is informative, but at a heavily reduced weight.

        Returns activations in [0, 1] for 9 muscles (one leg).
        """
        # Start from default (0.20 for iliopsoas, which has no EMG channel).
        # index 3 (iliopsoas) keeps this default; all others are overwritten.
        act = self._default_act_leg  # (9,)

        act = act.at[0].set((emg[6] + emg[7]) / 2.0)  # hamstrings ← avg(bicepsfem, semitend)
        act = act.at[1].set(emg[6] * 0.5)              # bifemsh ← bicepsfem × 0.5
        act = act.at[2].set(jnp.clip(0.25 + emg[9] * 0.15, 0.0, 1.0))  # glut_max: 0.25 floor + gluteusmed signal
        # act[3] stays at 0.20 → iliopsoas (no EMG channel)
        act = act.at[4].set(emg[5])                     # rect_fem ← rectusfemoris
        act = act.at[5].set((emg[3] + emg[4]) / 2.0)  # vasti ← avg(vastusmed, vastuslat)
        act = act.at[6].set(emg[0])                     # gastroc ← gastrocmed
        act = act.at[7].set(emg[2])                     # soleus
        act = act.at[8].set(emg[1])                     # tib_ant ← tibialisanterior

        return jnp.clip(act, 0.0, 1.0)

    # ── Main interface called by LocoMuJoCo at each env step ─────────────

    def generate_action(
        self,
        env: Any,
        action: Union[np.ndarray, jnp.ndarray],
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """Decode latent z → 18 muscle ctrl values at the current gait phase.

        Called by LocoMuJoCo at every env step. The return value is written
        directly to data.ctrl[action_indices] by the base class — so this
        must return MuJoCo ctrl values in ctrlrange [0, 1], NOT normalized
        [-1, 1]. (DefaultControl is a replacement for this step, not a wrapper
        applied on top of it.)

        Args:
            action: z ∈ R^d from PPO policy, in [-1, 1] (latent code).
            data: MuJoCo/MJX data containing data.time (current sim time).
            backend: jnp when MJX/JAX, np when CPU — both work here.

        Returns:
            (ctrl_18, carry): 18-dim muscle ctrl values in [0, 0.5] (MuJoCo
            ctrlrange [0, 1]), plus unchanged carry.

        Math and indexing:
            phase_r = (data.time mod stride_duration) / stride_duration ∈ [0, 1)
            t_idx_r = round(phase_r × 100) ∈ {0, ..., 100}   — right leg
            t_idx_l = (t_idx_r + 50) mod 101                  — left leg (50% offset)
            emg_r = emg_stride[t_idx_r]   (11-ch right-leg frame)
            emg_l = emg_stride[t_idx_l]   (11-ch left-leg frame, ~50% ahead)
            ctrl_r = emg_mapper(emg_r)    muscles 0–8, in [0, 1]
            ctrl_l = emg_mapper(emg_l)    muscles 9–17, in [0, 1]
            ctrl_18 = clip([ctrl_r, ctrl_l], 0, 0.5)

        Why separate phase for each leg:
            Human gait is contralateral — right and left legs alternate with
            ~50% phase offset. The EMG stride template contains right-leg
            activations only. At right heel strike (t_idx=0), the left leg
            is mid-swing/push-off, corresponding to frame 50 of the template.
            Without this offset both legs activate simultaneously (bunny hop).
        """
        # Step 1: decode latent z → full (101, 11) EMG stride template.
        emg_stride = self._decode_emg_stride(action)  # (101, 11), in [0, 1]

        # Step 2: right-leg gait phase from simulation time.
        phase = backend.mod(data.time / self._stride_duration, 1.0)  # ∈ [0, 1)

        # Step 3: frame indices for right and left legs.
        # Right leg: current phase.
        # Left leg: 50% offset (contralateral alternation).
        t_idx_r = jnp.int32(jnp.round(phase * 100.0))
        t_idx_r = jnp.minimum(t_idx_r, 100)           # clamp: round(100.0) = 100 ✓
        t_idx_l = jnp.mod(t_idx_r + 50, 101)          # ∈ {0, ..., 100}

        # Step 4: extract per-leg EMG frames.
        emg_frame_r = emg_stride[t_idx_r]  # (11,) right leg
        emg_frame_l = emg_stride[t_idx_l]  # (11,) left leg (phase-shifted)

        # Step 5: map each leg's EMG → 9 muscle ctrl values independently.
        ctrl_r = self._map_emg_to_9muscles(emg_frame_r)  # (9,) muscles 0–8
        ctrl_l = self._map_emg_to_9muscles(emg_frame_l)  # (9,) muscles 9–17
        ctrl_18 = jnp.concatenate([ctrl_r, ctrl_l])      # (18,)

        ctrl_18 = jnp.clip(ctrl_18, 0.0, 1.0)

        return ctrl_18, carry
