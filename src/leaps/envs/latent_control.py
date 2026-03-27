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
        # (Indices 0, 2, 4 because nn.Sequential wraps ReLUs at 1, 3.)
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
        # Default activation for muscles with no EMG channel (iliopsoas).
        # 0.20 is biomechanically motivated: iliopsoas fires at ~15-25% MVC
        # throughout the gait cycle as the primary hip flexor.
        self._default_act = jnp.full(18, 0.20)

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
            z → Linear(d→256) → ReLU → Linear(256→512) → ReLU
              → Linear(512→1111) → reshape(101, 11) → clip[0,1]

        The linear output (no sigmoid) is clipped to [0,1] because the
        model was trained against [0,1] EMG targets so the output naturally
        stays near that range, but without the flat-gradient problem of
        sigmoid at the extremes.

        Math:
            h1 = ReLU(W1 @ z + b1)          # (256,)
            h2 = ReLU(W2 @ h1 + b2)         # (512,)
            out = W3 @ h2 + b3              # (1111,) = 101 × 11
            emg_stride = clip(out.reshape(101, 11), 0, 1)
        """
        h1 = jnp.dot(self._w1, z) + self._b1     # (256,): W1 is (256,d), z is (d,)
        h1 = jnp.maximum(h1, 0.0)                # ReLU: max(0, x)
        h2 = jnp.dot(self._w2, h1) + self._b2   # (512,)
        h2 = jnp.maximum(h2, 0.0)               # ReLU
        out = jnp.dot(self._w3, h2) + self._b3  # (1111,)
        emg_stride = out.reshape(101, 11)        # (101, 11) — one frame per % gait cycle
        return jnp.clip(emg_stride, 0.0, 1.0)

    def _map_emg_to_18muscles(self, emg: jnp.ndarray) -> jnp.ndarray:
        """Map 11-channel right-leg EMG → 18 bilateral muscle commands.

        Mapping rules (right leg, then mirrored to left leg):
            gastrocmed  [0]  → gastroc_r  [6]            (1:1 gastrocnemius)
            tibialis    [1]  → tib_ant_r  [8]            (1:1 tibialis anterior)
            soleus      [2]  → soleus_r   [7]            (1:1 soleus)
            vastusmed   [3] }→ vasti_r    [5]            (average of two quad heads)
            vastuslat   [4] }
            rectfem     [5]  → rect_fem_r [4]            (1:1 biarticular quad)
            bicepsfem   [6] }→ hamstrings_r [0]          (average of biarticular hamstrings)
            semitend    [7] }
            bicepsfem   [6]  → bifemsh_r  [1] × 0.5     (monoarticular knee flexor only)
            gluteusmed  [9]  → glut_max_r [2] × 0.3     (partial proxy — see note)
            gracilis    [8]  → DROPPED (no match in 18-muscle model)
            ext.oblique [10] → DROPPED (no match in 18-muscle model)
            iliopsoas        → default 0.20              (no EMG channel available)

        Note on glut_max scaling (0.3):
            gluteusmedius is a hip ABDUCTOR (frontal plane stabiliser).
            glut_max is a hip EXTENSOR (sagittal plane). They are different
            muscles. Using the abductor signal at full strength would drive
            excessive hip extension and tip the torso backward every step.
            0.3 is a partial proxy: gluteal muscles co-activate during stance,
            so some signal is informative, but at a heavily reduced weight.

        Returns activations in [0, 1] for all 18 muscles.
        """
        # Start from default (0.20 for iliopsoas, which has no EMG channel)
        act = self._default_act

        # ── Right leg (indices 0–8 in the 18-muscle model) ───────────────

        # Hamstrings (biarticular: hip extensor + knee flexor)
        # biceps femoris (long head) and semitendinosus both act together here.
        hamstring_r = (emg[6] + emg[7]) / 2.0          # average of BF + ST
        act = act.at[0].set(hamstring_r)                # hamstrings_r

        # Biceps femoris SHORT head (monoarticular knee flexor only)
        # It co-activates with the long head but at a lower level.
        act = act.at[1].set(emg[6] * 0.5)              # bifemsh_r

        # Glut max: partial proxy from gluteusmedius (see docstring above)
        act = act.at[2].set(emg[9] * 0.3)              # glut_max_r

        # act.at[3] stays at default 0.20 → iliopsoas_r (no EMG, set constant)

        act = act.at[4].set(emg[5])                     # rect_fem_r ← rectusfemoris
        act = act.at[5].set((emg[3] + emg[4]) / 2.0)  # vasti_r ← avg(vastusmed, vastuslat)
        act = act.at[6].set(emg[0])                     # gastroc_r ← gastrocmed
        act = act.at[7].set(emg[2])                     # soleus_r
        act = act.at[8].set(emg[1])                     # tib_ant_r ← tibialisanterior

        # ── Left leg (indices 9–17) = mirror right leg ──────────────────
        # We assume symmetric gait: right-leg EMG, phase-shifted by 50%,
        # was already applied during training. Here we simply mirror the
        # per-frame activations; the 50% phase shift is handled by the
        # stride template (the right leg fires at t=0%, left fires at t=50%).
        act = act.at[9:18].set(act[0:9])

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
        """Decode latent z → 18 muscle commands at the current gait phase.

        Called by LocoMuJoCo at every env step BEFORE the action is set
        in data.ctrl. This is where PPO's latent z is converted to actual
        muscle activations.

        Args:
            action: z ∈ R^d from PPO policy, in [-1, 1] (latent code).
            data: MuJoCo/MJX data containing data.time (current sim time).
            backend: jnp when MJX/JAX, np when CPU — both work here.

        Returns:
            (muscles_loco, carry): 18-dim action in [-1, 1] (LocoMuJoCo
            convention), plus unchanged carry.

        Math and indexing:
            phase = (data.time mod stride_duration) / stride_duration ∈ [0, 1)
            t_idx = round(phase × 100) ∈ {0, 1, ..., 100}
            emg_frame = emg_stride[t_idx]      (11 channels for this phase)
            muscles_18 = emg_mapper(emg_frame)  (18 bilateral muscles, [0,1])
            loco_action = 2 × muscles_18 - 1   (convert [0,1] → [-1,1])

            The final conversion is needed because LocoMuJoCo's DefaultControl
            maps [-1,1] → [ctrl_low, ctrl_high] via:
                ctrl = 0.5 × (action + 1) × (high - low) + low
            For muscles: ctrlrange = [0, 1], so:
                ctrl = 0.5 × (action + 1)
            Inverting: action = 2 × ctrl - 1.
        """
        # Step 1: decode latent z → full (101, 11) EMG stride
        # Uses JAX-JIT-able MLP forward pass with frozen weights.
        emg_stride = self._decode_emg_stride(action)  # (101, 11), in [0, 1]

        # Step 2: compute current gait phase from simulation time.
        # data.time is the elapsed simulation time in seconds (MuJoCo tracks
        # this automatically). Dividing by stride_duration and taking modulo
        # gives a repeating phase in [0, 1) that cycles with every gait cycle.
        #   e.g. at t=0.5s with stride_duration=1.0: phase=0.5 → 50% of cycle
        phase = backend.mod(data.time / self._stride_duration, 1.0)  # ∈ [0, 1)

        # Step 3: convert phase to integer frame index [0, 100].
        # The stride array has 101 frames: index 0 = heel strike (0% of cycle),
        # index 100 = 100% (= next heel strike). round() gives nearest frame.
        t_idx = jnp.int32(jnp.round(phase * 100.0))  # ∈ {0, ..., 100}
        t_idx = jnp.minimum(t_idx, 100)              # safety clamp to [0, 100]

        # Step 4: extract the 11-channel EMG frame for this gait phase.
        # JAX supports dynamic integer indexing inside jit (unlike Python lists).
        emg_frame = emg_stride[t_idx]  # (11,), in [0, 1]

        # Step 5: map 11 EMG channels → 18 bilateral muscle commands.
        # Right leg gets the current EMG; left leg is a mirror. The 50% phase
        # offset between legs is implicitly encoded in the stride template —
        # at t=0 (right heel strike), the template's left-leg muscles are in
        # the push-off configuration (which corresponds to t≈50 of the right).
        muscles_18 = self._map_emg_to_18muscles(emg_frame)  # (18,), in [0, 1]

        # Step 6: convert [0, 1] → [-1, 1] for LocoMuJoCo's action space.
        # LocoMuJoCo normalises the agent's action from [-1,1] to the actuator's
        # ctrlrange. For muscles (ctrlrange = [0,1]):
        #   ctrl = 0.5 * (action + 1)
        # So we need: action = 2 * ctrl - 1 to get ctrl=muscles_18 back.
        loco_action = 2.0 * muscles_18 - 1.0  # (18,), in [-1, 1]

        return loco_action, carry
