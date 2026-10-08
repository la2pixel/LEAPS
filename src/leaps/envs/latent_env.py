"""Latent-action-prior wrapper for SCONE/Hyfydy gym envs.

The policy outputs a latent z (dim k) plus a full-size residual a_full. A
frozen EMG decoder (priors/decoder_k*_AB06_corrected) maps z to 11 EMG
channels, EMGToMuscleMapper maps those onto the model's muscles, and the two
are blended per actuator:

    a_hat = EMGToMuscleMapper(decoder(z))        # 0 where no EMG channel maps
    final_action = (1 - w) * a_hat + w * a_full

w is mapped_residual_weight on actuators that have an EMG channel and
residual_weight (1.0) elsewhere, so unmapped muscles are fully free. Both
terms are in [0, 1]. The decoder is never updated by RL.

Controls: null_prior (a_hat = 0, w forced to 1), untrained_decoder (same
architecture, random weights), dep_content (a_hat from DEP's correlation
matrix, see dep_content_prior.py).
"""

from typing import Optional

import gym
import numpy as np
import torch
import torch.nn as nn

from leaps.envs.dep_content_prior import DepContentPrior
from leaps.envs.emg_mapping import EMG_CHANNELS, EMGToMuscleMapper


def build_decoder(dim_latent: int, dim_a: int = 11, hidden: int = None, cond_dim: int = 0) -> nn.Sequential:
    """cond_dim widens the input layer for a conditioned decoder (speed_cond)."""
    h = hidden or 2 * dim_latent
    return nn.Sequential(
        nn.Linear(dim_latent + cond_dim, h), nn.Tanh(),
        nn.Linear(h, dim_a), nn.Sigmoid(),
    )


class _LengthReflex:
    """Positive fiber-length feedback added to a_hat (state_feedback_gain)."""

    def __init__(self, n_actuators: int, gain: float):
        self.gain = gain
        self._max_len = np.zeros(n_actuators)
        self._min_len = np.ones(n_actuators) * 100.0

    def step(self, fiber_lengths: np.ndarray) -> np.ndarray:
        self._max_len = np.maximum(fiber_lengths, self._max_len)
        self._min_len = np.minimum(fiber_lengths, self._min_len)
        length_u = (fiber_lengths - self._min_len) / (self._max_len - self._min_len + 0.1)
        return self.gain * length_u


class LatentActionPriorWrapper(gym.ActionWrapper):
    """The declared [0, 1] action Box is not enforced: the actor's tanh output
    reaches action() unscaled, so z_u = action[:dim_latent] is in [-1, 1] and
    is rescaled to the decoder's training range [z_lo, z_hi]."""

    def __init__(
        self, env, model_name: str, decoder_path: Optional[str] = None, norm_path: Optional[str] = None,
        dim_latent: int = 6, residual_weight: float = 0.5, mirror_left: bool = False,
        null_prior: bool = False, mapped_residual_weight: Optional[float] = None,
        untrained_decoder: bool = False, speed_cond: Optional[float] = None,
        dep_content: bool = False,
        loosened_actuators: Optional[list[str]] = None, loosened_residual_weight: Optional[float] = None,
        state_feedback_gain: float = 0.0,
    ):
        if dep_content and speed_cond is not None:
            raise ValueError("dep_content and speed_cond are mutually exclusive -- there is no decoder to condition")
        self.dep_content = dep_content
        super().__init__(env)
        actuator_names = [a.name() for a in env.unwrapped.model.actuators()]

        self.mapper = EMGToMuscleMapper(
            model_name, actuator_names,
            mirror_left=mirror_left,
            null_prior=null_prior,
        )
        self.n_actuators = len(actuator_names)
        self.dim_latent = dim_latent
        self.state_feedback_gain = state_feedback_gain
        self._reflex = _LengthReflex(self.n_actuators, state_feedback_gain) if state_feedback_gain > 0 else None

        # mapped mask from a null_prior=False mapper: with null_prior the
        # wrapper's own mapper has all-zero weights
        _reference_mapper = EMGToMuscleMapper(
            model_name, actuator_names, mirror_left=mirror_left, null_prior=False,
        )
        self._mapped_mask = _reference_mapper.weights.sum(axis=1) > 0
        self._right_mask = np.array([n.endswith("_r") for n in actuator_names])
        self._left_mask = np.array([n.endswith("_l") for n in actuator_names])
        self._mapped_names = [n for n, m in zip(actuator_names, self._mapped_mask) if m]

        # mapped_residual_weight applies only to actuators with an EMG prior.
        # Not with null_prior: a_hat = 0 there, so a low w would only cap the
        # reachable activation at w.
        if mapped_residual_weight is not None and not null_prior:
            w = np.full(self.n_actuators, residual_weight, dtype=np.float32)
            w[self._mapped_mask] = mapped_residual_weight
            self.residual_weight = w
        else:
            self.residual_weight = residual_weight

        # per-actuator override of mapped_residual_weight
        if loosened_actuators:
            if mapped_residual_weight is None or null_prior:
                raise ValueError(
                    "loosened_actuators requires mapped_residual_weight (and null_prior=False) "
                    "-- there is no per-actuator residual_weight array to override otherwise"
                )
            if loosened_residual_weight is None:
                raise ValueError("loosened_actuators given without loosened_residual_weight")
            missing = [n for n in loosened_actuators if n not in actuator_names]
            if missing:
                raise ValueError(f"loosened_actuators not found on this body: {missing}")
            idx = [actuator_names.index(n) for n in loosened_actuators]
            self.residual_weight[idx] = loosened_residual_weight

        # speed_cond: a fixed target speed in m/s, or "env" to read the env's
        # target_vel at every reset. Needs a speed-conditioned decoder.
        self.speed_cond = speed_cond
        self._speed_cond_from_env = speed_cond == "env"
        if self.dep_content:
            self.dep_content_prior = DepContentPrior(self._mapped_mask, self.n_actuators)
            self._speed_cond_tensor = None
        else:
            cond_dim = 1 if speed_cond is not None else 0
            self.decoder = build_decoder(dim_latent, cond_dim=cond_dim)
            # untrained_decoder keeps the random init
            if not untrained_decoder:
                self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
            # deprl sets the default device to cuda; the decoder runs on cpu
            self.decoder.to("cpu")
            self.decoder.eval()
            for p in self.decoder.parameters():
                p.requires_grad = False

            norm = np.load(norm_path)
            self.p01, self.p99 = norm["p01"], norm["p99"]
            self.z_lo, self.z_hi = norm["z_lo"], norm["z_hi"]

            if speed_cond is not None:
                self._speed_lo, self._speed_hi = float(norm["speed_lo"]), float(norm["speed_hi"])
                init_speed = (
                    float(self.env.unwrapped.target_vel) if self._speed_cond_from_env else float(speed_cond)
                )
                self._speed_cond_tensor = self._speed_tensor(init_speed)
            else:
                self._speed_cond_tensor = None

        # policy outputs: dim_latent (z) + n_actuators (a_full)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(dim_latent + self.n_actuators,), dtype=np.float32
        )

    def _speed_tensor(self, speed: float) -> torch.Tensor:
        speed_u = (speed - self._speed_lo) / (self._speed_hi - self._speed_lo)
        return torch.tensor([[speed_u]], dtype=torch.float32, device="cpu")

    def _from_unit(self, x_u: np.ndarray) -> np.ndarray:
        return (x_u * (self.p99 - self.p01 + 1e-8) + self.p01).astype(np.float32)

    def _to_z_domain(self, z_u: np.ndarray) -> np.ndarray:
        """[-1, 1] -> [z_lo, z_hi] per dimension."""
        return (z_u * (self.z_hi - self.z_lo) / 2 + (self.z_hi + self.z_lo) / 2).astype(np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        z_u = action[: self.dim_latent].astype(np.float32)
        a_full = action[self.dim_latent:]

        if self.dep_content:
            emg = np.zeros(len(EMG_CHANNELS), dtype=np.float32)
            fiber_lengths = self.env.unwrapped.model.muscle_fiber_length_array()
            a_hat = self.dep_content_prior.step(fiber_lengths)
        else:
            z = self._to_z_domain(z_u)
            with torch.no_grad():
                z_t = torch.from_numpy(z)[None]
                if self._speed_cond_tensor is not None:
                    z_t = torch.cat([z_t, self._speed_cond_tensor], dim=-1)
                emg_u = self.decoder(z_t)[0].numpy()
            emg = self._from_unit(emg_u)
            # a_hat from the sigmoid output, not from emg in raw units: the
            # per-channel P99 scale differs up to 13x and would leave [0, 1]
            a_hat = self.mapper.map_emg_to_muscles(emg_u)

        if self._reflex is not None:
            fiber_lengths = self.env.unwrapped.model.muscle_fiber_length_array()
            fb = self._reflex.step(fiber_lengths)
            a_hat = a_hat.copy()
            a_hat[self._mapped_mask] = np.clip(a_hat[self._mapped_mask] + fb[self._mapped_mask], 0.0, 1.0)

        # convex blend: a_full can pull a mapped muscle below a_hat as well as above it
        final_action = (1 - self.residual_weight) * a_hat + self.residual_weight * a_full

        # kept for _diagnostics_post_clip(), called by SconeWrapper after its clip
        self._last_a_hat = a_hat
        self._last_final_action = final_action
        self._last_emg = emg

        return final_action

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if self._speed_cond_from_env:
            self._speed_cond_tensor = self._speed_tensor(float(self.env.unwrapped.target_vel))
        return out

    def _diagnostics_post_clip(self, clip_lo: float, clip_hi: float) -> dict:
        """Prior/residual diagnostics on the values applied to the muscles,
        i.e. after SconeWrapper's clip to [clip_lo, clip_hi].
        residual = final_action - a_hat, signed."""
        eps = 1e-8
        a_hat = np.clip(self._last_a_hat, clip_lo, clip_hi)
        final_action = np.clip(self._last_final_action, clip_lo, clip_hi)
        residual = final_action - a_hat

        mapped_a_hat = a_hat[self._mapped_mask]
        mapped_residual = residual[self._mapped_mask]
        diagnostics = {
            "train/latent_prior_mean_mapped": float(mapped_a_hat.mean())
            if self._mapped_mask.any() else 0.0,
            "train/latent_residual_mean_unmapped": float(residual[~self._mapped_mask].mean())
            if (~self._mapped_mask).any() else 0.0,
            "train/latent_residual_share": float(
                np.abs(residual).sum() / (np.abs(residual).sum() + np.abs(a_hat).sum() + eps)
            ),
            "train/latent_left_leg_mean_action": float(final_action[self._left_mask].mean())
            if self._left_mask.any() else 0.0,
            "train/latent_right_leg_mean_action": float(final_action[self._right_mask].mean())
            if self._right_mask.any() else 0.0,
            "train/latent_left_leg_residual_mean": float(residual[self._left_mask].mean())
            if self._left_mask.any() else 0.0,
            # size of the correction relative to the prior, mapped muscles only
            "train/latent_residual_absmean_mapped": float(np.abs(mapped_residual).mean())
            if self._mapped_mask.any() else 0.0,
            "train/latent_prior_absmean_mapped": float(np.abs(mapped_a_hat).mean())
            if self._mapped_mask.any() else 0.0,
            "train/latent_residual_to_prior_ratio_mapped": float(
                np.abs(mapped_residual).mean() / (np.abs(mapped_a_hat).mean() + eps)
            ) if self._mapped_mask.any() else 0.0,
        }
        for name, r in zip(self._mapped_names, mapped_residual):
            diagnostics[f"train/latent_residual_abs/{name}"] = float(abs(r))
        for name, a in zip(self._mapped_names, mapped_a_hat):
            diagnostics[f"train/latent_prior_abs/{name}"] = float(abs(a))
        for name, e in zip(EMG_CHANNELS, self._last_emg):
            diagnostics[f"train/decoded_emg/{name}"] = float(e)

        return diagnostics
