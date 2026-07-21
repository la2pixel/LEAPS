"""Latent-action-prior wrapper for SCONE/Hyfydy gym envs.

Wraps an existing env (e.g. gym.make("sconewalk_h0918-v1")) so the policy
controls a low-dimensional latent z plus a full-size residual, instead of
the muscle-space action directly -- matching the reference paper's method:

    a_hat = EMGToMuscleMapper(decoder(z))   -- 0 for actuators with no substitute
    final_action = (1 - residual_weight) * a_hat + residual_weight * a_full

z and a_full both come from the policy each step. Every actuator gets both
contributions (not a partition) -- for actuators EMGToMuscleMapper has no
substitute for (including the entire left leg by default, since the mapper
only populates "_r" muscles unless mirror_left=True), a_hat is 0 there, so
final_action reduces to residual_weight * a_full, exactly as before this fix.

The decoder is the one trained in notebooks/get_synergy.ipynb -- frozen
here (requires_grad=False, eval mode), never updated by RL.

HERE! This used to be a plain sum, `a_hat + residual_weight * a_full`.
a_full is in [0, 1] (policy's action_space is a [0,1] Box) and never allowed
to go negative -- so under the old formula final_action was provably >=
a_hat pointwise, every mapped muscle, every timestep, regardless of a_hat's
own scale. The residual could only push a mapped muscle's activation up,
never correct it down, no matter how wrong a_hat was for that instant of the
gait cycle. That specifically handicapped the actuators that *do* carry a
real EMG prior, relative to a null_prior control (a_hat=0 everywhere, so its
residual always had the full [0, residual_weight] range available in either
direction). The convex-combination form fixes this: a_full can now pull
final_action toward a_hat's own value from either side, restoring genuine
bidirectional correction -- verified directly (not just by construction):
a live env smoke test with a_full forced to 0 showed final_action < a_hat on
mapped muscles, impossible under the old formula.

CORRECTION (caught by that same smoke test, not assumed): a_hat is NOT
bounded to [0, 1] the way this comment originally claimed. The decoder's
Sigmoid output *is* in [0,1], but `_from_unit()` (line ~131) immediately
rescales it back into raw EMG signal units via p01/p99 -- the inverse of the
[0,1] normalization the decoder was trained under (see get_synergies.py) --
before EMGToMuscleMapper's weighted average (rows sum to 1, but over
raw-unit values, not [0,1] ones) produces a_hat. Empirically, a_hat reached
~1.9 in a live smoke test at k=6. So "final_action is a convex combination
of two [0,1] values, hence bounded to [0,1]" is false in general -- the
bidirectional-correction property above still holds regardless (it only
needs a_full >= 0 and residual_weight in [0,1], not a_hat <= 1), but
final_action itself can still exceed 1, same as it could before this fix.
SconeWrapper's downstream clip to [0, 0.5] (clip_actions=True) or [0, 1.0]
otherwise is what actually bounds what the muscles see, exactly as before --
this fix changes the *shape* of that pre-clip value, not whether clipping
still matters.

Runs from before this change used the additive form and are NOT directly
comparable to runs after it: residual_weight now means "trust weight
between prior and residual" instead of "size of an additive bonus on top of
an always-full-strength prior." null_prior and plain-MPO/DEP-MPO runs are
unaffected (a_hat=0 makes the two formulas identical), so only the real-EMG
latent-prior conditions need re-running for a fair comparison.

Diagnostics (latent_residual_share etc.) are computed post-clip in
_diagnostics_post_clip(), not inline in action() -- see there for why.
"""

import gym
import numpy as np
import torch
import torch.nn as nn

from leaps.envs.emg_mapping import EMGToMuscleMapper


def build_decoder(dim_latent: int, dim_a: int = 11, hidden: int = None) -> nn.Sequential:
    """Same architecture as LatentActionAESmall.decoder in get_synergy.ipynb."""
    h = hidden or 2 * dim_latent
    return nn.Sequential(
        nn.Linear(dim_latent, h), nn.Tanh(),
        nn.Linear(h, dim_a), nn.Sigmoid(),
    )


class LatentActionPriorWrapper(gym.ActionWrapper):
    """Action space becomes (dim_latent + n_actuators,), both halves in [0, 1].

    The z half is a policy-facing unit box only -- internally it's rescaled
    to [z_lo, z_hi] (the decoder's actual trained latent range, roughly
    [-0.55, 0.55] and centered at 0, since the encoder has no output
    activation) before hitting the decoder. Without this, a [0,1]-bounded
    policy could never reach the negative half of the decoder's domain.

    residual_weight defaults to 0.5, matching the paper's humanoid setting --
    and, as of the convex-combination fix in action() below, now actually
    behaves like the paper's `w`: a blend weight between prior and residual,
    not an additive bonus on top of an always-full-strength prior.
    """

    def __init__(
        self, env, model_name: str, decoder_path: str, norm_path: str,
        dim_latent: int, residual_weight: float = 0.5, mirror_left: bool = False,
        null_prior: bool = False,
    ):
        super().__init__(env)
        actuator_names = [a.name() for a in env.unwrapped.model.actuators()]
        self.mapper = EMGToMuscleMapper(
            model_name, actuator_names, mirror_left=mirror_left, null_prior=null_prior
        )
        self.n_actuators = len(actuator_names)
        self.dim_latent = dim_latent
        self.residual_weight = residual_weight

        # For compensation tracking (see notes/representation_learning.md
        # discussion): which actuators get a real decoded prior vs. rely
        # entirely on the residual, and left/right split (left leg has no
        # prior unless mirror_left=True -- see emg_mapping.py).
        self._mapped_mask = self.mapper.weights.sum(axis=1) > 0
        self._right_mask = np.array([n.endswith("_r") for n in actuator_names])
        self._left_mask = np.array([n.endswith("_l") for n in actuator_names])
        # Per-muscle names for the actuators that *do* get an EMG prior --
        # these are the only ones where "residual" means a correction on
        # top of something, rather than being the sole control signal.
        self._mapped_names = [n for n, m in zip(actuator_names, self._mapped_mask) if m]

        self.decoder = build_decoder(dim_latent)
        self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
        # Explicit, not just map_location above -- nn.Linear() inside
        # build_decoder() creates its params on whatever torch's *current*
        # default device is (main.py sets this to cuda when available), so
        # the decoder can silently end up on GPU regardless of map_location.
        # torch.from_numpy(z) in .action() below is always CPU, so this must
        # be too, or it's a device-mismatch crash waiting on any code path
        # that doesn't happen to run inside a CPU-default subprocess.
        self.decoder.to("cpu")
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False

        norm = np.load(norm_path)
        self.p01, self.p99 = norm["p01"], norm["p99"]
        self.z_lo, self.z_hi = norm["z_lo"], norm["z_hi"]

        # policy outputs: dim_latent (z) + n_actuators (a_full)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(dim_latent + self.n_actuators,), dtype=np.float32
        )

    def _from_unit(self, x_u: np.ndarray) -> np.ndarray:
        return (x_u * (self.p99 - self.p01 + 1e-8) + self.p01).astype(np.float32)

    def _to_z_domain(self, z_u: np.ndarray) -> np.ndarray:
        return (z_u * (self.z_hi - self.z_lo) + self.z_lo).astype(np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        z_u = action[: self.dim_latent].astype(np.float32)
        a_full = action[self.dim_latent:]
        z = self._to_z_domain(z_u)

        with torch.no_grad():
            emg_u = self.decoder(torch.from_numpy(z)[None])[0].numpy()
        emg = self._from_unit(emg_u)
        a_hat = self.mapper.map_emg_to_muscles(emg)

        # HERE! Convex combination, not addition -- see the module
        # docstring's HERE! note for the full reasoning. a_hat and a_full are
        # both in [0, 1], so this blend is too, and a_full can now pull
        # final_action down toward 0 as well as up toward 1 relative to
        # a_hat. The old `a_hat + residual_weight * a_full` could only ever
        # push activation up (both terms were >= 0), so a mapped muscle's
        # activation could never be corrected below whatever the EMG prior
        # said for that instant, no matter how wrong it was.
        final_action = (1 - self.residual_weight) * a_hat + self.residual_weight * a_full

        # HERE! Diagnostics used to be computed right here, on unclipped
        # a_hat/final_action. SconeWrapper still clips the actuator input
        # downstream (to [0, 0.5] when clip_actions=True, [0, 1.0] otherwise
        # -- see deprl/env_wrappers/scone_wrapper.py), so we just stash the
        # raw pieces here; the actual diagnostics are computed post-clip in
        # _diagnostics_post_clip(), called from SconeWrapper._inner_step once
        # the real clip bound is known.
        self._last_a_hat = a_hat
        self._last_final_action = final_action

        return final_action

    def _diagnostics_post_clip(self, clip_lo: float, clip_hi: float) -> dict:
        """HERE! Residual/prior diagnostics computed on the values actually
        applied to the muscles, not the raw a_hat/final_action from action().

        Both a_hat and final_action get clipped to the same [clip_lo, clip_hi]
        SconeWrapper applies to the real actuator input. final_action is a
        convex combination of a_hat and a_full (both in [0, 1]), so it's
        already within [0, 1] before this clip -- clip_lo/clip_hi is a
        separate, tighter physiological constraint (0.5 when
        clip_actions=True), not there to contain overflow from this formula.

        residual = final_action - a_hat is signed now (post convex-
        combination fix -- see action()'s HERE! comment): positive means the
        policy pushed activation above the prior, negative means it pulled
        activation below the prior -- a real downward correction, which the
        old additive formula could never produce. Every field below already
        used abs() rather than a raw signed value for exactly this reason (a
        correction that flips sign across the gait cycle shouldn't cancel
        itself out in the mean), so the dict computation below is unchanged
        by the fix -- only the meaning of "residual" underneath it is.
        """
        eps = 1e-8
        a_hat = np.clip(self._last_a_hat, clip_lo, clip_hi)
        final_action = np.clip(self._last_final_action, clip_lo, clip_hi)
        residual = final_action - a_hat

        mapped_a_hat = a_hat[self._mapped_mask]
        mapped_residual = residual[self._mapped_mask]
        diagnostics = {
            "train/latent_prior_mean_mapped": float(mapped_a_hat.mean())
            if self._mapped_mask.any() else 0.0,
            # Unmapped actuators (a_hat=0 always) are unaffected by the
            # convex-combination fix: residual here is still exactly
            # residual_weight * a_full, still >= 0, same as before the fix.
            "train/latent_residual_mean_unmapped": float(residual[~self._mapped_mask].mean())
            if (~self._mapped_mask).any() else 0.0,
            "train/latent_residual_share": float(
                np.abs(residual).sum() / (np.abs(residual).sum() + np.abs(a_hat).sum() + eps)
            ),
            "train/latent_left_leg_mean_action": float(final_action[self._left_mask].mean())
            if self._left_mask.any() else 0.0,
            "train/latent_right_leg_mean_action": float(final_action[self._right_mask].mean())
            if self._right_mask.any() else 0.0,
            # Signed on purpose (net direction, not correction magnitude) --
            # can now be negative when mirror_left=True (left leg is mapped
            # in that case), where before it was always >= 0 by construction.
            "train/latent_left_leg_residual_mean": float(residual[self._left_mask].mean())
            if self._left_mask.any() else 0.0,
            # The actual "is the policy correcting a bad EMG prior" signal --
            # everything above either mixes in the 12 unmapped actuators
            # (residual is their *only* driver, not a correction) or uses a
            # signed mean, which cancels a real correction that flips sign
            # across the gait cycle. abs() + restricted to mapped-only fixes
            # both. Ratio compares correction size to the prior it's sitting
            # on top of, per mapped muscle only.
            "train/latent_residual_absmean_mapped": float(np.abs(mapped_residual).mean())
            if self._mapped_mask.any() else 0.0,
            "train/latent_prior_absmean_mapped": float(np.abs(mapped_a_hat).mean())
            if self._mapped_mask.any() else 0.0,
            "train/latent_residual_to_prior_ratio_mapped": float(
                np.abs(mapped_residual).mean() / (np.abs(mapped_a_hat).mean() + eps)
            ) if self._mapped_mask.any() else 0.0,
        }
        # Per-muscle, so a mismatch localized to e.g. just soleus_r doesn't
        # get diluted into a leg-wide average.
        for name, r in zip(self._mapped_names, mapped_residual):
            diagnostics[f"train/latent_residual_abs/{name}"] = float(abs(r))

        return diagnostics
