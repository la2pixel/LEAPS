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

from typing import Optional

import gym
import numpy as np
import torch
import torch.nn as nn

from leaps.envs.emg_mapping import EMG_CHANNELS, EMGToMuscleMapper
from leaps.envs.phase_mirror import PhaseMirror


def build_decoder(dim_latent: int, dim_a: int = 11, hidden: int = None, cond_dim: int = 0) -> nn.Sequential:
    """Same architecture as LatentActionAESmall.decoder in get_synergy.ipynb.
    cond_dim widens the input layer for a conditioned decoder (e.g.
    speed_cond below) -- 0 reproduces the original unconditioned shape."""
    h = hidden or 2 * dim_latent
    return nn.Sequential(
        nn.Linear(dim_latent + cond_dim, h), nn.Tanh(),
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
        null_prior: bool = False, mapped_residual_weight: Optional[float] = None,
        untrained_decoder: bool = False, speed_cond: Optional[float] = None,
        proxy_map: Optional[dict[str, str]] = None, mirror_mode: str = "static",
    ):
        super().__init__(env)
        actuator_names = [a.name() for a in env.unwrapped.model.actuators()]
        # mirror_mode="static" reproduces every existing config's behavior
        # exactly (mirror_left passed straight through to EMGToMuscleMapper,
        # same-instant copy). "phase" builds real-content-only ("_r" rows)
        # weights instead and hands them to a PhaseMirror, which fills the
        # "_l" rows itself with a heel-strike-timed lag -- see
        # phase_mirror.py for why this is a separate stateful class rather
        # than a EMGToMuscleMapper option.
        if mirror_mode not in ("static", "phase"):
            raise ValueError(f"mirror_mode must be 'static' or 'phase', got {mirror_mode!r}")
        self.mirror_mode = mirror_mode

        # proxy_map (e.g. envs.H2190_SCOPE_PROXY): extends both the real
        # content mapping and mapped_residual_weight's scope onto actuators
        # with no EMG channel of their own but a strong anatomical synergist
        # that does -- see emg_mapping.py for why this matters on low-coverage
        # bodies. None reproduces every existing config's behavior exactly.
        self.mapper = EMGToMuscleMapper(
            model_name, actuator_names,
            mirror_left=(mirror_left if mirror_mode == "static" else False),
            null_prior=null_prior, proxy_map=proxy_map,
        )
        self.phase_mirror = (
            PhaseMirror(self.mapper.weights, actuator_names) if mirror_mode == "phase" else None
        )
        # Looked up once here, not per step -- same contact_force() access
        # pattern as gaitgym.py's _get_self_contact(), which is what
        # confirms "calcn_r"/"calcn_l" are the actual foot bodies.
        self._foot_bodies = None
        if mirror_mode == "phase":
            bodies = {b.name(): b for b in env.unwrapped.model.bodies()}
            self._foot_bodies = {"r": bodies["calcn_r"], "l": bodies["calcn_l"]}
        self.n_actuators = len(actuator_names)
        self.dim_latent = dim_latent

        # For compensation tracking (see notes/representation_learning.md
        # discussion): which actuators get a real decoded prior vs. rely
        # entirely on the residual, and left/right split (left leg has no
        # prior unless mirror_left=True -- see emg_mapping.py). Computed from
        # a null_prior=False reference mapper regardless of this wrapper's
        # own null_prior setting -- null_prior deliberately leaves
        # self.mapper.weights all-zero (see EMGToMuscleMapper), so reading
        # the mask off self.mapper directly would silently give an
        # all-False mask on every null_prior run, breaking both the
        # diagnostics below and the mapped_residual_weight override.
        _reference_mapper = EMGToMuscleMapper(
            model_name, actuator_names, mirror_left=mirror_left, null_prior=False,
            proxy_map=proxy_map,
        )
        self._mapped_mask = _reference_mapper.weights.sum(axis=1) > 0
        self._right_mask = np.array([n.endswith("_r") for n in actuator_names])
        self._left_mask = np.array([n.endswith("_l") for n in actuator_names])
        # Per-muscle names for the actuators that *do* get an EMG prior --
        # these are the only ones where "residual" means a correction on
        # top of something, rather than being the sole control signal.
        self._mapped_names = [n for n, m in zip(actuator_names, self._mapped_mask) if m]

        # residual_weight is normally a scalar (paper's fixed blend weight
        # everywhere). mapped_residual_weight lets the hard-constraint
        # ablation pin a *different*, typically much lower, weight on just
        # the actuators that actually have a real EMG-derived prior --
        # everything else keeps the plain scalar residual_weight. Kept as a
        # plain float when mapped_residual_weight is unset so every existing
        # config's behavior is bit-for-bit unchanged.
        if mapped_residual_weight is not None:
            w = np.full(self.n_actuators, residual_weight, dtype=np.float32)
            w[self._mapped_mask] = mapped_residual_weight
            self.residual_weight = w
        else:
            self.residual_weight = residual_weight

        # speed_cond: a fixed, known walking speed (m/s) to condition the
        # decoder on every step -- unlike gait phase (no online ground-truth
        # signal exists in the sim), every sconewalk_* env targets one
        # constant speed, so this is genuinely deployable, not just a
        # training-time diagnostic. Requires a decoder trained with
        # speed_train (see leaps/models/train_speed_decoder.py) and a
        # norm.npz with speed_lo/speed_hi -- widens the decoder's input
        # layer by 1, so an unconditioned decoder.pt will not load into a
        # speed_cond decoder shape (mismatched state_dict, fails loudly).
        self.speed_cond = speed_cond
        cond_dim = 1 if speed_cond is not None else 0
        self.decoder = build_decoder(dim_latent, cond_dim=cond_dim)
        # HERE! untrained_decoder is the content-vs-availability control: skip
        # loading the fitted state dict entirely, so self.decoder stays at
        # its random nn.Linear init. This gives a_hat the same nonzero,
        # similarly-scaled output range as the real condition (same p01/p99
        # rescaling below, same Sigmoid-bounded architecture) but with no
        # learned relationship to actual muscle synergy content -- unlike
        # null_prior (a_hat=0 identically), which under a tight
        # mapped_residual_weight forces those actuators toward near-total
        # paralysis (final_action = mapped_residual_weight * a_full, e.g. a
        # hard 0.1 ceiling at mapped_residual_weight=0.1) rather than
        # isolating whether the EMG *content* specifically matters. Same
        # architecture/rescaling as the trained case in every other respect,
        # so this isolates content from availability instead of conflating
        # "no prior" with "crippled activation range."
        if not untrained_decoder:
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

        if speed_cond is not None:
            speed_lo, speed_hi = float(norm["speed_lo"]), float(norm["speed_hi"])
            speed_u = (speed_cond - speed_lo) / (speed_hi - speed_lo)
            # HERE! explicit device="cpu" -- torch.tensor() without it picks
            # up the ambient default device (main.py sets this to cuda when
            # available), while torch.from_numpy(z) in action() below is
            # always CPU regardless. Mismatched device torch.cat crashed a
            # live run on this exact gotcha (already flagged for the decoder
            # itself a few lines up) before this fix.
            self._speed_cond_tensor = torch.tensor([[speed_u]], dtype=torch.float32, device="cpu")
        else:
            self._speed_cond_tensor = None

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
            z_t = torch.from_numpy(z)[None]
            if self._speed_cond_tensor is not None:
                z_t = torch.cat([z_t, self._speed_cond_tensor], dim=-1)
            emg_u = self.decoder(z_t)[0].numpy()
        emg = self._from_unit(emg_u)
        if self.mirror_mode == "phase":
            contact_r = np.sum(np.abs(self._foot_bodies["r"].contact_force().array()))
            contact_l = np.sum(np.abs(self._foot_bodies["l"].contact_force().array()))
            a_hat = self.phase_mirror.step(emg, contact_r, contact_l)
        else:
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
        # Raw decoded EMG, pre-muscle-mapping -- 2026-08-11: previously
        # computed every step and discarded, never logged. Not clipped (the
        # physiological clip applies to actuator inputs post-mapping, not
        # here) -- logged as-is.
        self._last_emg = emg

        return final_action

    def reset(self, **kwargs):
        # PhaseMirror's buffer/heel-strike state must not carry over between
        # episodes -- a stale buffer from the previous episode would produce
        # a nonsense lagged lookup on the first steps of a new one.
        if self.phase_mirror is not None:
            self.phase_mirror.reset()
        return self.env.reset(**kwargs)

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

        # 2026-08-11: per-muscle PRIOR magnitude (as opposed to the residual
        # correction on top of it) and the raw pre-mapping 11-channel decoded
        # EMG -- neither was logged before, so "which of the 11 EMGs/mapped
        # muscles does the prior itself lean on, and does that shift over
        # training" was unanswerable from any existing run.
        for name, a in zip(self._mapped_names, mapped_a_hat):
            diagnostics[f"train/latent_prior_abs/{name}"] = float(abs(a))
        for name, e in zip(EMG_CHANNELS, self._last_emg):
            diagnostics[f"train/decoded_emg/{name}"] = float(e)

        return diagnostics
