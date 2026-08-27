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

CORRECTION (caught by that same smoke test, not assumed) -- HISTORICAL, see
RE-FIX below: for a while, a_hat was NOT bounded to [0, 1] the way this
comment originally claimed. The decoder's Sigmoid output *is* in [0,1], but
`_from_unit()` (line ~131) immediately rescaled it back into raw EMG signal
units via p01/p99 -- the inverse of the [0,1] normalization the decoder was
trained under (see get_synergies.py) -- before EMGToMuscleMapper's weighted
average (rows sum to 1, but over raw-unit values, not [0,1] ones) produced
a_hat. Empirically, a_hat reached ~1.9 in a live smoke test at k=6, and up to
5.36 on rect_fem-mapped actuators specifically in a later, fuller check
(rectus femoris's raw EMG scale is ~5-13x every other channel -- a dataset
artifact, not signal). This broke more than boundedness: it made those
actuators structurally uncorrectable back toward 0 far more often than other
muscles, for reasons that had nothing to do with the EMG content itself.

RE-FIX, 2026-08-19: a_hat is computed from `emg_u` (the raw Sigmoid output)
now, not from `emg` (the raw-unit rescale) -- `emg`/`_last_emg` is kept
around only for diagnostic logging, never fed to the blend. a_hat is once
again provably bounded [0,1] for every mapped actuator, with no cross-muscle
scale disparity -- verified against a real rollout (rect_fem max dropped
from 5.358 to 0.998). So the sentence this correction originally struck down
-- "a_hat and a_full are both in [0,1], so this blend is too" -- is true
again, this time for a load-bearing reason (uniform per-channel scale), not
by accident.

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

from leaps.envs.dep_content_prior import DepContentPrior
from leaps.envs.emg_mapping import EMG_CHANNELS, EMGToMuscleMapper


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
    """Action space is declared as (dim_latent + n_actuators,) in [0, 1], but
    that Box is aspirational, not enforced -- SconeWrapper's _inner_step
    walks the ActionWrapper chain and calls .action() directly on the raw
    agent output, with no ActionRescaler (tonic's [-1,1]-to-true-bounds
    step, deprl/vendor/tonic/environments/wrappers.py) anywhere in this
    path. The actor head (SquashedMultivariateNormalDiag) squashes through
    tanh regardless of what this class declares, so z_u = action[:dim_latent]
    genuinely arrives as [-1, 1], not [0, 1] -- confirmed against an actual
    rollout, not just the actor code (leaps/models/extract_rollout_z_distribution.py).

    The z half is rescaled to [z_lo, z_hi] (the decoder's actual trained
    latent range, roughly [-0.55, 0.55] and centered at 0, since the encoder
    has no output activation) before hitting the decoder -- via
    _to_z_domain(), which maps the true [-1, 1] input domain, not [0, 1].
    Two implementations: _range_fix_domain (default, per-dimension) or
    _whiten_domain (use_whitening=True, respects real cross-dimension
    correlation -- see those methods' own docstrings).

    residual_weight defaults to 0.5, matching the paper's humanoid setting --
    and, as of the convex-combination fix in action() below, now actually
    behaves like the paper's `w`: a blend weight between prior and residual,
    not an additive bonus on top of an always-full-strength prior.
    """

    def __init__(
        self, env, model_name: str, decoder_path: Optional[str] = None, norm_path: Optional[str] = None,
        dim_latent: int = 6, residual_weight: float = 0.5, mirror_left: bool = False,
        null_prior: bool = False, mapped_residual_weight: Optional[float] = None,
        untrained_decoder: bool = False, speed_cond: Optional[float] = None,
        proxy_map: Optional[dict[str, str]] = None,
        dep_content: bool = False, use_whitening: bool = False, whiten_radial_cap: float = 0.5,
        loosened_actuators: Optional[list[str]] = None, loosened_residual_weight: Optional[float] = None,
    ):
        # dep_content: a_hat sourced from DEP's self-organized C matrix
        # instead of the decoder -- genuinely never touches decoder_path/
        # norm_path, unlike untrained_decoder (which still loads norm.npz's
        # rescaling bounds, just skips the state dict). See
        # dep_content_prior.py for why.
        if dep_content and speed_cond is not None:
            raise ValueError("dep_content and speed_cond are mutually exclusive -- there is no decoder to condition")
        self.dep_content = dep_content
        self.use_whitening = False  # overridden below when a decoder is actually loaded
        super().__init__(env)
        actuator_names = [a.name() for a in env.unwrapped.model.actuators()]
        # Phase-mirroring (PhaseMirror, heel-strike-timed lag) removed
        # 2026-08-18 -- never beat static mirroring in the w=0.1/w=0.5 sweep
        # and was unstable specifically at w=0.1 (the operating point that
        # actually matters). Static (same-instant copy of the "_r" row onto
        # "_l") is the only mirroring mode now.

        # proxy_map (e.g. envs.H2190_SCOPE_PROXY): extends both the real
        # content mapping and mapped_residual_weight's scope onto actuators
        # with no EMG channel of their own but a strong anatomical synergist
        # that does -- see emg_mapping.py for why this matters on low-coverage
        # bodies. None reproduces every existing config's behavior exactly.
        self.mapper = EMGToMuscleMapper(
            model_name, actuator_names,
            mirror_left=mirror_left,
            null_prior=null_prior, proxy_map=proxy_map,
        )
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
        #
        # FIXED 2026-08-19: never apply the override when null_prior=True.
        # a_hat is identically 0 for null_prior (see EMGToMuscleMapper), so
        # final_action = residual_weight * a_full on mapped actuators --
        # pinning that to a low mapped_residual_weight doesn't express any
        # "trust the prior less" decision (there's no prior), it just caps
        # the actuator's reachable activation at exactly that weight (e.g.
        # 10% max activation at w=0.1) for the whole run, for no reason.
        # Verified closed-form: a_full in [-1,1], so final_action in [-w,w]
        # pre-clip -- backbones and real/untrained-prior (nonzero a_hat) have
        # no such ceiling. Null-prior now always gets the same full-range
        # residual_weight=1.0 as an unmapped actuator, regardless of w.
        if mapped_residual_weight is not None and not null_prior:
            w = np.full(self.n_actuators, residual_weight, dtype=np.float32)
            w[self._mapped_mask] = mapped_residual_weight
            self.residual_weight = w
        else:
            self.residual_weight = residual_weight

        # loosened_actuators: per-actuator override on top of mapped_residual_weight,
        # for muscles whose prior content is real but structurally asymmetric with
        # its antagonist (e.g. h1622's glut_med has an EMG channel, add_mag doesn't --
        # forcing both through the same tight blend trusts the informed side and
        # blindly constrains the uninformed one). Requires the per-actuator array
        # above, so mapped_residual_weight must be set too -- a scalar
        # self.residual_weight has no per-actuator slots to override.
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

        # speed_cond: condition the decoder on the target walking speed every
        # step -- unlike gait phase (no online ground-truth signal exists in
        # the sim), every sconewalk_* env has a known target speed, so this is
        # genuinely deployable. Requires a decoder trained with speed_train
        # (see leaps/models/train_speed_decoder.py) and a norm.npz with
        # speed_lo/speed_hi -- widens the decoder's input layer by 1, so an
        # unconditioned decoder.pt will not load into a speed_cond decoder
        # shape (mismatched state_dict, fails loudly).
        #   - a float: a fixed speed (m/s), constant for the whole run.
        #   - the string "env": track the wrapped env's own target_vel,
        #     re-read on every reset() -- for sconewalk_*_vcond-v1, where
        #     target_vel is resampled per episode from target_vel_range.
        self.speed_cond = speed_cond
        self._speed_cond_from_env = speed_cond == "env"
        if self.dep_content:
            # No decoder at all for this mode -- a_hat comes from DEP's own
            # C matrix (see dep_content_prior.py), not a decoded z. self.mapper
            # above was still built (unused for computing a_hat here, but
            # self._mapped_mask was derived from a real, null_prior=False
            # reference mapper regardless of category -- reused as-is).
            self.dep_content_prior = DepContentPrior(self._mapped_mask, self.n_actuators)
            self._speed_cond_tensor = None
        else:
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

            self.use_whitening = use_whitening
            if use_whitening:
                if "whiten_A" not in norm or "whiten_mu" not in norm:
                    raise ValueError(
                        f"use_whitening=True but {norm_path!r} has no whiten_A/whiten_mu -- "
                        "retrain the decoder with the current train_single_subject_decoder.py "
                        "(older norm.npz files predate whitening support)."
                    )
                if whiten_radial_cap <= 0:
                    raise ValueError(f"whiten_radial_cap must be positive, got {whiten_radial_cap}")
                self.whiten_A, self.whiten_mu = norm["whiten_A"], norm["whiten_mu"]
                self.whiten_radial_cap = whiten_radial_cap

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
        """Normalized target speed as the [[speed_u]] tensor cat'd onto z in
        action(). explicit device="cpu" -- torch.tensor() without it picks up
        the ambient default device (main.py sets this to cuda when available),
        while torch.from_numpy(z) in action() is always CPU; a mismatched-device
        torch.cat crashed a live run on this exact gotcha before."""
        speed_u = (speed - self._speed_lo) / (self._speed_hi - self._speed_lo)
        return torch.tensor([[speed_u]], dtype=torch.float32, device="cpu")

    def _from_unit(self, x_u: np.ndarray) -> np.ndarray:
        return (x_u * (self.p99 - self.p01 + 1e-8) + self.p01).astype(np.float32)

    def _to_z_domain(self, z_u: np.ndarray) -> np.ndarray:
        """z_u is the raw tanh-squashed actor output, [-1, 1] -- see the
        class docstring. Dispatches to whichever domain map is configured."""
        if self.use_whitening:
            return self._whiten_domain(z_u)
        return self._range_fix_domain(z_u)

    def _range_fix_domain(self, z_u: np.ndarray) -> np.ndarray:
        """Maps [-1, 1] onto [z_lo, z_hi] per dimension, independently.
        Correct range, but assumes the latent dims are uncorrelated -- they
        aren't (notebooks/autoencoder_loss.ipynb) -- so this wastes decoder
        capacity on combinations that never occur in real muscle synergies."""
        return (z_u * (self.z_hi - self.z_lo) / 2 + (self.z_hi + self.z_lo) / 2).astype(np.float32)

    def _whiten_domain(self, z_u: np.ndarray) -> np.ndarray:
        """z = whiten_A @ radial_cap(z_u) + whiten_mu, clipped to [z_lo, z_hi].

        whiten_A/whiten_mu (from norm.npz) rotate+scale by the real
        training data's covariance, so the box the policy explores matches
        real synergy correlation structure instead of an axis-aligned one.

        The radial cap runs first because the actor's independent-per-dim
        tanh output concentrates near the [-1,1]^k hypercube's *corners*
        (measured: median ||z_u|| = 86% of the max possible norm). Fed
        straight through the dense whiten_A, that combines constructively
        and overshoots [z_lo, z_hi] by ~2x, with several dims blowing out
        *simultaneously* (correlated by the rotation) -- a plain post-hoc
        clip alone doesn't fix this (~10% severe saturation vs the old
        range-fix's 3.3%). Capping ||z_u|| before the rotation does.

        whiten_radial_cap trades saturation for decoded signal variance --
        0.4-0.5 beats the old range-fix outright at ~76-85% of its
        variance; smaller values cut saturation further but flatten the
        prior toward a near-constant signal. Full derivation, trade-off
        curve, and the corner-concentration measurement:
        notebooks/autoencoder_loss.ipynb."""
        r_max = self.whiten_radial_cap * np.sqrt(len(z_u))
        r = np.linalg.norm(z_u)
        z_u_capped = z_u * min(1.0, r_max / max(r, 1e-8))
        z = self.whiten_A @ z_u_capped + self.whiten_mu
        return np.clip(z, self.z_lo, self.z_hi).astype(np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        z_u = action[: self.dim_latent].astype(np.float32)
        a_full = action[self.dim_latent:]

        if self.dep_content:
            # z_u is produced by the policy but never used -- see __init__'s
            # dep_content note. a_hat comes from DEP's own self-organized
            # C matrix instead of a decoded z.
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
            # FIXED 2026-08-19: a_hat is computed from emg_u (the decoder's
            # raw Sigmoid output, bounded [0,1] per channel), NOT from emg
            # (the raw-EMG-unit rescale via _from_unit). emg/_last_emg is
            # kept as-is purely for diagnostic logging
            # (train/decoded_emg/<channel>) -- it was never meant to feed
            # the control blend. Rationale: raw EMG-unit scale varies ~13x
            # across channels for dataset-artifact reasons (rectus femoris
            # p99 ~5-13x every other channel), which made a_hat reach up to
            # 5.36 on rect_fem-mapped actuators vs ~1.0 on others -- a
            # cross-muscle disparity with no physiological meaning, that
            # made those actuators structurally uncorrectable far more often
            # than others under the blend below. map_emg_to_muscles is a
            # weighted average with rows summing to <=1, so using emg_u
            # instead makes a_hat provably bounded [0,1] for every mapped
            # actuator -- verified against a real rollout: rect_fem max
            # dropped from 5.358 to 0.998, matching every other muscle.
            a_hat = self.mapper.map_emg_to_muscles(emg_u)

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
        out = self.env.reset(**kwargs)
        # speed_cond="env": the wrapped env resamples target_vel per episode
        # (sconewalk_*_vcond-v1), so re-read it here -- action() then decodes
        # every step of this episode conditioned on the new target speed.
        if self._speed_cond_from_env:
            self._speed_cond_tensor = self._speed_tensor(float(self.env.unwrapped.target_vel))
        return out

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
