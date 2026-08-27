"""Correlated-EMG-noise exploration: an agent-wrapper class factory.

Structurally parallel to deprl.custom_agents.dep_factory (a class factory
that subclasses the base MPO instance's own class), but sources structured
exploration noise from real human EMG cross-muscle correlation instead of
DEP's online Hebbian C-matrix. Lives entirely in this repo -- nothing
under depRL-sconegym/deprl/vendor/ or depRL-sconegym/deprl/custom_*.py is
read or modified.

Mechanism: some fraction of steps (intervention_proba, mirroring
deprl.custom_agents.StochSwitchDep's own pattern), perturb the RL-proposed
action on the actuators that have a real EMG substitute (per
leaps.envs.emg_mapping.EMGToMuscleMapper) with noise_scale * (L @ eps),
where L is the Cholesky factor of one of three correlation matrices
(_mapped_cholesky): the real cross-actuator EMG correlation (default), the
identity (null_corr=True), or the real correlation values relabeled onto
different actuator pairs via a derangement (shuffle_corr=True). The
perturbed action is written back to self.last_actions, matching how
deprl.custom_agents' own DEP-switch agents log the action actually taken --
the replay buffer and critic must learn from what physics actually
executed, not from the pre-perturbation proposal.

Only correlation *structure* is borrowed, never EMG amplitude: see project
memory (session 2026-07-30) for why absolute EMG magnitude has no
principled reason to transfer from a human electrode to a simulated
muscle's activation range, while cross-muscle correlation/timing does
(confirmed directly against this project's own 22-subject dataset).

null_corr alone only proves "some correlation beats none" -- it can't rule
out "any off-diagonal structure of about this magnitude would have done
just as well, the specific anatomy was never necessary." shuffle_corr is
the control that closes that gap (2026-08-01, added after a review flagged
this as the missing piece that would make the untrained-decoder-style
content-vs-availability argument work here too): same magnitude
distribution of correlations as the real matrix, reattached to
biomechanically arbitrary pairs. Real beating shuffled (not just null) is
what would actually support "the real anatomy matters," not just "some
structure helps."
"""

from __future__ import annotations

import h5py
import numpy as np

from .envs.emg_mapping import EMGToMuscleMapper

# Verified once via a live SCONE model, not guessed -- SCONE's actuator
# ordering has no reason to match any external convention. Regenerate with:
#   env = deprl.environments.Gym('sconewalk_h0918-v1', scaled_actions=False)
#   [a.name() for a in env.unwrapped.model.actuators()]
ACTUATOR_NAMES = {
    "h0918": [
        "hamstrings_r", "bifemsh_r", "glut_max_r", "iliopsoas_r", "rect_fem_r",
        "vasti_r", "gastroc_r", "soleus_r", "tib_ant_r",
        "hamstrings_l", "bifemsh_l", "glut_max_l", "iliopsoas_l", "rect_fem_l",
        "vasti_l", "gastroc_l", "soleus_l", "tib_ant_l",
    ],
}


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A random permutation of range(n) with no fixed points.

    Rejection sampling -- fine here since n is always small (6-12 mapped
    actuators). Used by the shuffled-correlation control below: relabeling
    every actuator's identity simultaneously in both the row and column
    sense of a correlation matrix (a permutation *similarity*, P @ R @ P.T)
    is guaranteed to still be a valid correlation matrix -- same eigenvalues
    (so still PSD), and the diagonal stays exactly 1 regardless of the
    permutation, since (P R P.T)[i,i] = R[perm(i), perm(i)] = 1 for any
    permutation. That's what makes it a safe way to build a scrambled
    control: unlike resampling fresh random correlation values (which needs
    extra work to stay PSD and won't match the real magnitude distribution
    by construction), this reuses the *exact same multiset* of real
    pairwise correlation values, just reattached to different actuator
    pairs. A derangement (perm(i) != i for every i) guarantees no actuator
    can keep its own true correlation partner by chance -- a plain random
    permutation could still fix a point (or even be the identity) with
    nonzero probability, especially at small n.
    """
    perm = np.arange(n)
    while True:
        rng.shuffle(perm)
        if not np.any(perm == np.arange(n)):
            return perm


def _mapped_cholesky(
    model_name: str, h5_path: str, mirror_left: bool, null_corr: bool,
    shuffle_corr: bool, ridge: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cholesky factor of the cross-actuator correlation (or a control
    variant), restricted to actuators with a real EMG substitute.

    Three variants, not two -- null_corr alone can't isolate the actual
    claim. null_corr=True (identity) answers "does *any* correlation
    structure beat none at all." It can't answer "does it have to be the
    *real, anatomically correct* pairing, or would any off-diagonal
    structure of about the same magnitude do just as well" -- removing all
    structure at once conflates "wrong structure" with "no structure."
    shuffle_corr=True answers that second question directly: same
    correlation *values*, same magnitude distribution, reattached to
    (mostly) biomechanically arbitrary pairs via a derangement (see
    _derangement). If real beats shuffled, the specific anatomy is doing
    real work, not just "having some correlated noise." If real and
    shuffled tie (and both beat null), the mechanism only needed *some*
    structure, and the biological source wasn't actually necessary --
    same logic as the untrained-decoder control that made the
    hard-constraint result credible (content vs. mere availability), applied
    here to noise structure instead of a decoded action target.

    Returns (mapped_idx, L) -- mapped_idx are action-vector column indices,
    L is len(mapped_idx) x len(mapped_idx).
    """
    if null_corr and shuffle_corr:
        raise ValueError(
            "null_corr and shuffle_corr are mutually exclusive controls -- "
            "null tests 'some correlation vs. none', shuffle tests 'the real "
            "correlation vs. the same values scrambled onto other pairs'. "
            "Allowing both silently would leave it ambiguous which question "
            "a run was actually answering."
        )

    actuator_names = ACTUATOR_NAMES[model_name]
    mapper = EMGToMuscleMapper(model_name, actuator_names, mirror_left=mirror_left)
    mapped_idx = np.where(mapper.weights.sum(axis=1) > 0)[0]
    if len(mapped_idx) == 0:
        raise ValueError(f"no mapped actuators for model {model_name!r}")

    with h5py.File(h5_path, "r") as f:
        emg = f["activations"][:]  # (n_samples, 11), pooled across all subjects/strides

    mapped_series = mapper.map_batch(emg)[:, mapped_idx]  # (n_samples, n_mapped)
    n = len(mapped_idx)
    r = np.eye(n) if null_corr else np.corrcoef(mapped_series.T)

    if shuffle_corr:
        # Fixed per `seed` (not the class instance's own noise rng, which
        # doesn't exist yet at this point in construction) so relaunching
        # the same seed reproduces the same scrambled pairing, same as
        # every other seeded piece of this project.
        perm = _derangement(n, np.random.default_rng(seed))
        r = r[np.ix_(perm, perm)]

    r_reg = (1.0 - ridge) * r + ridge * np.eye(n)
    L = np.linalg.cholesky(r_reg)
    return mapped_idx, L


def emg_noise_factory(
    instance,
    model_name: str,
    h5_path: str,
    mirror_left: bool = False,
    null_corr: bool = False,
    shuffle_corr: bool = False,
    noise_scale: float = 0.05,
    intervention_proba: float = 0.1,
    ridge: float = 0.05,
    temporal_corr: float = 0.0,
    seed: int = 0,
):
    """Class factory. Call exactly like deprl.custom_agents.dep_factory:

        leaps.agents.emg_noise_factory(deprl.custom_mpo_torch.TunedMPO(),
            model_name='h0918', h5_path='/path/to/emg_activations_v2.h5',
        )(replay=deprl.custom_replay_buffers.AdaptiveEnergyBuffer(...))

    temporal_corr (2026-08-02, new): the noise_scale/intervention_proba
    sweep and the mirror+higher-intervention_proba run all found real vs.
    null indistinguishable -- but every one of those tested only the
    cross-actuator half of what Lattice/gSDE (the related-work paper this
    mechanism is modeled on, project memory "Related work" section) found
    to matter. Lattice's own reported benefit came from noise that is BOTH
    cross-actuator correlated AND temporally smooth (state-dependent,
    resampled once per rollout, not freshly i.i.d. at every single
    intervention). The original implementation above only ever did the
    former -- every `hit` draws a fresh independent `eps`, so consecutive
    interventions are uncorrelated in time even though they're correlated
    across muscles at any single instant. temporal_corr closes that gap
    without touching the cross-actuator mechanism at all: when >0, the
    underlying noise source is an AR(1)/Ornstein-Uhlenbeck process,
    evolving one step per env at every call (not just on intervention
    hits, so it's already "warmed up" whenever an intervention occurs),
    normalized to stay unit-variance regardless of the decay coefficient
    (`x_t = temporal_corr * x_{t-1} + sqrt(1 - temporal_corr**2) * noise`)
    so noise_scale means the same thing whether temporal_corr is 0 or not
    -- only the time-structure changes, not the magnitude, keeping this a
    fair single-variable extension of the existing real/null/shuffle
    comparison. temporal_corr=0.0 (default) reproduces the original
    per-hit-i.i.d. code path exactly, byte-for-byte -- every run launched
    before this date remains reproducible unchanged.
    """
    mapped_idx, L = _mapped_cholesky(
        model_name, h5_path, mirror_left, null_corr, shuffle_corr, ridge, seed,
    )
    n_mapped = len(mapped_idx)
    rng = np.random.default_rng(seed)

    class EMGCorrelatedNoise(instance.__class__):
        _ou_state = None  # (n_envs, n_mapped), lazily created on first step()

        def step(self, observations, steps, muscle_states=None, greedy_episode=None):
            actions = super().step(observations, steps)
            if greedy_episode:
                return actions
            n_envs = actions.shape[0]

            eps_source = None
            if temporal_corr > 0.0:
                if self._ou_state is None or self._ou_state.shape[0] != n_envs:
                    self._ou_state = rng.normal(size=(n_envs, n_mapped))
                else:
                    fresh = rng.normal(size=(n_envs, n_mapped))
                    self._ou_state = (
                        temporal_corr * self._ou_state
                        + np.sqrt(1.0 - temporal_corr ** 2) * fresh
                    )
                eps_source = self._ou_state

            hit = np.where(rng.uniform(size=n_envs) < intervention_proba)[0]
            if len(hit) > 0:
                eps = eps_source[hit] if eps_source is not None else rng.normal(
                    size=(len(hit), n_mapped)
                )
                actions[np.ix_(hit, mapped_idx)] += noise_scale * (eps @ L.T)
                self.last_actions = actions.copy()
            return actions

        def test_step(self, observations, steps, muscle_states=None):
            return super().test_step(observations, steps)

        def reset(self):
            pass

    return EMGCorrelatedNoise
