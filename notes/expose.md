# Does Real Motor Data Scale as an Exploration Prior for Overactuated RL?

**Project**: LEAPS (Learning Humanoid Locomotion from EMG-Based Latent Action Priors)
**Status**: exposé / working hypothesis, not yet a full experiment matrix

---

## 1. Motivation

DEP-RL (Schumacher et al., ICLR 2023) showed that the core obstacle to RL on
musculoskeletal models is exploration: independent Gaussian noise cancels out across
antagonist muscle pairs, so it takes DEP's online, self-organizing correlation structure
to explore effectively. Lattice (Chiappa et al., 2023) later showed the same problem can
be addressed by structuring noise across actuators and time inside the policy's latent
space. Both are *online* mechanisms — neither uses real biological motor data.

The obvious next question: if agonist-antagonist coordination is exactly what real
muscles already do, why not use recorded human EMG as the coordination signal instead
of (or alongside) a self-organizing or noise-structuring mechanism?

## 2. Related work

- **Lattice** (Chiappa, Marin Vargas, Huang, Mathis, 2023, [arXiv:2305.20065](https://arxiv.org/abs/2305.20065)).
  Injects temporally- *and* actuator-correlated noise into the policy's latent state
  instead of independent Gaussian action noise, modeling the resulting action noise as
  a full-covariance multivariate Gaussian. Lattice-SAC gets 18% higher reward than
  unstructured exploration on PyBullet Humanoid; Lattice-PPO on MyoSuite musculoskeletal
  tasks finds policies that are 20–60% more energy-efficient. Entirely online — no real
  motor data involved. This is the natural "structured exploration" partner for our
  residual policy (§5.3).

- **Latent Action Priors for Locomotion** (Hausdörfer, von Rohr, Lefort, Schoellig,
  2024, [arXiv:2410.03246](https://arxiv.org/abs/2410.03246), IROS 2025). Our
  `LatentActionPriorWrapper` is architecturally based on this paper: a small dataset of
  expert demonstrations is compressed into a latent action space that serves as an
  inductive bias, with an RL policy learning a residual on top; combined with a
  style-based imitation reward, the agent both exceeds expert-level reward and
  transfers better than an unconstrained policy. Notably, their expert data is
  low-dimensional robot torque trajectories from the *same* embodiment the policy
  controls — no cross-embodiment or sensor-coverage gap. Our setting (human EMG →
  musculoskeletal model with a different body and incomplete surface-EMG coverage,
  §4) stress-tests this architecture exactly where its core assumption — that the
  prior and the agent's action space substantially overlap — is least likely to hold.
  We currently omit their style-based imitation reward term; adding it back is a
  natural next step (§5.5).

- **Muscle Synergy Priors Enhance Biomechanical Fidelity** (Park, Choi, Ahn, Ahn,
  2026, [arXiv:2603.10474](https://arxiv.org/abs/2603.10474)). Extracts low-dimensional
  synergies from human walking trials and uses them as **hard constraints** (not a
  soft residual blend) on a 3D muscle-driven model trained across varying speed,
  slope, and terrain. Produces markedly more realistic knee kinematics, joint moments,
  and ground reaction forces than an unconstrained RL baseline, from a small amount
  of EMG data. This is the strongest existing evidence that synergy priors *can* work
  well — but it is tested on one fixed model complexity across varying task
  conditions, never across a scaling axis of muscle count/anatomical fidelity. It also
  hard-constrains rather than residual-blends, which may be precisely why it doesn't
  suffer the residual-dominance failure mode we see (§3) — an open question our
  proposal addresses directly (§5.4).

## 3. What we tried

We trained a frozen autoencoder on the Camargo EMG dataset (22 subjects, ~51k strides,
11-channel EMG) to get a `k`-dimensional synergy decoder, then wrapped a sconewalk
h0918 RL policy so it outputs a latent code `z` decoded into a muscle-activation prior,
plus an RL residual on top (`LatentActionPriorWrapper`, `residual_weight=0.5`).

**Result at k=6, sconewalk h0918, 10M steps (1 seed each so far):**

| condition | test/episode_score | notes |
|---|---|---|
| plain MPO | 9121 | no structured exploration |
| DEP-MPO | 9176–9238 (2 seeds) | current SOTA baseline |
| EMG-prior + residual (k=6) | 9141 | ≈ plain MPO, below DEP |

The prior does not help performance. More tellingly, `latent_residual_share = 0.79`:
79% of the final action magnitude comes from the RL residual, only 21% from the EMG
prior — and this residual share is essentially uniform across all six mapped muscle
groups (0.46–0.51 each). The policy isn't refining the biological prior; it's overriding
it almost everywhere, roughly equally.

## 4. Why this might not be a bug, but a scaling law

`emg_mapping.py` gives a mechanistic reason this could get *worse*, not better, with
model fidelity. Surface EMG can only ever reach superficial muscles — deep or small
muscles have no possible surface channel, not just a missing recording:

| model | muscles/leg | muscles with real EMG | coverage |
|---|---|---|---|
| h0918 | 9 | 6 (some averaged pairs) | ~67% |
| h1622 | 11 | 7 | ~64% |
| h2190 | 45 | ~9–10 | ~22% |

Iliopsoas, glut_max, the adductors, vas_int — none of these have a surface EMG
substitute at any model scale. As anatomical fidelity increases from h0918 to h2190,
the *fraction* of actuators a real-data prior can inform collapses, while the
actuators it can't touch grows combinatorially. If the residual-share trend follows
this coverage ratio, that is a quantitative, falsifiable claim: **real biological data
becomes structurally less sufficient as an exploration/coordination prior precisely as
overactuation increases** — the opposite of the intuitive assumption that more
biologically realistic models should benefit more from biological data.

## 5. Proposed paper

1. **Establish the trend is real, not a k=6 artifact.** Re-run mpo/EMG-prior with 2+
   seeds each; sweep `k` (3, 6, 8 [decoder exists], 18-full-rank) on h0918 to separate
   "not enough latent capacity" from "structural coverage gap."
2. **Scale the model axis.** Repeat the walk-task comparison on h1622 and h2190
   (currently unrun) and test whether `latent_residual_share` tracks the EMG coverage
   ratio (67% → 64% → 22%) rather than staying flat.
3. **Constructive fix #1 — structured exploration on the residual.** Layer Lattice-style
   actuator-and-time-correlated latent noise (§2) onto the residual policy itself,
   instead of plain Gaussian noise. Hypothesis: real data supplies *what* direction is
   biologically plausible; structured exploration supplies *how to search* around
   it — and the combination should degrade more gracefully with scale than either
   mechanism alone.
4. **Constructive fix #2 — hard constraint instead of soft residual.** Park et al.
   (§2) get strong fidelity from synergies precisely because they constrain the action
   space rather than let a residual override it. Re-run the k-sweep with the synergy
   decoder as a hard constraint (`residual_weight → 0`, or a reduced-rank action space)
   to test whether residual-dominance is an artifact of our soft-blend design choice
   rather than of the EMG coverage gap itself — and whether hard-constraining still
   holds up as muscle count scales to h2190, where Park et al. never tested.
5. **Reintroduce the imitation-style reward.** Hausdörfer et al. (§2) pair their latent
   prior with a style-based imitation reward; we currently don't have an analogous
   term. Adding one is a cheap way to test whether residual-dominance is a training
   *pressure* problem (nothing in the objective rewards staying close to the prior)
   rather than purely a capacity/coverage problem.

## 6. Honest scope/risk

- h1622 has no baseline runs yet; h2190 has a config but no runs. This is a genuine
  compute commitment (3 models × ~4 conditions × 2–3 seeds × 10M steps).
- The whole hook depends on the residual-share-vs-coverage trend being monotonic. A
  flat or noisy trend across model scale removes the paper's central claim.
- We only have walk-task infrastructure mature enough to trust; run-task is deferred
  to keep model scale as the only varying axis.
