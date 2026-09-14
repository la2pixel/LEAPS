# Draft — Introduction, Related Work, Methods

Status: first full draft, 2026-09-02. Written from the locked spine
(`notes/spine.md`), the verified related-work audit (`project_leaps_overview`
memory), and current repo/config state. Supersedes `expose.md` as the
citable version for these three sections — `expose.md` predates R1–R5, the
reframe, and any h1622/h2190 results, and reads as an early hypothesis
document, not thesis prose. Kept on disk for its own record, not deleted.

Open items flagged inline with **[TBD]** — mostly the clip/reward-variant
recipe decision (still running) and exact figure/table numbers pending R5.

---

## 1. Introduction

Reinforcement learning on musculoskeletal models is harder than on
torque-actuated robots for a specific, structural reason: these systems are
overactuated, and independent per-actuator exploration noise cancels out
across antagonist muscle pairs before it ever produces useful movement.
Schumacher et al. (2023) showed that this exploration problem, not credit
assignment or reward design, is the dominant obstacle for RL in
overactuated, muscle-driven control, and addressed it with DEP-RL — an
online, self-organizing correlation-based exploration signal that discovers
structured, coordinated activation patterns without any external
demonstration data.

Real human muscles already solve this coordination problem: agonist and
antagonist muscles are recruited together, not independently, and surface
electromyography (EMG) is a direct, minutes-to-record measurement of that
coordination. This raises a natural question: can recorded human EMG serve
as a *prior* on muscle coordination, replacing or augmenting an online
mechanism like DEP-RL's self-organizing exploration, or a purely
demonstration-based prior?

The architecture most directly suited to testing this is Latent Action
Priors (LAP; Hausdörfer et al., 2024) — a frozen, low-dimensional decoder
trained on a small set of expert demonstrations, wrapped around an RL
policy so the policy's action is a blend of the decoded prior and a learned
residual. LAP's own reported priors, however, are drawn exclusively from
sources that already presuppose a controller: hand-designed feedforward
trajectories, or rollouts from an RL policy already trained on the same
embodiment. Every prior-work source we are aware of shares this property —
hand-tuned feedforward control, self-trained policy rollouts
(self-distillation), inverse-simulation-derived synergies (Park et al.,
2026), and self-play-derived synergies (SAR) are all, in a precise sense,
*circular*: none of them can exceed the quality of a controller the
practitioner must already have. For a 90-muscle musculoskeletal model, this
circularity becomes a structural dead end — nobody hand-designs ninety
excitation trajectories, and no pre-existing policy exists to roll out.

Recorded human EMG breaks this circularity. It is an independent
measurement of biological muscle coordination that requires no working
controller to obtain, and — to the best of our knowledge — no prior work
on action-space priors for musculoskeletal RL uses it as the prior source.
This thesis adapts the LAP architecture to use a frozen EMG-synergy decoder
as the prior, and asks a direct empirical question: under what conditions,
if any, does an EMG-derived prior actually help — in task performance, in
sample efficiency, and in the biomechanical realism of the resulting
movement — as anatomical fidelity (and therefore the actuator space the
prior must cover) scales up.

We report this as a **characterization, not a state-of-the-art result**.
The central finding is not "EMG priors win," uniformly — it is a precise
account of *when* they win, *when* they don't, and *why*, across three
musculoskeletal models of increasing complexity (18, 22, and 90 muscles)
and multiple ways of injecting the prior into the policy's action space.
Four results anchor this account:

1. **The prior helps when it is structurally forced to** — under a
   hard-constraint blend, real EMG content beats a null (zeroed) prior with
   a clear, replicated margin, and beats an untrained-decoder control even
   more decisively, on every model tested. Sample efficiency is measurably
   better under the real prior. Under a *soft* blend, this separation
   disappears — real content and no content perform statistically
   indistinguishably, which we treat as an informative contrast, not a
   failure of the method.
2. **A cross-domain hyperparameter finding**: LAP's own recommendation
   (`w=0.5`, prior/residual blend weight, tuned for torque-actuated
   humanoids) does not transfer to muscle-actuated MPO — `w≤0.1` is as good
   or better, and more stable, on all three bodies we tested.
3. **Individual EMG content is traceable through the whole pipeline**: two
   subjects with distinct EMG content produce distinguishably different
   simulated gaits, and this distinction survives both the decoding step
   and 10M steps of RL training.
4. **A negative result, reported as a finding rather than suppressed**:
   conditioning the prior on a per-episode target walking speed — LAP's own
   suggested extension — does not separate a real EMG prior from a
   content-free control on the one body we tested it on, though it does
   change the *character* of the resulting muscle activation (smooth and
   low vs. saturated and bang-bang) without changing task reward.

**[TBD]** A fifth axis — direct biomechanical-fidelity comparison against
real human kinematics, joint moments, and cross-subject EMG variability,
replicating the evaluation protocol of Park et al. (2026) — is in progress
at the time of writing and will be reported alongside these four.

---

## 2. Related Work

**DEP-RL (Schumacher et al., 2023, ICLR, arXiv:2206.00484).** Introduces an
online, self-organizing exploration signal for overactuated systems:
`a_t = tanh(κ·C·s_t + h_t)`, where `C` is a running correlation matrix
updated from recent state derivatives (`τĊ = ṡₜṡₜ₋Δₜᵀ − C`). This
requires no external demonstration data at all — the coordination structure
emerges purely from the agent's own recent experience. We use DEP-RL
(specifically, MPO combined with DEP exploration, "DEP-MPO") as one of our
two content-free backbone baselines throughout, and as the point of
comparison for whether an EMG prior can substitute for or improve on an
online self-organizing mechanism.

**Lattice (Chiappa et al., 2023, arXiv:2305.20065).** Structures
exploration noise across actuators and time by injecting correlated noise
into the policy's latent state rather than independent per-actuator
Gaussian noise, modeled as a full-covariance multivariate Gaussian.
Reported gains of +18% reward (SAC, PyBullet Humanoid) and 20–60% better
energy efficiency (PPO, MyoSuite musculoskeletal tasks). Like DEP-RL, this
is a purely online mechanism with no external demonstration data — a
natural structured-exploration counterpart to our residual-policy design,
not implemented in the present work.

**Latent Action Priors (Hausdörfer et al., 2024, arXiv:2410.03246, IROS
2025).** The architecture this thesis's `LatentActionPriorWrapper` is based
on: a small set of expert demonstrations is compressed into a
low-dimensional latent action space via a frozen autoencoder, and an RL
policy learns a residual on top of the decoded prior,
`a = (1-w)·decode(z) + w·a_residual`, evaluated on torque-actuated
robots (Ant, HalfCheetah, Humanoid — MuJoCo; Unitree A1, H1). Two points of
divergence from the reference method are load-bearing for our setting.
First, their expert data is *already in the target actuator space* — a
robot's own torque trajectory, either hand-designed or rolled out from an
already-trained policy on the same embodiment — so no cross-domain mapping
step exists in their method. Ours does: we decode into an 11-channel
EMG space first and map to muscles second, because no actuator-matched
expert demonstration exists for a human EMG dataset — a necessary
adaptation, not an arbitrary deviation, and the direct source of the
partial-actuator-coverage problem this thesis characterizes (§4). Second,
their own reported prior sources — hand-designed feedforward control or
self-trained policy rollouts — are exactly the circular sources motivating
this work's use of EMG instead (§1).

**Park et al. (2026, arXiv:2603.10474), "Muscle Synergy Priors Enhance
Biomechanical Fidelity."** Uses synergies derived from inverse
musculoskeletal simulation (not EMG) as a *hard* constraint (not a soft
residual blend) on a 90-muscle model trained across varying speed, slope,
and terrain, evaluated primarily via biomechanical-fidelity metrics —
per-muscle correlation against experimental EMG and RMSE-ratio against
cross-subject variability (Camargo et al.; Scherpereel et al.) — rather
than task reward. Reports markedly more realistic knee kinematics, joint
moments, and ground-reaction forces than an unconstrained baseline. This is
the strongest existing evidence that a synergy-derived prior *can* produce
real fidelity gains, and directly motivates both our hard-constraint
condition (§1, finding 1) and the evaluation protocol we adopt for the
fifth, in-progress result (§1). Their own inverse-simulation synergy source
achieves full actuator coverage by construction — a comparison point for
why our EMG-based prior's partial coverage is a genuine, harder-case test
of the same idea, not a strictly worse replication of theirs.

**SAR — Synergistic Action Representation.** Derives synergies from an
agent's own self-play rollouts rather than external data, and combines
prior and free action via a convex blend, `a = φ·a_SAR + (1-φ)·a_open`.
Independently confirms the convex-combination blend form used in this
thesis's corrected implementation (three sources — LAP's own shipped code,
Park et al.'s framing, and SAR — now agree on this form).

**Reward and training baseline (arXiv:2309.02976), "Natural and Robust
Walking using Reinforcement Learning without Demonstrations in
High-Dimensional Musculoskeletal Models."** Uses the same three
musculoskeletal models (h0918, h1622, h2190) as this thesis. Our `full`
reward-variant coefficients and adaptive effort-cost mechanism trace
term-for-term to this paper's formulation; our MPO hyperparameters match
theirs on 11 of 13 fields. Reports naturalism benchmarks (percentage of a
gait cycle within one standard deviation of real kinematic data) for these
exact models with *no* demonstration data at all (H0918 ≈ 67%, H1622 ≈
73%, H2190 ≈ 50%) — a baseline this thesis's biomechanical-fidelity work
(§1, item 5) is evaluated against.

**Badie et al. (2025), "Bioinspired morphology and task curricula for
learning locomotion in bipedal muscle-actuated systems," Communications
Engineering.** A double curriculum (morphological scaling plus staged task
difficulty) rather than a demonstration- or synergy-based prior; addresses
a substantially harder problem (learning to run from scratch, multi-speed)
than the single fixed-velocity walking task used for the core comparisons
in this thesis. Included as directly relevant prior work from the same
research lineage, not as a method this thesis builds on.

**Positioning.** Across every related-work source reviewed, none uses
recorded human EMG as the coordination-prior source: Park et al.'s synergy
basis is inverse-simulation-derived; SAR's is self-play; Lattice injects
pure online noise with no external data at all; LAP's expert data is
same-embodiment robot demonstration. This is the precise novelty this
thesis's method occupies.

---

## 3. Methods

### 3.1 Simulation environment and body models

All experiments use the SCONE/Hyfydy musculoskeletal simulator with three
bipedal body models of increasing anatomical complexity, referred to
throughout by their model identifiers:

| model | total actuators (both legs) | muscles/leg |
|---|---|---|
| h0918 | 18 | 9 |
| h1622 | 22 | 11 |
| h2190 | 90 | 45 |

The task is walking at a target forward velocity; episodes terminate on a
fall or after a fixed horizon. **[TBD: confirm exact episode length /
termination criteria for the final writeup.]**

### 3.2 EMG dataset and coverage

The coordination prior is trained on the Camargo et al. dataset — 22
subjects, approximately 51,000 gait strides, 11-channel surface EMG
(gastrocnemius medialis, tibialis anterior, soleus, vastus medialis, vastus
lateralis, rectus femoris, biceps femoris, semitendinosus, gracilis,
gluteus medius, external oblique) across multiple locomotion modes.

Surface EMG can only ever measure superficial muscles; deep or small
muscles (iliopsoas, gluteus maximus, the adductors, vastus intermedius)
have no possible surface-EMG substitute at any model scale — a physical
limit of the sensing modality, not a data-collection gap. This produces a
coverage fraction that *falls* as anatomical fidelity rises, which is the
structural mechanism this thesis's characterization is built around:

| model | actuators with a real-EMG substitute | coverage |
|---|---|---|
| h0918 | 12 / 18 | 67% |
| h1622 | 12 / 22 | 55% |
| h2190 | 22 / 90 | 24% |

(These are re-verified counts from the actual `EMGToMuscleMapper` output,
not estimated from muscle-group tables; an earlier h1622 estimate of 64%
did not match and is superseded here.)

### 3.3 Synergy decoder

A frozen autoencoder is trained per subject on the Camargo EMG data: one
hidden layer of width `2k` (tanh activation), sigmoid output, `k`-dimensional
latent code. The training objective combines reconstruction error with a
soft-barrier penalty on latent magnitude, `exp((‖a_l‖∞/scale)^10) − 1`
above a threshold (this work: threshold = 0.5, scale = 0.6). This
architecture and loss form replicate the *prose* of Hausdörfer et al.'s
decoder description; their actual shipped reference implementation uses
two hidden layers, not one, and a hard dead-zone inside the barrier term
that their paper text does not mention — a documented, deliberate
deviation, not an oversight (§2, LAP).

`k = 6` was selected via a held-out-R2 sweep over `k = 1..11` (the ceiling
set by the 11-channel EMG data itself): reconstruction quality climbs
monotonically to `k = 11` but the marginal gain past `k = 6` is small
(`k=6` already captures ~96% of the `k=11` asymptote), and this is
corroborated by a peak-corrected RL comparison across `k`, not the elbow
alone.

Production decoder training uses a single subject (`AB06`), not the full
22-subject pool: this avoids the single-gait-type overfitting failure mode
Hausdörfer et al. explicitly flag as a limitation of their own
one-gait-cycle training data, since the AB06 decoder still trains on
~1,984 strides spanning four locomotion modes for that subject, not one
repeating cycle.

### 3.4 Latent Action Prior wrapper

At each step, the policy outputs both a latent code `z` and a full-space
residual action `a_full`. The frozen decoder maps `z` to an EMG-channel
activation, which is then mapped to the subset of model muscles with a
real-EMG substitute (`EMGToMuscleMapper`) to produce `a_hat` — zero for any
actuator without a surface-EMG channel. The final action is a genuine
convex combination, not an additive correction:

```
final_action = (1 − w) · a_hat + w · a_full        (mapped actuators)
final_action = a_full                                (unmapped actuators)
```

Unmapped actuators are always fully free (`w = 1`) — the prior has no
content to offer them by construction. `mirror_left` (used throughout)
synchronously copies right-leg decoder weights onto the left leg; this is
a naive bilateral copy with no gait-phase offset, not a phase-mirrored
variant.

Two content-free control conditions isolate whether an effect is due to
the EMG *content* specifically, rather than to the wrapper's action-space
reparameterization on its own: a **null-prior** control forces `a_hat ≡ 0`
(the wrapper and blend mechanism are active, but there is no prior
content), and an **untrained-decoder** control uses a randomly initialized,
never-trained decoder in the same position.

### 3.5 RL algorithm and training

Policies are trained with MPO (off-policy), specifically the `TunedMPO`
variant from DEP-RL, chosen for direct comparability with the DEP-MPO
backbone baseline; this is a deliberate divergence from Hausdörfer et al.'s
on-policy PPO. The DEP-MPO backbone combines the same MPO algorithm with
DEP-RL's online correlation-based exploration signal (`κ = 1000`, matching
the value used in the reference authors' own published, distributed
baseline checkpoints for these exact three body models — not the value
associated with a different, arm-reaching task in the DEP-RL paper's own
hyperparameter table, an earlier misattribution corrected against direct
verification). Networks are 256-wide, two hidden layers; 20 parallel
environment workers per training run.

### 3.6 Reward variants **[methodological decision still in progress — see note]**

Two reward variants are used. `onlyVelRew` is a deliberately minimal,
floor-only velocity term (`exp(-(v - v_target)²)` while below target,
flat 1.0 once at or above it; every other coefficient — ground-reaction-
force shaping, joint-limit penalty, action smoothness, muscle-count cost —
is zero), used as a controlled backbone comparison across conditions for
results 1–4 (§1). `full` activates every coefficient and is closer to a
natural-walking objective (its coefficient values trace to the reward
formulation of arXiv:2309.02976, §2).

**Note on current status**: an in-progress investigation is isolating
whether gait-quality problems observed under `onlyVelRew` (in particular,
an unrealistic running-like gait on the most anatomically complex model)
are driven by the reward variant, by the action-clipping bound
(`clip_actions`, capping muscle excitation at 0.5 vs. leaving it
unbounded), or both — since the `onlyVelRew` floor reward has no mechanism
distinguishing an efficient walk from a running-style gait once the
velocity target is cleared. This will determine which reward
variant/clip-bound combination is reported as the primary recipe for the
biomechanical-fidelity results (§1, item 5); results 1–4 do not depend on
its outcome, as they are relative comparisons within one fixed recipe.

### 3.7 Evaluation

Task performance is `test/episode_score`, deterministic evaluation on
held-out episodes at fixed intervals during training. Convergence is
assessed by trend (slope over the last-N evaluation window) and gap-to-peak,
not variance alone — a single variance threshold can misclassify a run
that is still climbing, or one that dipped and recovered within its own
averaging window, as equally "done."

**[TBD]** Biomechanical-fidelity evaluation (§1, item 5) will report:
per-muscle Pearson correlation between simulated activation and
experimental EMG envelopes against a cross-subject human benchmark
(22 Camargo subjects, pairwise, n≈231); whole-body sagittal joint
kinematics and moments against the same reference; and a phase-bucketed
RMSE-ratio-vs-cross-human-variability heatmap — replicating the evaluation
protocol of Park et al. (2026).
