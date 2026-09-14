# Results section — reframed outline (2026-09-04)

Supersedes the R1/R5 expectations in `spine.md` (2026-08-27). The Aug-27 spine
expected "F2: prior WINS on per-muscle EMG fidelity". Session analysis
2026-09-04 (h0918 + h1622 + h2190, clip/noclip, onlyVelRew + full, F2 +
kinematics + kinetics + effort + learning curves) does not support that as a
blanket claim. Reframe below.

**Thesis of the section:** *When does an EMG-derived action prior help
musculoskeletal locomotion RL, and why.* A mechanistic characterization, not
a performance claim. Genuine positive result is narrow but clean and
replicated (R2 below).

Framing (unchanged, from spine): recorded surface EMG is the only
non-circular, actuator-space coordination prior for muscle-actuated control.
Everything below is "characterization of when the only-available prior helps".

---

## R1 — The locomotion task mechanically forces most of the human EMG pattern

Claim: an unconstrained policy optimizing only forward velocity recovers
near-human distal muscle timing with no reference at all.

Evidence:
- F2 (22-subject cross-human benchmark), onlyVelRew, h0918: MPO (no prior)
  SOL 0.88 / GAS 0.64 / TA 0.64 / VM 0.61 / VL 0.62 — at or near the human
  band. DEP-MPO similar. Replicated h1622 (SOL 0.72, GAS 0.68, TA 0.60),
  h2190 (SOL 0.77, GAS 0.87, TA 0.57).
- Mechanism: push-off (plantarflexors, ~40–55% cycle) and foot clearance
  (dorsiflexors, late swing) are mechanically obligatory for fast stable
  gait — there is no alternative solution, so RL rediscovers them. Same
  reason cross-human r is 0.65–0.90 on these muscles: shared mechanics, not
  a shared template.
- **Scope: DISTAL only.** The task forces plantar/dorsiflexor + quad timing.
  It does NOT force hamstring / glut-med timing (swing-leg placement,
  frontal-plane balance — negligible effect on forward speed). On h1622 and
  h2190 the unconstrained backbones go **anti-phase** on BF/ST/GMED (see R2).
  On h0918 (small, over-constrained) even the proximal muscles come out
  roughly right unsupervised, leaving no room for the prior.
- Figure: `results/biomech_fidelity/f2_h0918.png` (+ h1622, h2190).

## R2 — Per-muscle EMG-timing fidelity (Park F2): the prior BEATS the backbones on h1622, ties h2190, loses h0918 — and the split has a mechanism

Claim: on the aggregate per-muscle EMG-timing metric (mean Pearson r of
activation vs experimental EMG, across mapped muscles = Park et al. F2), the
prior wins on the bigger body. The direction is set by whether the
unconstrained backbone fails on the proximal (reward-underdetermined)
muscles.

**F2 mean r across muscles, onlyVelRew, current recipe** (`results/
biomech_fidelity/f2_{h0918,h1622,h2190}.{png,csv}`):
| body | real-LAP | MPO | DEP-MPO |
|---|---|---|---|
| h0918 | **0.04** | 0.47 | 0.51 |
| h1622 | **0.48** | 0.38 | 0.44 |
| h2190 | 0.30 | 0.36 | 0.24 |

- **h1622: real-LAP > both backbones.** Driven by hamstrings + glut-med:
  real-LAP BF 0.74 / ST 0.77 / GMED 0.70 (inside the human band 0.64–0.85);
  MPO BF −0.17 / ST −0.24 / GMED 0.41 — **anti-phase**. Those wins outweigh
  the prior's distal losses (SOL 0.36 vs MPO 0.72, GAS 0.38 vs 0.68).
- **h2190: real-LAP > DEP-MPO, ≈ MPO.** Same proximal pattern — real-LAP BF
  0.52 / ST 0.60 / GMED 0.41 vs every backbone negative on BF/ST/GMED.
- **h0918: real-LAP collapses (0.04).** The mechanism: on the small,
  over-constrained model the task forces the whole pattern *including* the
  hamstrings — MPO's BF/ST are already fine (0.42/0.39, not anti-phase) — so
  the prior has nothing to fix and only adds distal corruption.
- **Rule:** the prior wins F2 exactly where the unconstrained backbone goes
  anti-phase on the proximal muscles. That happens on h1622/h2190 (more
  redundant musculature, hamstring/glut-med timing genuinely
  underdetermined by the velocity reward) and not on h0918.
- Partial recovery under full reward on h0918 (real-LAP BF/ST 0.63/0.63 vs
  MPO 0.42/0.50 — prior back ahead on the proximal muscles even there).
- Kinematic corollary (h0918 onlyVelRew): prior suppresses swing-phase quad
  activation (rect_fem swing 0.28 vs MPO 0.58) → knee reaches ~human swing
  flexion (72°, human ~62°) where MPO under-flexes (41°). NB this corollary
  is h0918-specific and does NOT generalize — on h1622 the prior's knee is
  *stiff* (median flexion 3° vs MPO 22°, see R4).
  Figure: `renders/h0918_onlyvelrew_emgprior_vs_backbones_5subj.png`.

This is the F2-wins / F3-loses split the Aug-27 spine predicted, now
quantified: **the prior's contribution is per-muscle activation-timing
fidelity on redundant bodies, not whole-body mechanics (R4) or RL metrics
(R3).**

## R3 — No RL-metric benefit; the wrapper's constraint, not the EMG content, drives most effects

Claim: on the well-covered body the prior neither speeds learning nor raises
return; the trained decoder mostly buys back the action-space freedom the
wrapper removes.

Evidence (h0918 onlyVelRew, 2 seeds each, `renders/h0918_onlyvelrew_
residual_ablation_{train,test,effort}.png`):
- steps→90% ceiling: null prior (residual only) 1.2–1.8 M ≤ heavier residual
  1.6 M ≤ minimal residual 2.0–3.6 M. More freedom = faster. All reach the
  same ~1000 plateau.
- **Untrained decoder as prior: flatlines at return ~60, never walks.** A
  random frozen basis in the action path is strictly worse than no
  structure → the trained decoder's value is largely restoring lost DoF.
- Effort (⟨mean_m a_m²⟩_t): any real prior 0.33–0.36 vs null-prior 0.44 —
  ~20% less co-contraction, monotonic in constraint tightness. This + R2 are
  the only h0918 wins. Still 2–3× human activation.

## R4 — Reward shaping is a stronger implicit naturalness prior than the EMG action prior

Claim: the full SCONE reward's effort/smoothness/GRF penalties fix the
non-human features onlyVelRew leaves — for prior *and* backbone — and erase
the prior's onlyVelRew kinematic edge.

Evidence (h0918, `renders/h0918_full_natural_walking_5subj.png`):
| | peak GRF | loading rate | swing rect_fem |
|---|---|---|---|
| EMG prior onlyVelRew | 4.3 BW | 228 | 0.28 |
| EMG prior FULL | 3.3 BW | 136 | 0.07 |
| MPO onlyVelRew | 3.7 BW | 181 | 0.58 |
| MPO FULL (old recipe) | 2.7 BW | 109 | 0.17 |
- GRF impact ~halved, swing co-contraction cut ~3×, stiff knee fixed — for
  both. Under full reward MPO reaches ~human swing knee flexion too.
- Still broken for all: stance-phase knee flexion (~5° vs ~18°), ankle,
  residual impact transient (still 2.5–3× human).
- F2 under full reward: MPO-full still beats prior-full on distal (SOL 0.75
  vs 0.35). Full reward does not make the prior win on EMG.
- Caveat: no noclip MPO-full run for h0918 exists; MPO-full here is the old
  July recipe (net256, clip), n=1. Prior-full-clip n=1; prior-full-noclip
  n=3.

## R5 — Action clipping: equivalent gait, reward/stability trade-off (methods / secondary)

Claim: clip vs noclip produces indistinguishable gait fidelity; the choice
is a reward-vs-stability trade-off.

Evidence (h0918 emg_lap full, k6 w0.1):
- Reward: noclip ~9130–9330 vs clip ~9030 (~2% higher) but **noclip
  collapses to near-zero once mid-training in all 3 seeds** (recovers in
  0.4–0.8 M); clip smooth, never collapses, converges ~3× slower.
- Kinematics vs AB06: wash — clip mean-r 0.64 sits inside the noclip seed
  spread 0.49–0.80. EMG timing: wash, both far below human band.
- clip walks slower (1.14 vs 1.35 m/s), lower effort (0.085 vs 0.13–0.19),
  smoother; GRF gentler on average but confounded with speed (one noclip
  seed lands softest of all).
- No fidelity reason to prefer either. Report both; clip = the stable
  default.

## R6 — Injecting explicit timing partially recovers distal fidelity, but is bottlenecked by online phase estimation

Claim: a phase-indexed imitation reward moves the distal muscles from
anti-phase toward human, but the phase signal it depends on is unreliable
exactly while the gait is still rough.

Evidence:
- Naive kinematic style reward (DeepMimic hip/knee/ankle → AB06, style_coeff
  0.33, 2 seeds, 6.8 M): distal EMG mean r **−0.15 → +0.46** (SOL 0.00→0.50,
  GAS −0.06→0.56, TA −0.38→0.32). Does not reach the backbone (0.72),
  degrades quads (VM/VL 0.41→0.04), costs ~7% reward.
- Phase estimator (`_estimate_gait_phase`, heel-strike clock + predicted
  period), re-run on rollouts vs ground-truth heel strikes:
  - clean converged walker: period CV 3.5%, phase error **2.5% of a cycle**,
    pegged at φ=1 0.6% of the time — reliable.
  - the style-reward run's own gait (6.8 M): period CV 60%, phase error
    **22% of a cycle**, pegged at φ=1 **29%** of the time — unreliable.
  - Chicken-and-egg: needs a clean periodic gait to index the reference;
    the reward is what is supposed to produce one.
- Kinematic style probe FINISHED 2026-09-04: seed0 to 10 M, converged
  test score ~890 (vs ~1000 no-style → the ~7-11% tax). n=1 (seed1 killed).
- Muscle-space imitation reward (`_emg_imitation_reward`, 22-subj RVC
  reference, no joint-angle target) — pilot `final_experiments/other/
  emg_imit/h0918/..._c033_seed0`, 8 M, 2026-09-04. **Preliminary (6.2 M):
  NOT better than the kinematic probe.** Plateaus test score ~800 (heavier
  tax than kinematic's ~890), gaussian_vel stuck ~665 (policy trades
  velocity for imitation), raw imitation reward only ~0.34 (mediocre match),
  and it took a transient collapse at 6.2 M. c=0.33 looks too high for
  muscle space (harder-to-satisfy term dominates). Await 8 M + F2; if distal
  r does not clear ~0.5 this confirms a phase-free method (AMP) is needed,
  not a bigger/retuned tracking reward. Same phase-estimator dependency
  either way.

## R7 — Individual EMG traceability (retained from spine R3)

AB06 vs AB20: distinct EMG content → distinguishable simulated gait,
surviving decode + 10M-step RL. Personalization works at the decoder level;
RL integration not built.

## R8 — Cross-domain w / k sensitivity (retained from spine R2)

LAP's w=0.5 does not transfer to muscle-actuated MPO; w ≤ 0.1 is as good or
better and more stable on all 3 bodies. `notes/figs/sensitivity_w_by_body.png`,
`sensitivity_k_r2elbow.png`. R2 elbow → k=6.

**RL-level k-sweep (added 2026-09-10):** h0918 emg_lap onlyVelRew, w=0.1,
k ∈ {2,4,6,8,11}, n=3 seeds @20M. `biomech_fidelity.py --figure ksweep`
→ `results/biomech_fidelity/ksweep_h0918.{png,csv}`.
- **Return: flat ~1000 for every k** (confirms the R3 null is not a k
  artifact — no latent dim buys episode return on the covered body).
- **F2 mean r: flat and uniformly poor, 0.03–0.13 across all k** (SD
  0.07–0.09; k=11's 0.13 is not separable from k=2's 0.07). On h0918 the
  prior loses F2 regardless of bottleneck width. Per-muscle there IS a
  trade-off the mean hides: low k (k=2) keeps distal SOL/GAS nearer human
  (0.35/0.22, vs ≤0 for k≥4) but quads near-zero; k≥6 fixes VM/VL
  (0.43–0.59) but distal collapses. RF negative for all k (biarticular
  artifact). No clean k trend for hamstrings.
- **Effort: monotonic in k.** mean |activation| 0.38 (k=2) → 0.49 (k=11);
  rms 0.52 → 0.62. A tighter bottleneck = less co-contraction; k=2 lands
  near the R3 "real prior" effort band (0.33–0.36), k=11 exceeds the R3
  null-prior (0.44). The prior only buys the effort reduction if k is
  small.
- Reads as support for k=6 being a *compromise*, not an optimum: it is the
  R2 reconstruction elbow, mid-range on effort, and F2 is flat anyway.
- w-densify (w0.2/w0.3 ×3 @20M) finished the same batch — same one-command
  follow-up, not yet run.

## Discussion — when do action priors help locomotion, and what next

- Flat-ground walking at a target speed is close to the worst case for an
  action prior: the task is so mechanically over-constrained that ~80% of
  the human pattern is forced, leaving little residual freedom for the prior
  to fill. A prior would matter more where many solutions are viable
  (manipulation, rough terrain, no speed target).
- The prior's value is *plausibility on the task-underdetermined DoF*
  (hamstrings, glut-med), not return, sample efficiency, or whole-body
  kinematics.
- A frozen prior additionally *corrupts* the task-forced pattern by imposing
  a mistimed basis the residual cannot fully undo.
- Future work (NOT this thesis): (a) phase-conditioned decoder — φ as decoder
  input so it structurally cannot emit swing-phase soleus (inherits the same
  runtime φ-estimation issue); (b) **adversarial motion prior (AMP)** — a
  transition-level discriminator needs no explicit phase, removing the R6
  bottleneck; a muscle-activation-space discriminator would target EMG
  timing directly. Both are new methods, not revisions.

## Retired / do-not-cite

- Aug-27 spine R1 phrasing "prior helps where structurally forced" and R5
  expectation "prior WINS on F2" — replaced by R1–R3 above.
- Any pre-08-16 k=11 / generic-decoder results.
- h1622/h2190 "full reward" F2 (`f2_h1622_full`, `f2_h2190_full`) — old July
  k11 generic decoder + old MPO recipe, clip; use only to show the
  backbone-beats-old-prior direction, not the current method.
