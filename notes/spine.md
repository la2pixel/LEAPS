# LEAPS narrative spine — LOCKED 2026-08-27

Deadlines: thesis Sep 20, ICLR abstract Sep 18 / full Sep 25.
Paper shape: **characterization + enabling-modality**, not "we beat SOTA".
After this: NO new RL runs. onlyvelrew run set is frozen. Work = analysis + writing.

## Framing (not a result)

LAP needs expert actions *in the actuator space*. Every prior-work source is
circular: hand-tuned feedforward controllers or RL-policy rollouts
(self-distillation), inverse-simulation synergies (Park et al. 2026),
self-play synergies (SAR). For muscle-actuated control, **recorded surface
EMG is the only non-circular, actuator-space coordination prior** — an
independent measurement of biological muscle coordination, recorded in
minutes, never solving the control problem. The mostly-null reward result is
then *characterization of when the only-available prior helps*, not pass/fail.

Deep muscles (iliopsoas, glut_max, adductors, vas_int) have no surface-EMG
substitute at any model scale — a physical limit. Coverage: h0918 ~67%,
h1622 ~64%, h2190 ~22%.

## Load-bearing results

### R1 — the prior helps where it is structurally forced
Hard-constraint synergies, w=0.1: real EMG ≫ null, 3/3 seeds h0918,
replicated h1622. Full content ladder: trained ≫ null ≫ untrained_decoder
(a random decoder is *worse* than no prior). Sample-efficiency edge
(hard-constraint: real reaches score 8000 in 6.0M steps, null never reaches
it in 10M). Free-blend (soft w) is where real ≈ null — that contrast is the
point, not a failure.
Details/caveats: memory project_latent_emg_prior_experiment.

### R2 — cross-domain w-sensitivity
LAP's "w=0.5 for humanoids" does NOT transfer to muscle-actuated MPO.
w ≤ 0.1 is as good or better and more stable, on all 3 bodies.
Figure: `notes/figs/sensitivity_w_by_body.png`. Also
`sensitivity_k_r2elbow.png` (R2 elbow → k=6; be explicit that a matched-recipe
RL k-sweep does not exist). Artifact ce2a0d89-e7c8-4514-a56a-be44a8c432e5.

### R3 — individual EMG traceability
AB06 vs AB20: distinct EMG content → distinguishable simulated gait,
surviving decode + 10M-step RL. Personalization works at the decoder level
(own-fingerprint vs pooled R2 delta ~1e-4 after PCA-reduce + noise-inject fix);
RL integration never built.

### R4 — velocity-conditioned prior: NEGATIVE, kept as a finding
LAP's own suggested extension (condition prior on target velocity), tested
properly on h0918: `vcond_real ≈ vcond_null ≈ mpo_baseline ≈ uncond_real`
(seed0 done 10M: 970.0 / 973.9 / 972.6; uncond 973.7). Held-out speeds same —
interp equal, extrap fails identically for all arms. Only muscle-activation
*realism* differs (smooth ~0.07 vs bang-bang ~0.35). Reward is saturable
(vel_coeff=1, rest 0), DEP-MPO already solves multi-speed on h0918, easy
1-param gait family → no room for the prior. Reinforces the framing: the
prior's value is plausibility, not return.
Full: notes/experiments/velocity_conditioned.md, findings_log data.json key
`vcond_h0918`, results/vcond_heldout/.

### R5 — biomechanical-fidelity characterization (Park-figure replication) — TO BUILD
Analysis of the frozen onlyvelrew runs vs real human data. No new RL.
Target: exact-replica versions of Park et al. 2026 (arXiv:2603.10474)
Figs 1–5. Honest split expected:
- **Individual-muscle content fidelity (their F2): the prior likely WINS.**
  real-LAP variants beat mpo/dep-mpo badly on per-muscle EMG match
  (h1622 glut_med_r 0.82–0.85 vs 0.20–0.49). Cross-human benchmark framing
  (22 Camargo subjects, pairwise) — expect our prior inside the human band
  on mapped muscles, backbones outside.
- **Whole-body kinematics (their F3): the prior likely LOSES/TIES.**
  memory project_gait_kinematics_comparison: plain real-LAP never the best
  kinematic match on any body; mpo often wins. State it honestly.
- **Kinetics (joint moments): mixed and coverage-dependent.**
  h0918 real-LAP ties mpo (r 0.569 vs 0.541). h1622 untouched real-LAP
  disastrous (r 0.121) but **loosening → r 0.42–0.54, a 3–4× jump** — the
  cleanest large effect in the biomech work, belongs in the deck.
- **RMSE-ratio-vs-cross-human (their F5): generous normalization**, compute
  per channel×phase and let the heatmap tell the honest cell-by-cell story.
Mechanism for the whole-body gap: getting one muscle's activation right
doesn't yield a human joint trajectory if its mechanical neighbours (the
unmapped 36%+) aren't coordinated with it. Loosening buys the mapped muscle
room to reconcile. (Orphaned-antagonist-pair framing is a CONTRIBUTING factor
— h0918 natural experiment — but NOT sufficient: loosen_tib_ant helps most
and tib_ant has no coverage gap. Don't over-claim the mechanism.)

## Park et al. 2026 figure replication plan

Their setup: H2190, 90 musc, SAC, 75M steps, 5 seeds, NMF k=10 from
MocoInverse activations (100% coverage — the key difference). Refs: Camargo
(22 subj, 0.7–1.8 m/s), Scherpereel (8 subj, ±5° slope). No tables, 5 figs.

| their fig | content | our data | feasible | notes |
|---|---|---|---|---|
| F1 | RL learning curves, indep vs synergistic, 5 seeds ±SD | score curves (real-LAP/mpo/null/dep-mpo) | yes, trivial | = R1 in their plot style |
| F2 | per-muscle Pearson r, pred activation vs exp EMG envelope, 8 musc (SOL GAS TA ST BF RF VM VL); gray box = cross-human (22 subj, n=231 pairwise) + variability band; bottom = activation waveforms over gait cycle | .sto muscle-activation cols + emg_activations_v2.h5 | **yes — build first** | our strongest axis; the fingerprint / cross-human muscle-activity analysis |
| F3 | hip/knee/ankle sagittal angle+moment, AP+vert GRF, gait-cycle norm, exp vs methods, speed color-gradient 0.7–1.8 | .sto (769-col schema has all) | **partial** — single self-selected speed only (vcond killed) | build single-speed version, match each condition to nearest AB06 plateau |
| F4 | same as F3 across slope ±5° | none | **no** | out of scope; note "single-condition equivalent only" |
| F5 | heatmap: phase-dependent RMSE ratio (sim err ÷ cross-human RMSE; ~1.0 = within human variability), phases (loading response/mid-stance/pre-swing/swing) × vars, indep vs synergistic | .sto + cross-human RMSE from Camargo | **yes — headline metric** | generous normalization; compute per channel×phase |

## Build order (analysis only)

1. Cross-human benchmark cache: emg_activations_v2.h5, 22 Camargo subjects,
   treadmill strides near 1.2 m/s → pairwise Pearson r per muscle (n≈231) and
   pairwise RMSE per channel×gait-phase. The denominator for F2 and F5.
2. Promote `notebooks/biomech_fidelity.ipynb` cells 17–20 to
   `leaps/scripts/biomech_fidelity.py` (body, condition, checkpoint) →
   per-channel r, nRMSE, RMSE-ratio, phase-bucketed. Drop stale cells 1–16
   (pre-bugfix k=11, do not cite).
3. F2 → F5 → F3 → F1.
4. Conditions: real-LAP (tight_w01) · mpo · dep-mpo · null; + h1622
   loosen_glut_med / loosen_tib_ant.
5. Bodies: h0918 + h1622 clean; h2190 from pre-collapse checkpoint
   step_18000000 (~17M, peak ~693), flagged as not the final policy.

## Methodology caveats to carry

- Speed-matching: AB06 ankle is speed-sensitive across plateaus (r 0.60–0.91
  over 0.4 m/s gaps); hip/knee/GRF robust (r 0.97–0.99). Cite the
  ankle-excluded (hip+knee+GRF) ranking as primary; our actual residual
  speed mismatches are 0.09–0.13 m/s so hip/knee/GRF are safe.
- h2190 onlyvelrew never produces stable full-length gait (documented
  peak-then-collapse, memory project_h2190_mechanistic_findings) — report
  peak+timing, use pre-collapse checkpoints for .sto.
- onlyvelrew is a deliberately minimal reward for controlled backbone
  comparison, NOT an attempt to reproduce Schumacher natural walking (that
  needs the `full` reward variant — all coeffs on — explicitly out of scope).
