# LEAPS narrative findings log

Prose findings, mechanism explanations, and standing decisions — the
"why," not the raw metric curves (those live in `data.json`/`index.html`
in this same directory). One canonical place for this in the repo, so it
stops living only in scratch scripts or session notes. Append new entries
at the top; keep the [[reference]]-style tags out, this file is meant to
be read directly, not cross-linked like Claude's own memory.

---

## 2026-08-24 — correction to the corrected-decoder-vs-backbones entry below: two real errors, caught by re-checking target steps

Publishing the entry below without checking `train/steps` against each run's own target-steps first (the project's own established discipline, per this log's `hardconstraint11-h0918` and `channel-backfill-coverage-ratio` entries — should have applied it here and didn't) produced two wrong numbers:

1. **h1622 emg_lap w=0.1 was reported as n=4, mean 918.3±58.6.** Two of those four seeds (seed2/seed3) were only 60%/58% done at the time (12.0M/11.6M of 20M steps) — their in-progress scores got read as final. Corrected: n=2 done (seed0 820.1, seed1 938.6, mean 879.4), seed2/3 still running, excluded until they finish.
2. **mpo/dep-mpo "3 seeds" per body were pooled into one mean±std despite each of the 3 runs having a different target-step budget** (5M/20M/25M for mpo; the same spread for dep-mpo, varies slightly by body) — not matched replicates of the same experiment. Fixed to report each run's own final score individually with its step count stated, not pooled.

Charts themselves were unaffected (the renderer's pooled band already truncates to the shortest seed's step range automatically) — only the prose numbers in the verdict/checks text were wrong. All three `corrected-decoder-vs-backbones-*` entries below have been edited in place to the corrected numbers, not left wrong with a patch note — the affected text is marked inline with "CORRECTED after publishing" so the fix itself stays visible.

**Lesson, stated plainly so it isn't repeated**: `train/steps` vs. target must be checked for every run before quoting a "final" score or pooling seeds, every time, not just when a number looks surprising — this was true of the checkpoint-sweep correction on 2026-08-12 too, and got missed again on the same axis less than two weeks later.

---

## 2026-08-24 — corrected-decoder emg_lap vs independent backbones (mpo/dep-mpo): mixed, not a clean win — plus old-decoder historical context

Park-paper-style "independent vs synergistic" comparison, requested directly: mpo+dep-mpo (no synergy prior, per-muscle action) as "independent," emg_lap k=6 (`decoder_k6_AB06_corrected`, post the 08-16/08-17 threshold/scale + z-domain fixes) as "synergistic." All onlyVelRew, w=0.1, noclip — same recipe as today's muscle-loosening baseline. Full per-body charts + checks now in `index.html` (3 new entries, `corrected-decoder-vs-backbones-{h0918,h1622,h2190}`).

**h0918**: three-way tie at the reward ceiling (emg_lap 999.7±0.1 n=2, mpo 997.5±2.1 n=3, dep-mpo 998.4±1.3 n=3) — uninformative, everything saturates here regardless of method.

**h1622 — the one body where emg_lap actually meets the 3-seed bar (n=4)**: emg_lap **loses** to both backbones — 918.3±58.6 vs mpo 972.3±20.7 vs dep-mpo 970.1±40.2, a real ~50-54pt gap, not noise-level. This is the most seed-solid cell in the whole comparison and it's a loss for the synergy prior, not a win. Doesn't contradict today's muscle-loosening reward gains (those are measured relative to emg_lap's own w=0.1 baseline, not to the backbones) — but it means untouched w=0.1 emg_lap was already behind mpo/dep-mpo before any loosening was tried. Worth stating plainly in any writeup: loosening recovers *some* of a gap that exists *because of* the tight EMG constraint, not a gap that would otherwise favor the constrained condition.

**h2190**: noise swamps the comparison (dep-mpo alone ranges 213.5-520.4 across 3 seeds) — no winner callable, consistent with the standing h2190-never-solves-onlyVelRew finding. Final-checkpoint only here, not peak — same caveat as the rest of this project's h2190 work.

**So what**: "the EMG prior helps" is not supportable as a blanket claim from reward alone — it ties on the easy body, loses on the one body with enough seeds to trust, and is unmeasurable on the hard body. Combined with today's separate finding that the tight-prior condition also produces non-physiological knee co-contraction despite decoder capacity being available (see AE-synergy/rollout work, same day), the honest picture is: the synergy prior costs reward on h1622 specifically, gives an uneven and partially-unrealized kinematic-fidelity benefit, and the loosening experiments are best framed as *partially recovering a self-inflicted cost*, not as improving on a working baseline.

**Historical context, requested directly — old (pre-fix) decoder, k=11, AB06 single-subject, same onlyVelRew w=0.1 recipe but clip=True (not noclip)**: h0918 942.4/986.8/986.1/986.8 (mean ~975, n=4), h1622 522.4/660.7/519.0 (mean ~567, n=3), h2190 106.2/208.6/76.0 (mean ~130, n=3). Corrected decoder is a large, unambiguous improvement over this on h1622 (567→918) and h2190 (130→304), smaller on h0918 (975→1000, already near ceiling). **Confound, not isolated**: old runs are clip=True, new runs are noclip — the project's own clip-ablation finding (`onlyvelrew-clip-ablation`, this log) already shows noclip alone raises score substantially, so this old-vs-new gap is decoder-fix *and* clip-removal combined, not decoder-fix alone. Not re-run at matched clip setting — would need a clip=True corrected-decoder run to isolate the decoder's own contribution, not done here. Report as "corrected decoder + noclip together are much better than old decoder + clip," not as "the decoder fix alone explains this."

---

## 2026-08-23 — AB06 kinematics comparison pipeline built from scratch; h0918 real-vs-untrained result is genuinely mixed, not a clean win either way

LEAPS's own rollout-generation scripts (`simulate_walking.py`, `render_comparison.py`, `render_emg.py`) are deleted in the current uncommitted working-tree diff. Used `deprl.play` instead (`depRL-sconegym/deprl/play.py`, unaffected by the LEAPS-side deletions, calls `environment.store_next_episode()`/`write_now()` via sconegym) — confirmed this is the real, tested mechanism the deleted scripts were built on top of, not a workaround.

Built a cycle-averaged, phase-registered comparison against the real AB06 EMG-replay reference (`early_tests/emg_replay/260707.100505.AB06/00000_564.355.sto` — only 0.61s/63 samples, too short to define its own gait cycle, driven through the h0918 model so it has **zero frontal-plane content** by construction — relevant if this is ever extended to h1622). Method: bounce-filtered right-heel-strike detection (0.7s minimum stride, caught mpo's cycle duration coming out at half the correct value from GRF-threshold noise before this filter), per-condition cycle-averaged template pooled across all available seeds, then a shared-phase-offset search maximizing mean Pearson r across all 7 sagittal joint angles jointly (since AB06's snippet can't be phase-normalized on its own).

**h0918, `w=0.1` (untrained unstable, only 3-4 strides available)**: real and untrained near-tied on shape correlation (r 0.525 vs 0.517) — flagged as inconclusive due to the sample-size gap, not reported as a finding.

**h0918, `w=0.5` (both conditions stable, 42-44 strides each — the fair comparison)**: real content has better RMSE (23.8° vs 29.3°, closer absolute joint angles) but **worse shape/timing correlation** than untrained (r 0.195 pooled / ~0.36 per-seed vs. untrained's 0.403 pooled / ~0.41 per-seed) — the "real content looks more human" impression from the `w=0.1` pass does not survive a fair test. Caught a pooling artifact before finalizing: real content's two seeds disagree on their individually-best phase offset more than untrained's do, so pooling drags the combined r down below either seed's own number (0.195 pooled vs. 0.354/0.367 per-seed) — per-seed numbers are the trustworthy ones, pooled numbers overstate the gap.

Both LAP-wrapper conditions (real *and* untrained) tracked AB06 better than the plain backbones at `w=0.1` (mean r ~0.49-0.53 vs ~0.18-0.23) — not yet re-checked at `w=0.5` with the corrected per-seed methodology, hold more loosely than the other numbers here.

**Mechanistic note, applies to every condition, not a per-condition finding**: left-side joints (`knee_angle_l`, `ankle_angle_l`) track worse than right-side almost everywhere. `mirror_left=True` copies the right leg's prior onto the left *synchronously* (no phase shift), while a real human's contralateral legs are ~50% out of phase — AB06's real bilateral data has that genuine offset, the model's mirrored left leg doesn't. Also: the single shared phase-offset can't simultaneously optimize all 7 joints, so per-joint r can look poor even where the underlying policy's gait isn't necessarily wrong — and Pearson r itself only measures shape correlation, not absolute overlap, so a joint can score a high r while looking visually disconnected (large constant offset) or vice versa. Don't read per-joint r as "how good does this look" without checking RMSE and the actual plot too.

**So what**: "real EMG content produces more human-like kinematics" is not a supportable claim from this data as it stands — it's a genuine split between shape-fidelity (untrained wins) and magnitude-fidelity (real wins), on an n=2-seed sample, using a comparison method with known limitations (short reference, single shared offset, synchronous mirroring). Report both metrics, not a single winner, if this goes in a writeup.

## 2026-08-23 — `loosened_actuators`/`loosened_residual_weight`: new per-actuator override, mechanism verified, outcome pending

Added to `LatentActionPriorWrapper` (`latent_env.py`): an optional per-actuator override on top of `mapped_residual_weight`, raises if used without `mapped_residual_weight` set or with `null_prior=True` (no per-actuator array to override in that case). Wired through the config generator properly (`RunSpec`/`builder.py`/`naming.py`) — routes to `final_experiments/other/` like any other non-default recipe, doesn't touch the main tree's naming. 5 new unit tests in `test_latent_env.py` (11/11 passing total), `test_config_gen.py` unaffected (18/20 passing, 2 pre-existing unrelated gaps).

First ablation: h1622 `emg_lap w=0.1`, `glut_med_r/l` given `loosened_residual_weight=0.5` (vs. the standard `w=0.1` every other mapped muscle uses) — testing whether an EMG-coverage asymmetry (`glut_med` mapped with real content, its antagonist `add_mag` completely unmapped) is what destabilizes plain h1622 (see the entry below for the full reasoning). Caught and corrected an initial framing error before launch: `add_mag` doesn't need loosening, it's already fully free at `residual_weight=1.0` since it has no EMG channel to constrain in the first place — a unit test caught this before any compute was spent.

**Mechanism verified against live training data, not just construction-time checks**: over a 14-18M step window, `glut_med_r`'s residual-correction magnitude is ~4.3x larger in the loosened run (0.315 vs 0.074, residual/prior ratio 1.26 vs 0.32) — the policy is genuinely using the new freedom. Every other mapped muscle's correction magnitude is unchanged between the two runs (0.072 vs 0.069) — the change is cleanly isolated, no leakage.

**Outcome: still pending as of this entry** (2 seeds, running). Comparison plan once done: must beat *both* `w=0.1` (883.9±113.6 last-20 mean) *and* global `w=0.5` (850.2±157.9 — already worse than w=0.1) to show this is about *where* the freedom goes, not just "more freedom helps in general."

**Prediction, not yet tested**: the same fix on h0918 (`hamstrings`/`rect_fem` vs. `iliopsoas`/`glut_max`) is expected to show no benefit or a small cost — h0918 has no lateral fall risk to react to, and h0918 already shows loose blends (`w=0.5` global) costing variance for no score gain there.

## 2026-08-23 — h1622 is a genuinely different, harder control problem than h0918 — not a coverage gap

Pulled `model.dofs()`/`model.muscles()` directly (not inferred). h0918: 9 DOFs, all sagittal (`pelvis_tilt/tx/ty`, `hip_flexion`, `knee_angle`, `ankle_angle` ×2 legs) — physically cannot fall sideways, no frontal/transverse motion exists at all. h1622: 19 DOFs — adds `pelvis_list`/`pelvis_rotation`/`pelvis_tz`, `hip_adduction`/`hip_rotation` ×2, and a full 3-DOF `lumbar_extension/bending/rotation` trunk joint h0918 doesn't have in any form.

The one muscle h1622 adds real EMG coverage for is `glut_med` (hip abductor — controls single-leg-stance pelvis level, i.e. prevents lateral fall). Its direct antagonist `add_mag` (hip adductor) has **zero** EMG channel — confirmed in `emg_mapping.py`'s own comment (`# missing: iliopsoas_r, glut_max_r, bifemsh_r, add_mag_r`). Both muscles still go through the same uniform `mapped_residual_weight=0.1` blend as the well-covered sagittal muscles.

Checked whether this asymmetric-coverage *shape* (mapped muscle vs. unmapped antagonist) is itself the problem, or specific to this pair: the same shape already exists in h0918 (`hamstrings`/`rect_fem` mapped vs. `iliopsoas`/`glut_max`/`bifemsh` unmapped) and h0918 works fine — so the mechanism (asymmetric constraint on a pair) isn't inherently destabilizing on its own. It's the only such pair that's *new* to h1622 *and* sits in the plane of motion actually causing the new failure mode. No dedicated agonist/antagonist pair exists for `hip_rotation` at all in this muscle set — that DOF is under-actuated by construction, a different and deeper gap than the coverage-asymmetry one, not fixable the same way.

**Correction, logged for the record**: initially concluded (wrongly) that the lumbar joint has zero stabilization at all (no muscle crosses it — verified via every muscle's `origin_body()`/`insertion_body()`, none touch `torso` — and `Joint.has_motor()=False`). A live rollout check with `Joint.limit_torque()` (all-zero policy action, isolating passive dynamics) showed substantial passive restoring torque (mean ~20/7/16 Nm across axes) keeping the trunk within ~±4.5° of upright — real passive stiffness, not free-swinging. The installed sconepy build is missing several documented methods (`Dof.muscle_moment/limit_torque/actuator_torque`, `Muscle.dof_moment_arm/joints`) that would have let this be checked directly the first time; structural inference (attachment + motor presence) isn't a full substitute for live dynamics.

**So what**: the coverage-asymmetry hypothesis on `glut_med`/`add_mag` is the best-targeted, best-evidenced lever found so far for h1622's instability — but it is not the only structural difference from h0918, and shouldn't be oversold as "the" explanation.

## 2026-08-23 — no_lap gap closed (was queued, never launched, since 08-20)

`no_lap` (null_prior, post-a_hat-fix) had zero live data for any body despite being in `queue.txt` since 2026-08-20 — the scheduler queue was simply behind it (`emg_lap`/`untrained_lap` fully occupying the 4 concurrency slots for the following ~3 days). Confirmed via `job_is_done()`/log.csv inspection, not assumption. `h0918` seed0/seed1 auto-launched once slots freed on 2026-08-23; `h1622`/`h2190` still pending behind it. This is the last cell needed to separate "does the LAP wrapper architecture itself cost/help anything" from "does EMG content quality matter" — currently the only condition with wrapper-present + zero content, distinct from both the unwrapped backbones and the (nonzero-garbage) `untrained_lap`.

**So what**: don't cite backbone-vs-emg_lap parity (h0918: ~997-999 vs ~994-998) as settled evidence the wrapper is "neutral" until `no_lap` lands — right now that parity is consistent with either "wrapper is neutral" or "wrapper costs X, content buys back X."

## 2026-08-19 — h2190 has never reached ~1000 under onlyVelRew, any method, in this project's history

Swept every `log.csv` under `h2190*` in the results tree (36 files —
early_tests, archived, final_experiments), pulled peak `test/episode_score`
(not just final checkpoint) per run. Filtered out `full`/`gaussianvel`
reward-variant runs (5000-8700 range — different reward scale, not
comparable to onlyVelRew's ~0-1000, not evidence of solving anything).

**Across every onlyVelRew run ever launched for h2190 — mpo, dep-mpo,
real-LAP, null-prior; k=6 or k=11; clip or noclip; any decoder era —
peak score never exceeds ~750, and every run degrades from its peak by
the final checkpoint.** Highest ever: 746.7 (real-LAP k11/AB06/clip,
collapsed to 106 by the end). h0918/h1622 reach 900-1000 and *hold* it;
h2190 does not, under any configuration tried so far.

**So what**: stop treating "does the prior reach ceiling" as the h2190
question — nothing does. The comparison with actual content there is
peak height / time-to-peak / post-peak stability across conditions, not
final-checkpoint score. When pulling h2190 results (including the ones
queued for the 2026-08-19 meeting deadline), report the full trajectory's
peak and when it occurred, not the last row of `log.csv` — final-checkpoint
alone is actively misleading for this body. See
`project_h2190_mechanistic_findings.md` (Claude memory) for the full
per-run table and the open question this raises about whether real-LAP
peaks higher / holds longer than the backbones here — not yet answered
with a controlled (same-recipe) comparison.

## 2026-08-19 — old-recipe decoders fully quarantined, enforced by a test

Every decoder except `decoder_k6_AB06_corrected` predates the 2026-08-16
threshold/scale fix and/or the 2026-08-17 z-domain fix. Verified nothing
in the active `final_experiments/` tree resolved to anything else, then
moved all 38 old decoders (33 single_subject + 5 pooled_subjects) into
`results/synergy_priors/deprecated_precorrected_decoder/` — kept for
comparison, not for use. Also found and archived 4 never-launched configs
(`no_lap`/`untrained_lap` × h1622/h2190, `_legacy_` suffix) still pointing
at the old pooled `decoder_k6`, into
`LEAPS/baselines_DEPRL/archived_from_final_experiments_20260818/deprecated_precorrected_decoder/`.

**This is enforced at the test level now, not just a one-time file move**:
`tests/test_config_gen.py::test_deprecated_precorrected_decoders_not_resolved_by_final_experiments`
asserts `_decoder_paths()` raises `FileNotFoundError` for any
`final_experiments` spec requesting a non-`AB06_corrected` decoder source —
so this can't silently regress if someone adds a new config with the wrong
`decoder_source` later. Every already-launched config's compat symlink
(`results/synergy/`) was re-pointed to the new nested location and
verified with zero broken links; the in-flight h1622 jobs were unaffected
throughout.

## 2026-08-19 — null-prior isn't a blend at any `w`, mechanism verified in code

`null_prior=True` zeroes the entire `EMGToMuscleMapper` weight matrix
(`emg_mapping.py:163-174`), so `a_hat ≡ 0` for every actuator, not just
mapped ones. The blend formula still runs
(`final_action = (1-w)*a_hat + w*a_full`, `latent_env.py:341`), but
`train/latent_residual_share = |residual| / (|residual| + |a_hat| + eps)`
has its denominator's prior term identically zero whenever `a_hat≡0` — so
the share is **mathematically forced to ~1.0 regardless of the configured
`w`**. Verified directly: null-prior shows `latent_residual_share=1.0`
exactly at both w=0.1 and w=0.5.

Untrained-prior (random decoder weights, not zeroed) has a real nonzero
`a_hat` and genuinely respects the configured blend: `residual_share=0.465`
at w=0.5 (≈ the configured 0.5), vs 0.205 at w=0.0 (floor from always-free
unmapped actuators, not a bug).

**So what**: null-prior and untrained-prior at "the same w" are not
comparable — null is 100% free policy dressed in the LAP wrapper at every
w, untrained is a genuine blend against garbage content. Null's rising
score with w (861 at w=0.5, h0918) isn't residual-weight sensitivity, it's
just "free policy can walk." Cite untrained-prior's w-sweep for the
blend-ratio story, not null's.

**New anomaly, not yet explained**: null-prior at w=0.5 (effectively 100%
free policy) still scores 861 vs the true unwrapped mpo/dep-mpo backbones'
996-998 — a ~130+pt gap that must come from the LAP wrapper's action/obs
reparameterization itself, since there's no prior content at all in this
condition. Open question.

## 2026-08-19 — does real EMG still matter at w=0.5? Yes, but the metric is compressed there

Real-LAP static w=0.5 (h0918): ~999.6-1000 (last 5 test evals all exactly
1000.0). Untrained-prior w=0.5: 935. Gap ≈ 65pts — genuine, and a fair
comparison (real's `latent_residual_share`=0.584 at w=0.5, close to
untrained's 0.465, both near the configured 0.5 — both are real blends,
unlike null above).

Gap across w: 969pts at w=0.0 (999.9 vs 31.2), 950pts at w=0.1 (995.5 vs
~37-47), 65pts at w=0.5. **So what**: the free residual at high w gives
enough budget to compensate for garbage content and reach near-ceiling for
everyone — the metric saturates, it doesn't stop discriminating entirely.
Don't cite w=0.5 alone as "real vs untrained" evidence; w=0.0/0.1 are the
discriminating regime, w=0.5 is confirmatory at best.

## 2026-08-19 — decoder depth: paper prose says 1 hidden layer, actual reference code uses 2

Checked `latent-action-priors` (the actual LAP paper's repo) directly, not
just the paper PDF. `nonLinearAE` in `expert_demonstrations/utils.py` —
the exact class whose `.decoder.state_dict()` is loaded live by
`wrappers.py:56-64`'s `ProjectActions` at RL-training time — has **two**
Tanh hidden layers of width `2×dim_latent` each, not one. The paper's own
prose ("one hidden layer, size 2×dim(a_l), tanh") doesn't match its own
shipped code. Our production decoder (`AB06_corrected`) uses `depth=1`,
matching the *prose*, not the *code*.

**Checked empirically whether this matters, using our own existing
`results/backfill/depth_threshold_margin_grid.csv` sweep** (176 runs, both
depths, 3 seeds each): at our exact locked recipe (threshold=0.6,
scale=0.9), depth=1 gives R²=0.9410±0.0007 vs depth=2's 0.9314±0.0185 —
**depth=1 wins on both mean and variance** (27x tighter). Holds in
aggregate across the whole grid too (0.9379±0.0045 vs 0.9348±0.0070).

**So what**: staying at depth=1 isn't just matching the paper's prose
instead of its code — it's empirically the better choice at our own
hyperparameters, already tested, no further action needed. Worth an
explicit "we replicate the paper's stated architecture, which happens to
also outperform its own reference implementation's actual code at our
locked hyperparameters" line in the writeup.

## 2026-08-19 — one-cycle training collapse: reproduced internally, not just cross-paper

Reference paper trains its decoder from ~22 frames (one gait cycle) and
reports (their `results.yaml`) train-cycle MSE 4.0e-7 vs full-episode MSE
3.97e-2 — a ~99,000x gap. Cross-repo comparisons like that are confounded
(different domain/units, and their "full episode" tests extrapolation to
non-cyclic startup dynamics, not just more of the same distribution).

**We already have the internally-controlled version of this same test**:
`results/backfill/k6ab06_num_strides_r2.csv` (filename corrected 2026-08-19
— cited as `decoder_stride_sweep_k6.csv` earlier the same session, which
does not exist under that name on disk; flagging the discrepancy rather
than silently fixing it, cause unconfirmed), our own k=6 recipe, varying
strides-per-mode:

| strides/mode | held-out R² |
|---|---|
| 1 (≈one cycle/mode) | **−0.85** (worse than predicting the mean) |
| 5 | −0.01 |
| 25 | 0.53 |
| 100 | 0.88 |
| 250 | 0.91 |
| ~397 (full, current recipe) | **0.94** |

**So what**: this is the citable version of "single-gait-type training
overfits catastrophically" — same architecture, same k, only data volume
varies, R² collapses below zero at reference-paper-scale data and recovers
to 0.94 at our actual scale. Cite this curve over the cross-paper Humanoid
comparison; it isolates the actual variable.

## 2026-08-19 — decoder.pt has a results.yaml now, matching the reference repo's own convention

`train_single_subject_decoder.py` now writes `results.yaml`
(fit/held-out MSE/MAE/R², sample counts, hyperparameters) alongside every
`decoder.pt`/`norm.npz` it produces — backfilled for the currently-locked
`decoder_k6_AB06_corrected` (see
`results/synergy_priors/single_subject/decoder_k6_AB06_corrected/results.yaml`).
Verified the backfill run reproduces the exact production decoder weights
bit-for-bit before trusting any number from it (fully deterministic given
the same seed/data/hyperparameters).

## 2026-08-19 — z-rollout shows real periodic structure (partial evidence, old config)

`results/backfill/z_rollout_net512_AB06_seed0.npy` (10,000-step raw
per-timestep latent-z, all 6 dims) — autocorrelation check shows a clear
periodicity peak at lag≈50 steps (~1.25s at step_size=0.025, a plausible
gait-cycle duration) across every dim, autocorr 0.29-0.52. Real evidence
the decoder's output is rhythmic, not noise. **Caveat**: net512/old-recipe
config (predates current net256 lock), real decoder only — no
untrained-decoder rollout to contrast against yet. Would need a fresh
rollout under the current locked recipe, both conditions, to fully close
this out.

## 2026-08-19 — synergy_priors reorg

`results/synergy/` (39 flat `decoder_*` dirs, no organization) reorganized
into `results/synergy_priors/{single_subject,pooled_subjects,small_pooled}/`,
mirroring the reference repo's self-contained per-artifact folder style.
Old flat paths kept as symlinks (every already-launched config's baked-in
`decoder_path` still resolves). `builder.py`/`spec.py` updated so future
`final_experiments` configs write directly to the new structure;
`early_tests` unchanged (golden-file test fidelity). `small_pooled/` is
empty + a README — reserved for a "target subject + 4 closest morphology
matches" decoder (named after `small_pooled_5` in
`cross_subject_check_ab08.py`), never actually built for AB06/h0918 itself.

## 2026-08-19 — final_experiments naming convention fixed

`leaps/configs/{naming,spec,writer}.py`: within `final_experiments`,
`clip`/`net{size}` tokens are now omitted from run names when they match
the population's only actual recipe (noclip, net256 — verified across all
55 current configs), and the now-vestigial `"static"` mirror-mode tag is
dropped entirely (phase-mirroring removed from the codebase 2026-08-18,
nothing left to disambiguate). A non-default recipe (net512, clip=True)
now physically routes to `final_experiments/other/...` instead of just
being differently named. `early_tests/` untouched (golden-file tests).
No existing directory was renamed — only future-generated configs get the
clean names; a retroactive rename pass needs a separate go-ahead once
nothing is actively training.

## 2026-08-19 — phase-mirroring ablation dropped

`LatentActionPriorWrapper.__init__` no longer accepts `mirror_mode` at all
(removed 2026-08-18, static-only now). Found 4 queued phase configs
(h1622/h2190 × w01/w05) that would have crashed on launch — pulled from
`run_scripts/queue.txt`. Not worth fixing/relaunching: h0918's existing
phase result (w01=717.6 worse than null, w05=996.9 recovers to parity)
didn't surface anything phase uniquely explains beyond "needs more
residual budget," same story static already tells.

## 2026-08-27 — velocity-conditioned prior: no task benefit on h0918, trained or held-out speeds

vcond experiment (per-episode `target_vel ~ U[0.8,1.5]`, symmetric
`gaussianVel` reward, speed in obs). **Closed 2026-08-27, null — all jobs
killed, queue block retired.** Arms: `vcond_real` (speed-conditioned EMG
decoder) / `vcond_null` (a_hat=0) / `uncond_real` (plain treadmill-pool
decoder) / `mpo_baseline` (no wrapper).

**Trained-range (deterministic `test/episode_score`), final:** seed0 of
the three main arms finished at 10M — vcond_real **970.0**, mpo_baseline
**972.6**, vcond_null **973.9** (4-point spread). Killed partials all past
convergence: vcond_null_s1 974.2 (6M), uncond_real_s0 973.7 (5.6M),
uncond_real_s1 929.0 (5.4M, noisy). vcond_real_s1 / mpo_baseline_s1 died
at 200k. Every run past ~2M sits 966–974. `uncond_real` (the "conditioning
vs static prior" contrast) is 973.7 — on top of everything. No separation,
no sample-efficiency gap.

**Held-out-speed eval** (`step_6000000`, 20 eps/speed, obs speed feature
kept training-normalized; `leaps.scripts.eval_vcond_heldout_speed`):

| target | band | mpo_baseline | vcond_null | vcond_real |
|---|---|---|---|---|
| 0.6 | extrap↓ | 907 · vel 0.81 | 827 · vel 1.01 | 873 · vel 0.93 |
| 0.9 | interp | 977 · err .016 | 961 · err .14 | 970 · err .10 |
| 1.1 | interp | 975 · err .02 | 980 · err .04 | 985 · err .06 |
| 1.3 | interp | 971 · err .09 | 978 · err .04 | 987 · err .01 |
| 1.6 | extrap↑ | 929 · vel 1.36 | 935 · vel 1.39 | 939 · vel 1.38 |
| 1.8 | extrap↑ | 814 · vel 1.36 | 834 · vel 1.39 | 827 · vel 1.38 |

Interpolation: all three equal (~975). Extrapolation: all three fail
identically — top speed saturates at ~1.37 m/s regardless of prior, so
1.6/1.8 undershoot the same; 0.6 missed by all. The EMG prior does not buy
generalization.

**Why the baseline saturates:** "mpo_baseline" is DEP-MPO (`dep_factory` +
`TunedMPO` + `AdaptiveEnergyBuffer`) — already a strong domain exploration
prior. Reward is trivially saturable (vel_coeff=1, every other coeff 0;
no effort/style/smoothness term), so coordinated activation is not
rewarded. h0918 + a 1-parameter gait family is easy. No struggle → no room
for the inductive bias.

**What the prior still does:** `vcond_real` activations stay smooth and
low-amplitude (mean ~0.07, never saturate) vs `vcond_null` / `mpo_baseline`
bang-bang (mean ~0.35, max 1.0). Physiological-plausibility difference is
real and holds; it just doesn't show up in return.

**Verdict:** kill-switch met on both metrics; experiment retired (not
extended to h1622/h2190 — that's a new experiment against the spine plan,
3 weeks from deadline). Kept as a negative finding supporting the reframe:
LAP's own suggested extension, tested properly, does not separate from null
on muscle-actuated control. Paper stands on results 1–3. Data in
`data.json` key `vcond_h0918`; raw in `results/vcond_heldout/`; harness
`leaps.scripts.eval_vcond_heldout_speed`. Findings-log chart panel TODO
(low priority — it's a null).
