# Checkpoint-selection rule for the RL return results (defense / thesis)

One rule, applied identically to every method × body × seed. Implemented in
`leaps.scripts.select_checkpoint` (Phase 1) + `leaps.scripts.reeval_checkpoint`
(Phase 2), consolidated by `leaps.scripts.return_table`. Locked 2026-09-08.

## The rule

1. **Evaluation signal.** During training, every ~2·10^5 steps the policy is
   rolled out deterministically (mean action, no exploration noise) and the
   undiscounted episode return `test/episode_score/mean` is logged. This is
   the only quantity the rule looks at — separate from the stochastic
   training return, as in TD3 (Fujimoto et al. 2018) and SAC (Haarnoja et
   al. 2018).
2. **Candidate set.** The checkpoints actually saved to disk: every 2·10^6
   steps in `[2M, B]`, where `B` = min over the seeds of the last logged
   step, capped at the 20M budget. Same grid for every arm.
3. **Score of a candidate.** The centred moving average of the evaluation
   curve over a window of **W = 5 evals (≈ ±0.4M steps)** around that
   checkpoint. Smoothing, not the single best eval, so a one-point spike
   cannot be selected — the standard early-stopping practice (Prechelt
   1998, "Early Stopping — But When?").
4. **t\* = arg max** over candidates, **chosen per seed independently.**
5. **Reported number (Phase 2).** A fresh **N = 100**-episode deterministic
   re-evaluation at each seed's `step_<t*>.pt`. This — not the smoothed
   training-curve value — is what goes in the table.
6. **Aggregation.** Mean ± SD across seeds (≥ 3 where available), individual
   seeds always shown as points (Henderson et al. 2018; Colas et al. 2019).
7. **Transparency column.** Every table also reports `final10` = mean of the
   last 10 evaluations (end-of-training performance under the fixed budget)
   and the `t* − final10` gap. Nothing about late collapse is hidden
   (Machado et al. 2018).
8. **Robustness column.** `fall%` at t\* = fraction of the 100 re-eval
   episodes that terminate early (the model falls). Reported alongside
   return so a high mean built on frequent falls is visible.

## Why this rule, and what it is *not*

* **Not "best seed / best checkpoint ever seen."** That is the failure mode
  Henderson et al. (2018) and Agarwal et al. (2021, *Statistical
  Precipice*) call out: a max over a noisy set is an upward-biased point
  estimate, especially with few seeds. Here the max is taken only over a
  **smoothed** curve and only over **saved** checkpoints, per seed, and the
  end-of-training number is reported next to it.
* **Not "final checkpoint only."** Machado et al. (2018) recommend
  end-of-budget performance to remove selection bias — appropriate when
  training is monotone. It is not for h1622/h2190 here (see below), so we
  report *both* t\* and `final10` and let the gap carry the instability
  finding rather than pretending a degraded final policy is the result.
* **Reduces to the baselines' convention when training is stable.** LAP
  (Hausdörfer et al. 2024, arXiv:2410.03246) and the muscle-walking MPO
  paper (arXiv:2309.02976, same H0918/H1622/H2190 models) report converged
  performance averaged over seeds. On h0918 our t\* sits on the plateau and
  `t* ≈ final10` for every method — the rule is a no-op there and matches
  them. It only diverges from "just take the end" where the training curve
  itself is non-monotone.

## How the three bodies behave under it (the general trend)

| body | what the curves do | t\* vs final10 | reading |
|---|---|---|---|
| **h0918** (small, over-constrained) | every method saturates the return ceiling by ~4–8M and stays there | `t* ≈ final10` (gap < 15) for all 4 methods | rule is a no-op; prior = MPO = DEP-MPO = null on return |
| **h1622** (mid) | mostly converged, occasional transient drop-and-recover; EMG-prior seeds fall more often | small gap for backbones, larger for EMG prior + null (transient dips land in the last-10 window) | t\* selection matters here; still no prior advantage on return |
| **h2190** (large, most redundant) | **no method converges** — breakthrough to ~350–570 then decline; ~75–91% of episodes fall even at t\* | large gap (`final10` ≈ half of `score@t*`) for every method | report peak (t\*) + fall% + the drop; the instability *is* the result |

**Trend:** as the musculoskeletal model grows, (i) achievable return falls
for all methods, (ii) training stability collapses for all methods, and
(iii) the EMG action prior neither closes nor materially widens the
method-to-method gap on return at any body. Return is not where the prior
contributes — its effect is on per-muscle activation-timing fidelity (F2 /
R2), which the return table is not designed to show.

Figure: `results/checkpoint_selection/RETURN_TABLE.png` (per-seed points,
grouped by body) and `results/checkpoint_selection/return_trend.png`
(cross-body trend, 4 method lines).

## References

- Henderson, Islam, Bachman, Pineau, Precup, Meger (2018). *Deep
  Reinforcement Learning that Matters.* AAAI.
- Machado, Bellemare, Talvitie, Veness, Hausknecht, Bowling (2018).
  *Revisiting the Arcade Learning Environment.* JAIR 61.
- Agarwal, Schwarzer, Castro, Courville, Bellemare (2021). *Deep
  Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS.
- Colas, Sigaud, Oudeyer (2019). *A Hitchhiker's Guide to Statistical
  Comparisons of RL Algorithms.* (and Colas et al. 2018, *How Many Random
  Seeds?*)
- Prechelt (1998). *Early Stopping — But When?* In *Neural Networks: Tricks
  of the Trade.*
- Fujimoto, van Hoof, Meger (2018). *Addressing Function Approximation Error
  in Actor-Critic Methods* (TD3). Haarnoja et al. (2018). *Soft
  Actor-Critic.* — deterministic periodic evaluation.
- Hausdörfer, von Rohr, Lefort, Schoellig (2024). *Latent Action Priors for
  Locomotion with Deep Reinforcement Learning.* arXiv:2410.03246. — baseline.
- *Natural and Robust Walking using RL without Demonstrations in
  High-Dimensional Musculoskeletal Models.* arXiv:2309.02976. — MPO recipe
  and the H0918/H1622/H2190 models.
