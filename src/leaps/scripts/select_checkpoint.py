"""Checkpoint-selection rule for the final RL tables (locked 2026-09-08).

Rule, applied identically to every method/body/seed:
  1. From log.csv take test/episode_score/mean, sampled every ~200k steps.
  2. Candidate checkpoints = the saved ones: multiples of SAVE_STEP (2M) in
     [2M, B], where B = the common horizon (min over seeds of last logged
     step, capped at the nominal budget).
  3. Score each candidate by the centred moving average of the test curve
     over a window of W=5 evals (~+/-0.4M) around that step.
  4. t*  = argmax candidate.  Each seed picks its own t*.
  5. Report value = smoothed score at t*  (provisional; a fresh N=100 eval at
     t* is the number that goes in the paper -- see --reeval, not done here).
  6. Aggregate = mean +/- sd over seeds, individual seeds kept.
  Secondary column "final10" = mean of the last 10 evals, for transparency
  (shows the h2190 decline; nothing hidden).

Also writes a <label>.csv of (seed, t_star) that biomech_fidelity.py / the
gait scorecard consume, so fidelity and return refer to the same policy.

Usage:
  python select_checkpoint.py RUNDIR [RUNDIR ...] --label emg_lap_h1622_w01_onlyvelrew
each RUNDIR contains one <timestamp>/log.csv.
"""
import argparse
import csv
import glob
import os
import statistics
import sys

SAVE_STEP = 2_000_000
NOMINAL_BUDGET = 20_000_000
WINDOW = 5                 # evals, centred
STEP_COL = "train/steps"
SCORE_COL = "test/episode_score/mean"


def load_curve(rundir):
    hits = glob.glob(os.path.join(rundir, "**", "log.csv"), recursive=True)
    if not hits:
        return None
    path = sorted(hits, key=os.path.getmtime)[-1]
    rows = list(csv.reader(open(path)))
    hdr = {name: i for i, name in enumerate(rows[0])}
    si, ci = hdr[STEP_COL], hdr[SCORE_COL]
    steps, scores = [], []
    for r in rows[1:]:
        try:
            s, v = float(r[si]), float(r[ci])
        except (ValueError, IndexError):
            continue
        steps.append(s)
        scores.append(v)
    return path, steps, scores


def smoothed_at(steps, scores, target, window):
    """Centred MA of `scores` over the `window` evals nearest `target` step."""
    order = sorted(range(len(steps)), key=lambda i: abs(steps[i] - target))
    picked = sorted(order[:window])
    return statistics.mean(scores[i] for i in picked)


def _centred_ma(xs, w):
    out = []
    for i in range(len(xs)):
        lo, hi = max(0, i - w // 2), min(len(xs), i + w // 2 + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def _plot(label, runs, cands, window, rows_out, star_agg, fin_agg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_seed = {r["seed"]: r for r in rows_out}
    n = len(runs)
    ncol = 2 if n > 1 else 1
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 2.9 * nrow),
                             squeeze=False)
    for ax, r in zip(axes.flat, runs):
        st = [s / 1e6 for s in r["steps"]]
        sc = r["scores"]
        sm = _centred_ma(sc, window)
        info = by_seed[r["seed"]]
        tstar_m = info["t_star"] / 1e6
        ax.plot(st, sc, color="0.7", lw=0.9, label="raw test score")
        ax.plot(st, sm, color="#1f77b4", lw=2.0, label=f"centred MA (W={window})")
        cand_m = [c / 1e6 for c in cands]
        cand_v = [smoothed_at(r["steps"], sc, c, window) for c in cands]
        ax.plot(cand_m, cand_v, "o", ms=4, color="#1f77b4", alpha=.5,
                label="candidates (2M grid)")
        # final-10 window
        w0 = st[-10] if len(st) >= 10 else st[0]
        ax.axvspan(w0, st[-1], color="#ff7f0e", alpha=.12)
        ax.axhline(info["final10"], color="#ff7f0e", ls="--", lw=1.2,
                   label=f"final10 = {info['final10']:.0f}")
        # t*
        ax.axvline(tstar_m, color="#d62728", ls=":", lw=1.4)
        ax.plot([tstar_m], [info["score_at_tstar"]], "*", ms=16,
                color="#d62728", label=f"t* = {tstar_m:.0f}M  ({info['score_at_tstar']:.0f})")
        ax.set_title(f"{r['seed']}", fontsize=10)
        ax.set_xlabel("steps (M)")
        ax.set_ylabel("test/episode_score")
        ax.legend(fontsize=6.5, loc="lower right")
        ax.grid(alpha=.25)
    for ax in axes.flat[n:]:
        ax.axis("off")
    ms, ss = star_agg
    mf, sf = fin_agg
    fig.suptitle(f"{label}\nrule: {ms:.0f} ± {ss:.0f}   |   final10: {mf:.0f} ± {sf:.0f}"
                 f"   (score@t* is the provisional proxy; real number = N=100 re-eval at t*)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundirs", nargs="+")
    ap.add_argument("--label", required=True)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--budget", type=int, default=None,
                    help="override common horizon B (steps)")
    ap.add_argument("--outdir",
                    default="LEAPS/results/checkpoint_selection")
    ap.add_argument("--plot", metavar="PATH", default=None,
                    help="write a per-seed diagnostic figure here")
    args = ap.parse_args()

    runs = []
    for d in args.rundirs:
        got = load_curve(d)
        if got is None:
            print(f"!! no log.csv under {d}", file=sys.stderr)
            continue
        path, steps, scores = got
        seed = next((tok for tok in os.path.basename(d.rstrip("/")).split("_")
                     if tok.startswith("seed")), os.path.basename(d.rstrip("/")))
        runs.append(dict(seed=seed, dir=d, path=path, steps=steps, scores=scores))
    if not runs:
        sys.exit("no runs loaded")

    horizon = args.budget or min(NOMINAL_BUDGET, min(r["steps"][-1] for r in runs))
    cands = list(range(SAVE_STEP, int(horizon) + 1, SAVE_STEP))
    print(f"\n=== {args.label} ===")
    print(f"seeds: {len(runs)}   common horizon B = {horizon/1e6:.1f}M   "
          f"candidates: {cands[0]/1e6:.0f}..{cands[-1]/1e6:.0f}M step 2M   window: {args.window} evals\n")

    hdr = f"{'seed':>6} {'t* (M)':>7} {'score@t*':>9} {'final10':>8} {'raw tail5':>10} {'peak-final':>10}"
    print(hdr)
    print("-" * len(hdr))

    rows_out = []
    star_scores, final_scores = [], []
    for r in sorted(runs, key=lambda x: x["seed"]):
        sm = {c: smoothed_at(r["steps"], r["scores"], c, args.window) for c in cands}
        tstar = max(cands, key=lambda c: sm[c])
        score_star = sm[tstar]
        final10 = statistics.mean(r["scores"][-10:])
        tail5 = statistics.mean(r["scores"][-5:])
        star_scores.append(score_star)
        final_scores.append(final10)
        rows_out.append(dict(label=args.label, seed=r["seed"], t_star=tstar,
                             score_at_tstar=round(score_star, 1),
                             final10=round(final10, 1),
                             run_dir=r["dir"]))
        print(f"{r['seed']:>6} {tstar/1e6:>7.0f} {score_star:>9.1f} "
              f"{final10:>8.1f} {tail5:>10.1f} {score_star - final10:>10.1f}")

    def ms(xs):
        return (statistics.mean(xs),
                statistics.pstdev(xs) if len(xs) > 1 else 0.0)
    m_star, s_star = ms(star_scores)
    m_fin, s_fin = ms(final_scores)
    print("-" * len(hdr))
    print(f"{'MEAN':>6} {'':>7} {m_star:>9.1f} {m_fin:>8.1f}")
    print(f"{'SD':>6} {'':>7} {s_star:>9.1f} {s_fin:>8.1f}")
    print(f"\nreport:  {m_star:.1f} +/- {s_star:.1f}   (secondary 'final': {m_fin:.1f} +/- {s_fin:.1f})")

    if args.plot:
        _plot(args.label, sorted(runs, key=lambda x: x["seed"]), cands,
              args.window, rows_out, (m_star, s_star), (m_fin, s_fin), args.plot)
        print(f"wrote {args.plot}")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.label}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
