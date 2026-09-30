"""EMG-activation RMSE-ratio (Park et al. 2026 F5 convention), applied to the
per-muscle channels instead of joint kinematics -- companion check for F2
(biomech_fidelity.py), which reports Pearson r (shape/timing) only.

Same definition as rmse_ratio.py (their joint-kinematics F5): RMSE between
simulated and experimental RVC-normalized activation, divided by the mean
pairwise RMSE among the 22 Camargo subjects. Ratio ~1.0 = sim error is about
as large as normal human-to-human variability; the *generous* end of their
two-metric evaluation.

Why this script exists: Pearson r is invariant to a constant offset/scale
mismatch, RMSE is not. This project already caught the two metrics
disagreeing once (kinematics, h0918 w=0.5 -- real EMG content beats untrained
on RMSE, 23.8 vs 29.3 deg, but LOSES on r; findings_log/narrative.md). Same
risk applies to F2's muscle-timing numbers and hasn't been checked directly.

    python -m leaps.analysis.emg_rmse_ratio --body h1622 --tstar \\
        --conditions mpo,dep-mpo,real-LAP --caption-order
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations

import numpy as np

from leaps.scripts.biomech_fidelity import (
    CAPTION_ORDER, CKPT_PREF, MIN_ROWS, OUT, PARK_ABBR, PARK_ORDER, RUNS,
    RUNS_FULL, cross_human, load_tstar_map, pool_condition,
)
from leaps.envs.emg_mapping import MODEL_MAPS


def cross_human_rmse(sub_means: dict, channels: list[str], ch_idx: dict) -> dict[str, float]:
    """Per channel: mean RMSE over all subject pairs of RVC-normalized waveforms."""
    subs = list(sub_means)
    out = {}
    for c in channels:
        i = ch_idx[c]
        vals = [np.sqrt(np.mean((sub_means[a][:, i] - sub_means[b][:, i]) ** 2))
                for a, b in combinations(subs, 2)]
        out[c] = float(np.mean(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="h1622", choices=["h0918", "h1622", "h2190"])
    ap.add_argument("--regime", default="onlyvel", choices=["onlyvel", "full"])
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--tstar", action="store_true")
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--caption-order", action="store_true")
    a = ap.parse_args()

    speed = a.speed if a.speed is not None else (1.2 if a.regime == "onlyvel" else 1.3)
    tol = a.tol if a.tol is not None else (0.15 if a.regime == "onlyvel" else 0.25)
    body = a.body

    runs = RUNS if a.regime == "onlyvel" else RUNS_FULL
    ckpt_pref = load_tstar_map() if a.tstar else CKPT_PREF.get(body)
    cmap = MODEL_MAPS[body]
    order = CAPTION_ORDER if a.caption_order else PARK_ORDER
    channels = [c for c in order if c in cmap]

    chn, sub_means, ch_idx, pair_r, envelope = cross_human(speed, tol)
    hmean = {c: envelope[c][0] for c in channels}
    baseline = cross_human_rmse(sub_means, channels, ch_idx)

    conds = a.conditions.split(",") if a.conditions else list(runs[body])
    unknown = [cd for cd in conds if cd not in runs[body]]
    if unknown:
        raise SystemExit(f"unknown condition(s) for {body}: {unknown}; available: {list(runs[body])}")

    rows = []
    waves = {}
    for cd in conds:
        model_muscles = sorted({cmap[c] for c in channels})
        w, n_ep, n_cyc, n_skip = pool_condition(runs[body][cd], model_muscles,
                                                min_rows=MIN_ROWS.get(body, 2500),
                                                prefer_ckpt=ckpt_pref)
        waves[cd] = w

    print(f"{body} {a.regime}  cross-human RMSE baseline (RVC units):")
    for c in channels:
        print(f"  {PARK_ABBR[c]:6s} {baseline[c]:.3f}")
    print()
    print(f"{'muscle':8s}" + "".join(f"{cd + ' r':>12s}{cd + ' rmse_ratio':>16s}" for cd in conds))
    for c in channels:
        i = ch_idx[c]
        mm = cmap[c]
        line = f"{PARK_ABBR[c]:8s}"
        for cd in conds:
            wv = waves[cd].get(mm)
            if wv is None:
                line += f"{'--':>12s}{'--':>16s}"
                continue
            from scipy import stats
            r = float(stats.pearsonr(wv, hmean[c])[0])
            rmse = float(np.sqrt(np.mean((wv - hmean[c]) ** 2)))
            ratio = rmse / baseline[c]
            line += f"{r:>12.3f}{ratio:>16.3f}"
            rows.append(dict(body=body, muscle=PARK_ABBR[c], model_muscle=mm,
                             condition=cd, r=r, rmse=rmse,
                             rmse_baseline=baseline[c], rmse_ratio=ratio))
        print(line)

    if rows:
        OUT.mkdir(parents=True, exist_ok=True)
        suffix = ("" if a.regime == "onlyvel" else "_full") + ("_tstar" if a.tstar else "")
        suffix += "_" + "-".join(conds)
        if a.caption_order:
            suffix += "_captionorder"
        csvp = OUT / f"f2_rmse_{body}{suffix}.csv"
        with open(csvp, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\nwrote {csvp}")


if __name__ == "__main__":
    main()
