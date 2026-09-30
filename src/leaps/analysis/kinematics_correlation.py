"""Kinematics/kinetics agreement as Pearson r, mirroring the muscle-timing analysis: mean r of the
simulated mean cycle against each reference subject's mean cycle, judged against the pairwise
correlations between the reference subjects (5th percentile). Same runs, t* checkpoints and
cycle pooling as kinematic_match.py; human strides loaded as in kinematic_match.build_human_band.

    python -m leaps.analysis.kinematics_correlation
Output: results/biomech_fidelity/kinematics_correlation.csv
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from leaps.data import load_mat_table, find_stride_intervals, time_normalize_stride
from leaps.data.metadata import SUBJECTS
from leaps.scripts import kinematic_match as km
from leaps.scripts.biomech_fidelity import RUNS, MIN_ROWS, load_tstar_map
from leaps.data.metadata import CAMARGO_DATA_ROOT

SPEED, TOL, FS = 1.35, 0.25, 1000.0
QUANT = [f"{j}_angle" for j in km.JOINTS] + [f"{j}_moment" for j in km.JOINTS] + ["grf"]


def subject_means():
    out = {}
    for s in km.SUBJ:
        base = glob.glob(f"{CAMARGO_DATA_ROOT}/{s}/*/treadmill")[0]
        mass = SUBJECTS[s].mass
        acc = {q: [] for q in QUANT}
        for gon_f in sorted(glob.glob(f"{base}/gon/*.mat")):
            stem = Path(gon_f).name
            try:
                g, cnd = load_mat_table(gon_f), load_mat_table(f"{base}/conditions/{stem}")
                gc, fp = load_mat_table(f"{base}/gcRight/{stem}"), load_mat_table(f"{base}/fp/{stem}")
                idd = load_mat_table(f"{base}/id/{stem}")
            except Exception:
                continue
            for t0, t1 in find_stride_intervals(gc["Header"], gc["HeelStrike"]):
                sp = cnd["Speed"][(cnd["Header"] >= t0) & (cnd["Header"] < t1)]
                if len(sp) == 0 or abs(float(np.mean(sp)) - SPEED) > TOL:
                    continue
                m = (g["Header"] >= t0) & (g["Header"] < t1)
                if m.sum() < 20:
                    continue
                tn = lambda x: time_normalize_stride(km._lowpass(x, FS)[:, None])[:, 0]
                for j in km.JOINTS:
                    acc[f"{j}_angle"].append(tn(g[km.GON[j]][m]))
                mi = (idd["Header"] >= t0) & (idd["Header"] < t1)
                if mi.sum() >= 20:
                    for j in km.JOINTS:
                        if km.MOMENT_COL[j] in idd:
                            acc[f"{j}_moment"].append(tn(idd[km.MOMENT_COL[j]][mi] / mass))
                mf = (fp["Header"] >= t0) & (fp["Header"] < t1)
                acc["grf"].append(tn(fp["Treadmill_R_vy"][mf] / (mass * 9.81)))
        out[s] = {q: np.mean(v, axis=0) for q, v in acc.items() if v}
    return out


def main():
    hum = subject_means()
    subs = list(hum)
    tmap = load_tstar_map()
    conds = dict(km.METHODS, untrained="untrained")
    rows = []
    for body in ("h0918", "h1622", "h2190"):
        for cond, label in conds.items():
            if cond not in RUNS[body]:
                continue
            sd = km.pool_sim(RUNS[body][cond], tmap, min_rows=MIN_ROWS.get(body, 2500))
            if sd is None:
                continue
            sim = {f"{j}_angle": sd["ang"][j] for j in km.JOINTS}
            sim.update({f"{j}_moment": sd["mom"][j] for j in km.JOINTS})
            sim["grf"] = sd["grf"]
            for q in QUANT:
                if sim[q] is None or any(q not in hum[s] for s in subs):
                    continue
                rr = np.array([stats.pearsonr(sim[q], hum[s][q])[0] for s in subs])
                pairs = [stats.pearsonr(hum[a][q], hum[b][q])[0]
                         for i, a in enumerate(subs) for b in subs[i + 1:]]
                rows.append(dict(body=body, variant=label, quantity=q, r_mean=rr.mean(), r_std=rr.std(),
                                 human_r_mean=np.mean(pairs), human_p5=np.percentile(pairs, 5),
                                 in_human_band=rr.mean() >= np.percentile(pairs, 5)))
    df = pd.DataFrame(rows)
    df.to_csv(km.OUT / "kinematics_correlation.csv", index=False)
    pv = df.pivot_table(index=["body", "variant"], columns="quantity", values="r_mean")[QUANT]
    print(pv.round(2).to_string())
    print("\nhuman reference (mean pairwise r / 5th pct):")
    print(df.groupby("quantity")[["human_r_mean", "human_p5"]].first().loc[QUANT].round(2).to_string())


if __name__ == "__main__":
    main()
