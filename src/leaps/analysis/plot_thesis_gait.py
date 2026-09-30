"""Thesis gait figures in the white badge style of KINETICS_h0918_prior_w05.png: one row per
condition, one column per joint, human mean +-1 SD in grey, experimental match in a badge (gold =
best of the rows for that joint). All seeds pooled, same loader and band as kinematic_match.py, so
the badges equal the KINEMATIC_MATCH table. Curves are shifted by their mean offset, as E_y is.

    python plot_thesis_gait.py --body h0918 --quantity angle --conds mpo dep-mpo real-LAP-w05
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from leaps.scripts.biomech_fidelity import RUNS, MIN_ROWS, load_tstar_map
from leaps.scripts.kinematic_match import JOINTS, build_human_band, match_fraction, pool_sim
from leaps.paths import FIGURES_DIR

OUT = str(FIGURES_DIR / "thesis")
LABEL = {"mpo": "MPO", "dep-mpo": "DEP-MPO", "null": "Null prior", "untrained": "Untrained prior",
         "real-LAP-w00": "EMG prior, $w=0$", "real-LAP": "EMG prior, $w=0.1$",
         "real-LAP-w05": "EMG prior, $w=0.5$"}
COLOR = {"mpo": "#1f77b4", "dep-mpo": "#9085e9", "null": "#c98500", "untrained": "#d55181",
         "real-LAP-w00": "#56b4e9", "real-LAP": "#d95926", "real-LAP-w05": "#e64545"}
# quantity -> (pool_sim key, band key suffix, column title unit)
QUANT = {"angle": ("ang", "", "angle (deg)"), "moment": ("mom", "_moment", "moment (Nm/kg)")}
BAND_FILL, BAND_LINE, GOLD, GREY = "#cfd4da", "#6b6f75", "#f2c14e", "#e6e6e6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="h0918")
    ap.add_argument("--quantity", default="angle", choices=QUANT)
    ap.add_argument("--conds", nargs="+", default=["mpo", "dep-mpo", "real-LAP-w05"])
    ap.add_argument("--sem", action="store_true",
                    help="shade +-1 standard error of the mean (SD/sqrt(cycles)) instead of +-1 SD, "
                         "for every condition")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--defense", action="store_true",
                    help="reproduce the 09-18 defense figures: 20M rollouts, best seed per condition "
                         "by mean hip/knee/ankle match")
    a = ap.parse_args()
    key, suf, unit = QUANT[a.quantity]

    band, tmap, pct = build_human_band(1.35, 0.25), load_tstar_map(), np.linspace(0, 100, 101)
    mr = MIN_ROWS.get(a.body, 2500)
    if a.defense:
        tmap = {rd: 20_000_000 for c in a.conds for rd in RUNS[a.body][c]}

        def best_seed(c):
            ps = [(rd, pool_sim([rd], tmap, mr)) for rd in RUNS[a.body][c]]
            return max(((rd, p) for rd, p in ps if p is not None), key=lambda x: np.mean(
                [match_fraction(x[1]["ang"][j], band[f"{j}_mean"], band[f"{j}_sd"])[0] for j in JOINTS]))

        picked = {c: best_seed(c) for c in a.conds}
        pooled = {c: p for c, (rd, p) in picked.items()}
        for c, (rd, p) in picked.items():
            print(f"{c}: best seed {rd.split('_')[-1]}")
    else:
        pooled = {c: pool_sim(RUNS[a.body][c], tmap, mr) for c in a.conds}
    # columns: the three joints, plus vertical GRF (in body weights) on the angle figure
    cols = JOINTS + (["grf"] if a.quantity == "angle" else [])

    def curve(p, j):
        if j == "grf":
            return p["grf"], p["grf_sd"], band["grf_mean"], band["grf_sd"]
        return p[key][j], p[f"{key}_sd"][j], band[f"{j}{suf}_mean"], band[f"{j}{suf}_sd"]

    score = {(c, j): match_fraction(*[curve(pooled[c], j)[i] for i in (0, 2, 3)])[0]
             for c in a.conds for j in cols}
    best = {j: max(a.conds, key=lambda c: score[c, j]) for j in cols}

    fig, axes = plt.subplots(len(a.conds), len(cols), figsize=(4 * len(cols), 2.9 * len(a.conds)), sharex=True, sharey="col",
                             squeeze=False)
    for ri, c in enumerate(a.conds):
        p = pooled[c]
        for ci, j in enumerate(cols):
            ax = axes[ri, ci]
            m, sd, hm, hs = curve(p, j)
            if a.sem:
                sd = sd / np.sqrt(p["n_cyc"])
            m = m - (m - hm).mean()
            ax.fill_between(pct, hm - hs, hm + hs, color=BAND_FILL, alpha=0.7, lw=0,
                            label="human $\\pm1$ SD")
            ax.plot(pct, hm, color=BAND_LINE, lw=1)
            ax.fill_between(pct, m - sd, m + sd, color=COLOR[c], alpha=0.2, lw=0)
            ax.plot(pct, m, color=COLOR[c], lw=2.2,
                    label=LABEL[c] + (" ($\\pm1$ SEM)" if a.sem else ""))
            ax.text(0.96, 0.93, f"match = {score[c, j]:.2f}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=12, fontweight="bold" if best[j] == c else "normal",
                    bbox=dict(boxstyle="round,pad=0.3", fc=GOLD if best[j] == c else GREY, ec="none"))
            ax.spines[["top", "right"]].set_visible(False)
            if ri == 0:
                ax.set_title("vertical GRF (BW)" if j == "grf" else f"{j} {unit}", fontsize=12)
            if ri == len(a.conds) - 1:
                ax.set_xlabel("gait cycle [%]")
            if ci == 0:
                ax.legend(frameon=False, loc="lower left", fontsize=9)
        print(c, {j: round(score[c, j], 2) for j in cols}, "episodes:", p["n_ep"])
    fig.tight_layout()
    name = f"{a.body}_{a.quantity}_{'_'.join(a.conds)}" + ("_defense" if a.defense else "") + ("_sem" if a.sem else "")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"{name}.{ext}"), dpi=300, facecolor="white", bbox_inches="tight")
    print("saved", name)


if __name__ == "__main__":
    main()
