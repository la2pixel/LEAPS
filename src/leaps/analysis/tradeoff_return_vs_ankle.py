"""Return vs. ankle-kinematics tradeoff across residual weight (w).

Real numbers only: return = mean reeval_mean across seeds (official t*-reeval
CSVs); ankle match = best-seed match_fraction (same disclosed, single-
criterion selection used throughout). One subplot per body, dual y-axis
(return left, ankle match_fraction right), x = w in {0.0, 0.1, 0.5}.

    python -m leaps.analysis.tradeoff_return_vs_ankle

Output -> LEAPS/results/biomech_fidelity/TRADEOFF_RETURN_VS_ANKLE.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leaps.scripts.biomech_fidelity import RUNS, load_tstar_map
from leaps.scripts.kinematic_match import build_human_band, pool_sim, match_fraction
from leaps.paths import RESULTS_DIR

OUT = RESULTS_DIR / "biomech_fidelity"
CKDIR = RESULTS_DIR / "checkpoint_selection"
W_COND = {0.0: "real-LAP-w00", 0.1: "real-LAP", 0.5: "real-LAP-w05"}
W_REEVAL = {0.0: "w00", 0.1: "w01", 0.5: "w05"}

INK = "#e8e8e8"
INK_MUTED = "#9aa0a6"
RETURN_C = "#6fa8dc"
ANKLE_C = "#e64545"


def get_return(body: str, w: float) -> float | None:
    f = CKDIR / f"emg_lap_{body}_{W_REEVAL[w]}_onlyvelrew_reeval.csv"
    if not f.exists():
        return None
    vals = [float(r["reeval_mean"]) for r in csv.DictReader(open(f))]
    return float(np.mean(vals)) if vals else None


def get_ankle_match(body: str, w: float, band: dict) -> float | None:
    tmap = load_tstar_map()
    best = -1.0
    for seed_dir in RUNS[body].get(W_COND[w], []):
        p = pool_sim([seed_dir], tmap)
        if p is None:
            continue
        y = p["ang"]["ankle"]
        y = y - (y - band["ankle_mean"]).mean()
        best = max(best, match_fraction(y, band["ankle_mean"], band["ankle_sd"])[0])
    return best if best >= 0 else None


def main() -> None:
    band = build_human_band(1.35, 0.25)
    bodies = ["h0918", "h1622", "h2190"]
    ws = [0.0, 0.1, 0.5]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.patch.set_alpha(0)

    for bi, body in enumerate(bodies):
        rets = [get_return(body, w) for w in ws]
        ankles = [get_ankle_match(body, w, band) for w in ws]

        axL = axes[bi]
        axL.set_facecolor("none")
        for spine in axL.spines.values():
            spine.set_visible(False)
        axL.spines["bottom"].set_visible(True)
        axL.spines["left"].set_visible(True)
        axL.spines["bottom"].set_color(RETURN_C)
        axL.spines["left"].set_color(RETURN_C)
        axL.tick_params(axis="y", colors=RETURN_C, labelsize=11)
        axL.tick_params(axis="x", colors=INK_MUTED, labelsize=12)
        axL.plot(ws, rets, color=RETURN_C, marker="o", markersize=9, lw=2.4, label="return")
        axL.set_xticks(ws)
        axL.set_xticklabels([f"w={w}" for w in ws])
        axL.set_xlabel("residual weight", color=INK, fontsize=13)
        if bi == 0:
            axL.set_ylabel("mean return (t*-reeval)", color=RETURN_C, fontsize=13)

        axR = axL.twinx()
        axR.set_facecolor("none")
        for spine in axR.spines.values():
            spine.set_visible(False)
        axR.spines["right"].set_visible(True)
        axR.spines["right"].set_color(ANKLE_C)
        axR.tick_params(axis="y", colors=ANKLE_C, labelsize=11)
        axR.plot(ws, ankles, color=ANKLE_C, marker="s", markersize=9, lw=2.4, label="ankle match")
        axR.set_ylim(0, 0.75)
        if bi == 2:
            axR.set_ylabel("ankle match_fraction", color=ANKLE_C, fontsize=13)

        axL.set_title(body.upper(), color=INK, fontsize=17, pad=14)

    handles = [plt.Line2D([0], [0], color=RETURN_C, marker="o", lw=2.4, label="return (left axis)"),
               plt.Line2D([0], [0], color=ANKLE_C, marker="s", lw=2.4, label="ankle match_fraction (right axis)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=2, frameon=False, fontsize=13, labelcolor=INK)
    fig.suptitle("Residual weight trades return against ankle kinematic fidelity",
                 color=INK, fontsize=15, y=1.14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "TRADEOFF_RETURN_VS_ANKLE.png"
    fig.savefig(p, dpi=150, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")
    for bi, body in enumerate(bodies):
        print(f"{body}: return={[get_return(body,w) for w in ws]}  ankle={[get_ankle_match(body,w,band) for w in ws]}")


if __name__ == "__main__":
    main()
