"""One compact kinematics figure per body: single row (hip/knee/ankle/GRF),
showing only the best-scoring real-prior condition (by match_overall, best
seed) -- for a 15-minute talk where a 3-row grid is too dense to read live.

Same data/selection logic as kinematics_schumacher_style.py, just condensed
to the one row worth putting on a slide.

    python -m leaps.analysis.kinematics_best_per_body

Output -> LEAPS/results/biomech_fidelity/BEST_KINEMATICS_<body>.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leaps.scripts.biomech_fidelity import RUNS, load_tstar_map
from leaps.scripts.kinematic_match import (
    JOINTS, ALIGN_OFFSET, build_human_band, pool_sim, match_fraction,
)
from leaps.paths import RESULTS_DIR

OUT = RESULTS_DIR / "biomech_fidelity"
COLS = JOINTS + ["grf"]
COL_TITLES = {"hip": "hip", "knee": "knee", "ankle": "ankle", "grf": "vertical GRF (BW)"}
REAL_PRIOR_LADDER = {
    "real-LAP-w00": "only prior (w=0.0)",
    "real-LAP": "prior + 0.1",
    "real-LAP-w05": "prior + 0.5 residual",
}

INK = "#e8e8e8"
INK_MUTED = "#9aa0a6"
BAND_FILL = "#c9cdd1"
BAND_LINE = "#e8e8e8"
RED = "#e64545"


def main() -> None:
    band = build_human_band(1.35, 0.25)
    tmap = load_tstar_map()
    pct = np.linspace(0, 100, 101)

    for body in ["h0918", "h1622", "h2190"]:
        # best (condition, seed) pair for this body, by match_overall
        best = dict(score=-1.0, label=None, pooled=None, seed=None)
        for cond, ml in REAL_PRIOR_LADDER.items():
            if cond not in RUNS[body]:
                continue
            for seed_dir in RUNS[body][cond]:
                p = pool_sim([seed_dir], tmap)
                if p is None:
                    continue
                s = np.mean([match_fraction(p["ang"][j], band[f"{j}_mean"], band[f"{j}_sd"])[0]
                             for j in JOINTS])
                if s > best["score"]:
                    best.update(score=s, label=ml, pooled=p, seed=seed_dir)

        print(f"{body}: best = {best['label']} ({best['seed']}), match_overall={best['score']:.2f}")

        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        fig.patch.set_alpha(0)
        for ci, j in enumerate(COLS):
            ax = axes[ci]
            ax.set_facecolor("none")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_color(INK_MUTED)
            ax.spines["left"].set_color(INK_MUTED)
            ax.tick_params(colors=INK_MUTED, labelsize=11)

            if j == "grf":
                ax.fill_between(pct, band["grf_mean"] - band["grf_sd"], band["grf_mean"] + band["grf_sd"],
                                 color=BAND_FILL, alpha=0.22)
                ax.plot(pct, band["grf_mean"], color=BAND_LINE, lw=1)
                y, sc = best["pooled"]["grf"], match_fraction(best["pooled"]["grf"], band["grf_mean"], band["grf_sd"])[0]
            else:
                ax.fill_between(pct, band[f"{j}_mean"] - band[f"{j}_sd"], band[f"{j}_mean"] + band[f"{j}_sd"],
                                 color=BAND_FILL, alpha=0.22)
                ax.plot(pct, band[f"{j}_mean"], color=BAND_LINE, lw=1)
                y = best["pooled"]["ang"][j]
                if ALIGN_OFFSET:
                    y = y - (y - band[f"{j}_mean"]).mean()
                sc = match_fraction(y, band[f"{j}_mean"], band[f"{j}_sd"])[0]
            ax.plot(pct, y, color=RED, lw=2.6)
            ax.set_title(f"{COL_TITLES[j]}   ({sc:.2f})", color=INK, fontsize=16)
            ax.set_xlabel("gait cycle [%]", color=INK, fontsize=12)

        fig.suptitle(f"{body.upper()} — best prior condition: {best['label']}   "
                     "(gray = human ±1SD, red = policy)",
                     color=INK, fontsize=17, y=1.04)
        fig.tight_layout()
        p = OUT / f"BEST_KINEMATICS_{body}.png"
        fig.savefig(p, dpi=150, transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
