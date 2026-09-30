"""Schumacher et al. 2023 figure convention: gray human band, red RL policy,
one condition per row -- so which prior variant tracks the human band best
is visually obvious without color-decoding a multi-line overlay.

Real prior weight ladder only (w=0.0/0.1/0.5) -- null/untrained are content-
necessity controls for a different question, not candidates for "our best
configuration," so they're excluded here (see conversation 2026-09-15).

Each subplot shows its match_fraction score (band-membership fraction, 0-1,
best near 1.0 -- same metric kinematic_match.py's match_overall is built
from). Within each column (joint), the best-scoring row is highlighted gold.

One figure per body: rows = w=0.0/0.1/0.5, columns = hip/knee/ankle/GRF.

    python -m leaps.analysis.kinematics_schumacher_style

Output -> LEAPS/results/biomech_fidelity/SCHUMACHER_STYLE_<body>.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leaps.scripts.biomech_fidelity import RUNS, load_tstar_map, MIN_ROWS
from leaps.scripts.kinematic_match import (
    JOINTS, ALIGN_OFFSET, build_human_band, pool_sim, match_fraction,
)
from leaps.paths import RESULTS_DIR

OUT = RESULTS_DIR / "biomech_fidelity"
COLS = JOINTS + ["grf"]
COL_TITLES = {"hip": "hip", "knee": "knee", "ankle": "ankle", "grf": "vertical GRF (BW)"}

BACKBONE_LADDER = {
    "mpo": "MPO",
    "dep-mpo": "DEP-MPO",
}
REAL_PRIOR_LADDER = {
    "real-LAP-w00": "only prior (w=0.0)",
    "real-LAP": "prior + 0.1",
    "real-LAP-w05": "prior + 0.5 residual",
}
# added 2026-09-18 on request: the ladder-only version never compared against
# the unconstrained backbones at all -- added here, own colors so they don't
# get confused with the gold "best in column" annotation.
ROW_COLOR = {
    "mpo": "#6fa8dc", "dep-mpo": "#b35806",
    "real-LAP-w00": "#e64545", "real-LAP": "#e64545", "real-LAP-w05": "#e64545",
}

INK = "#ffffff"
INK_MUTED = "#ffffff"
BAND_FILL = "#c9cdd1"
BAND_LINE = "#e8e8e8"
RED = "#e64545"
GOLD = "#f2c14e"


def main() -> None:
    band = build_human_band(1.35, 0.25)
    tmap = load_tstar_map()
    pct = np.linspace(0, 100, 101)

    for body in ["h0918", "h1622", "h2190"]:
        rows = [(cond, ml) for cond, ml in {**BACKBONE_LADDER, **REAL_PRIOR_LADDER}.items()
                if cond in RUNS[body]]
        n = len(rows)
        fig, axes = plt.subplots(n, 4, figsize=(16, 3.1 * n), sharex=True)
        fig.patch.set_alpha(0)
        if n == 1:
            axes = axes.reshape(1, 4)

        # Best checkpoint (= best seed, since each seed typically has one
        # saved checkpoint) per condition -- Schumacher/Park's own field
        # convention (verified 2026-09-15: both papers select the single
        # best-performing checkpoint pooled across seeds, not a per-seed
        # average). Selection criterion: highest match_overall (mean of
        # hip/knee/ankle match_fraction) -- disclosed, single criterion,
        # applied identically to every condition.
        pooled_by_row = []
        for cond, _ in rows:
            best_pooled, best_score, best_seed = None, -1.0, None
            for seed_dir in RUNS[body][cond]:
                p = pool_sim([seed_dir], tmap, min_rows=MIN_ROWS.get(body, 2500))
                if p is None:
                    continue
                s = np.mean([match_fraction(p["ang"][j], band[f"{j}_mean"], band[f"{j}_sd"])[0]
                             for j in JOINTS])
                if s > best_score:
                    best_pooled, best_score, best_seed = p, s, seed_dir
            pooled_by_row.append(best_pooled)
            print(f"  {body} {cond}: best seed = {best_seed} (match_overall={best_score:.2f})")

        # score[ri][ci] = match_fraction for that (condition, joint/grf) cell
        scores = np.full((n, len(COLS)), np.nan)
        for ri, pooled in enumerate(pooled_by_row):
            if pooled is None:
                continue
            for ci, j in enumerate(COLS):
                if j == "grf":
                    scores[ri, ci] = match_fraction(pooled["grf"], band["grf_mean"], band["grf_sd"])[0]
                else:
                    scores[ri, ci] = match_fraction(pooled["ang"][j], band[f"{j}_mean"], band[f"{j}_sd"])[0]
        best_row_per_col = np.nanargmax(scores, axis=0)

        for ri, (cond, ml) in enumerate(rows):
            pooled = pooled_by_row[ri]
            row_color = ROW_COLOR.get(cond, RED)

            for ci, j in enumerate(COLS):
                ax = axes[ri, ci]
                ax.set_facecolor("none")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.spines["bottom"].set_visible(True)
                ax.spines["left"].set_visible(True)
                ax.spines["bottom"].set_color(INK_MUTED)
                ax.spines["left"].set_color(INK_MUTED)
                ax.spines["bottom"].set_linewidth(1.4)
                ax.spines["left"].set_linewidth(1.4)
                ax.tick_params(colors=INK_MUTED, labelsize=14, width=1.4)

                if j == "grf":
                    ax.fill_between(pct, band["grf_mean"] - band["grf_sd"],
                                     band["grf_mean"] + band["grf_sd"],
                                     color=BAND_FILL, alpha=0.22)
                    ax.plot(pct, band["grf_mean"], color=BAND_LINE, lw=1)
                    if pooled is not None:
                        y, ysd = pooled["grf"], pooled["grf_sd"]
                        ax.fill_between(pct, y - ysd, y + ysd, color=row_color, alpha=0.20, lw=0)
                        ax.plot(pct, y, color=row_color, lw=2.2)
                else:
                    ax.fill_between(pct, band[f"{j}_mean"] - band[f"{j}_sd"],
                                     band[f"{j}_mean"] + band[f"{j}_sd"],
                                     color=BAND_FILL, alpha=0.22)
                    ax.plot(pct, band[f"{j}_mean"], color=BAND_LINE, lw=1)
                    if pooled is not None:
                        y, ysd = pooled["ang"][j], pooled["ang_sd"][j]
                        if ALIGN_OFFSET:
                            y = y - (y - band[f"{j}_mean"]).mean()
                        ax.fill_between(pct, y - ysd, y + ysd, color=row_color, alpha=0.20, lw=0)
                        ax.plot(pct, y, color=row_color, lw=2.2)

                sc = scores[ri, ci]
                if not np.isnan(sc):
                    is_best = ri == best_row_per_col[ci]
                    ax.text(0.97, 0.94, f"{sc:.2f}", transform=ax.transAxes,
                             ha="right", va="top",
                             fontsize=22 if is_best else 19,
                             fontweight="bold" if is_best else "normal",
                             color=GOLD if is_best else INK)

                if ri == 0:
                    ax.set_title(COL_TITLES[j], color=INK, fontsize=22, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(ml, color=INK, fontsize=18, fontweight="bold")
                if ri == n - 1:
                    ax.set_xlabel("gait cycle [%]", color=INK, fontsize=16)

        fig.suptitle(f"{body.upper()} vs. human band  (best seed per condition; "
                     "gray = human ± 1SD, blue/brown = MPO/DEP-MPO, red = prior; "
                     "number = match_fraction, best per column in gold)",
                     color=INK, fontsize=14, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = OUT / f"SCHUMACHER_STYLE_{body}.png"
        fig.savefig(p, dpi=150, transparent=True)
        plt.close(fig)
        print(f"wrote {p}")

        print(f"\n  match_fraction table -- {body}")
        print("  " + f"{'condition':16s}" + "".join(f"{c.upper():>8s}" for c in COLS))
        for ri, (cond, ml) in enumerate(rows):
            print("  " + f"{ml:16s}" + "".join(f"{scores[ri, ci]:8.2f}" for ci in range(len(COLS))))


if __name__ == "__main__":
    main()
