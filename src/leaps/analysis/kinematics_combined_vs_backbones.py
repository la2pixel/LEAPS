"""One combined figure, Schumacher Fig.1 layout: rows = bodies, columns =
hip/knee/ankle/GRF. Unlike Schumacher (1 line per panel, since they never
compare methods), each panel here has 3 lines: best prior condition, MPO,
DEP-MPO -- all best-seed-selected the same way, so it's a fair, credible
3-way comparison in the same footprint as their single-method figure.

    python -m leaps.analysis.kinematics_combined_vs_backbones

Output -> LEAPS/results/biomech_fidelity/KINEMATICS_COMBINED_VS_BACKBONES.png
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
# Fixed to w=0.1 only ("ours" everywhere else in the thesis) -- searching
# across all 3 weights AND all seeds gave "prior" 6-8 candidates to pick a
# max from vs. MPO/DEP-MPO's 2-3, an unfair selection-bias advantage
# (more candidates to max over inflates the winner regardless of true
# quality). Fixed here so all three methods search the same scope: best
# seed within one, pre-decided condition. Flagged 2026-09-15.
REAL_PRIOR_LADDER = ["real-LAP"]

INK = "#e8e8e8"
INK_MUTED = "#9aa0a6"
BAND_FILL = "#c9cdd1"
BAND_LINE = "#e8e8e8"
COLOR = {"prior": "#e64545", "mpo": "#6fa8dc", "dep-mpo": "#f2c14e"}
LABEL = {"prior": "prior (w=0.1)", "mpo": "MPO", "dep-mpo": "DEP-MPO"}
WIN_GREEN = "#3fb950"
TIE_ORANGE = "#d99a3f"
TIE_TOL = 0.03  # match_fraction difference within this counts as a tie -- disclosed, fixed threshold


def best_seed_pooled(body: str, cond_list: list[str], band: dict) -> tuple[dict | None, str | None, float]:
    """Best (condition, seed) by match_overall among cond_list -- same
    disclosed, single-criterion selection used throughout."""
    tmap = load_tstar_map()
    best_pooled, best_label, best_score = None, None, -1.0
    for cond in cond_list:
        for seed_dir in RUNS[body].get(cond, []):
            p = pool_sim([seed_dir], tmap)
            if p is None:
                continue
            s = np.mean([match_fraction(p["ang"][j], band[f"{j}_mean"], band[f"{j}_sd"])[0] for j in JOINTS])
            if s > best_score:
                best_pooled, best_label, best_score = p, seed_dir, s
    return best_pooled, best_label, best_score


def main() -> None:
    band = build_human_band(1.35, 0.25)
    pct = np.linspace(0, 100, 101)
    bodies = ["h0918", "h1622", "h2190"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True)
    fig.patch.set_alpha(0)

    for bi, body in enumerate(bodies):
        series = {}
        for key, conds in [("prior", REAL_PRIOR_LADDER), ("mpo", ["mpo"]), ("dep-mpo", ["dep-mpo"])]:
            pooled, seed, score = best_seed_pooled(body, conds, band)
            series[key] = pooled
            print(f"{body} {key}: best = {seed}, match_overall={score:.2f}")

        for ci, j in enumerate(COLS):
            ax = axes[bi, ci]
            ax.set_facecolor("none")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_color(INK_MUTED)
            ax.spines["left"].set_color(INK_MUTED)
            ax.tick_params(colors=INK_MUTED, labelsize=10)

            # compute scores first so the win-shading can be drawn behind everything
            score_lines = []
            for key in ["prior", "mpo", "dep-mpo"]:
                pooled = series[key]
                if pooled is None:
                    score_lines.append((key, None))
                    continue
                if j == "grf":
                    sc = match_fraction(pooled["grf"], band["grf_mean"], band["grf_sd"])[0]
                else:
                    y = pooled["ang"][j]
                    if ALIGN_OFFSET:
                        y = y - (y - band[f"{j}_mean"]).mean()
                    sc = match_fraction(y, band[f"{j}_mean"], band[f"{j}_sd"])[0]
                score_lines.append((key, sc))
            valid = dict(score_lines)
            valid = {k: v for k, v in valid.items() if v is not None}
            if valid:
                best_backbone = max([v for k, v in valid.items() if k != "prior"], default=None)
                prior_sc = valid.get("prior")
                if prior_sc is not None and best_backbone is not None:
                    if prior_sc > best_backbone + TIE_TOL:
                        ax.axvspan(0, 100, color=WIN_GREEN, alpha=0.10, zorder=0)
                    elif abs(prior_sc - best_backbone) <= TIE_TOL:
                        ax.axvspan(0, 100, color=TIE_ORANGE, alpha=0.10, zorder=0)

            if j == "grf":
                ax.fill_between(pct, band["grf_mean"] - band["grf_sd"], band["grf_mean"] + band["grf_sd"],
                                 color=BAND_FILL, alpha=0.22)
                ax.plot(pct, band["grf_mean"], color=BAND_LINE, lw=1)
            else:
                ax.fill_between(pct, band[f"{j}_mean"] - band[f"{j}_sd"], band[f"{j}_mean"] + band[f"{j}_sd"],
                                 color=BAND_FILL, alpha=0.22)
                ax.plot(pct, band[f"{j}_mean"], color=BAND_LINE, lw=1)

            for key in ["prior", "mpo", "dep-mpo"]:
                pooled = series[key]
                if pooled is None:
                    continue
                if j == "grf":
                    y = pooled["grf"]
                else:
                    y = pooled["ang"][j]
                    if ALIGN_OFFSET:
                        y = y - (y - band[f"{j}_mean"]).mean()
                ax.plot(pct, y, color=COLOR[key], lw=2.2, label=LABEL[key])

            if bi == 0:
                ax.set_title(COL_TITLES[j], color=INK, fontsize=16)
            if ci == 0:
                ax.set_ylabel(body.upper(), color=INK, fontsize=15)
            if bi == 2:
                ax.set_xlabel("gait cycle [%]", color=INK, fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=3, frameon=False, fontsize=13, labelcolor=INK)
    fig.suptitle(
        "Prior (w=0.1, fixed a priori) vs. MPO vs. DEP-MPO — real human band in gray\n"
        "green = prior wins by >0.03; orange = tie (within 0.03 of best backbone)",
        color=INK, fontsize=14, y=1.11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "KINEMATICS_COMBINED_VS_BACKBONES.png"
    fig.savefig(p, dpi=150, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
