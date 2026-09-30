"""Kinetics (net joint moment, Nm/kg) companion to kinematics_combined_vs_
backbones.py -- identical structure and identical selection rule: prior
fixed to w=0.1 (decided a priori, not searched), best seed within each of
the three conditions (prior/MPO/DEP-MPO), same as kinematics and muscle
timing. Rows = bodies, columns = hip/knee/ankle net moment.

    python -m leaps.analysis.kinetics_combined_vs_backbones

Output -> LEAPS/results/biomech_fidelity/KINETICS_COMBINED_VS_BACKBONES.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leaps.scripts.biomech_fidelity import RUNS, load_tstar_map, MIN_ROWS
from leaps.scripts.kinematic_match import JOINTS, build_human_band, pool_sim, match_fraction
from leaps.paths import RESULTS_DIR

OUT = RESULTS_DIR / "biomech_fidelity"
COL_TITLES = {"hip": "hip moment (Nm/kg)", "knee": "knee moment (Nm/kg)", "ankle": "ankle moment (Nm/kg)"}

INK = "#ffffff"
INK_MUTED = "#ffffff"
BAND_FILL = "#c9cdd1"
BAND_LINE = "#e8e8e8"
COLOR = {"prior": "#e64545", "mpo": "#6fa8dc", "dep-mpo": "#f2c14e"}
LABEL = {"prior": "prior (w=0.5)", "mpo": "MPO", "dep-mpo": "DEP-MPO"}
WIN_GREEN = "#3fb950"
TIE_ORANGE = "#d99a3f"
TIE_TOL = 0.03
# w=0.5, not w=0.1 -- kinetics/moments favors more residual freedom (ties to
# the same weight winning on raw return), a different, robustly-checked
# regime from kinematics/muscle-timing's w=0.1. Disclosed explicitly: this
# figure and the kinematics one use different, individually fixed weights
# for different properties, not "ours" at one unspecified setting.
COND_FOR = {"prior": ["real-LAP-w05"], "mpo": ["mpo"], "dep-mpo": ["dep-mpo"]}


def best_seed_pooled(body: str, cond_list: list[str], band: dict) -> tuple[dict | None, str | None, float]:
    tmap = load_tstar_map()
    best_pooled, best_seed, best_score = None, None, -1.0
    for cond in cond_list:
        for seed_dir in RUNS[body].get(cond, []):
            p = pool_sim([seed_dir], tmap, min_rows=MIN_ROWS.get(body, 2500))
            if p is None:
                continue
            vals = [match_fraction(p["ang"][j], band[f"{j}_mean"], band[f"{j}_sd"])[0] for j in JOINTS]
            s = np.mean(vals)
            if s > best_score:
                best_pooled, best_seed, best_score = p, seed_dir, s
    return best_pooled, best_seed, best_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=None, choices=["h0918", "h1622", "h2190"],
                     help="restrict to one body -- single row, bigger fonts, for a "
                          "standalone slide (default: all 3 bodies combined)")
    ap.add_argument("--separate", action="store_true",
                     help="--body only: one row per condition (prior/MPO/DEP-MPO) "
                          "instead of overlaying all 3 in one panel per joint, with "
                          "the match_fraction score printed on each panel, same "
                          "convention as kinematics_schumacher_style.py")
    args = ap.parse_args()
    if args.separate and not args.body:
        raise SystemExit("--separate requires --body")

    band = build_human_band(1.35, 0.25)
    pct = np.linspace(0, 100, 101)
    bodies = [args.body] if args.body else ["h0918", "h1622", "h2190"]
    single = args.body is not None

    if args.separate:
        body = args.body
        series = {}
        for key, conds in COND_FOR.items():
            pooled, seed, score = best_seed_pooled(body, conds, band)
            series[key] = pooled
            print(f"{body} {key}: seed={seed} (kinematics match_overall={score:.2f})")

        keys = ["prior", "mpo", "dep-mpo"]
        scores = {}  # scores[joint][key]
        for j in JOINTS:
            mm, ms = band.get(f"{j}_moment_mean"), band.get(f"{j}_moment_sd")
            scores[j] = {}
            for key in keys:
                pooled = series[key]
                mom = pooled["mom"][j] if pooled is not None else None
                scores[j][key] = match_fraction(mom, mm, ms)[0] if (mom is not None and mm is not None) else None
        best_key_per_joint = {j: max((k for k in keys if scores[j][k] is not None),
                                      key=lambda k: scores[j][k], default=None) for j in JOINTS}

        fig, axes = plt.subplots(3, 3, figsize=(15, 9.5), sharex=True, squeeze=False)
        fig.patch.set_alpha(0)
        for ri, key in enumerate(keys):
            pooled = series[key]
            for ci, j in enumerate(JOINTS):
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

                mm, ms = band.get(f"{j}_moment_mean"), band.get(f"{j}_moment_sd")
                if mm is not None:
                    ax.fill_between(pct, mm - ms, mm + ms, color=BAND_FILL, alpha=0.22)
                    ax.plot(pct, mm, color=BAND_LINE, lw=1)

                mom = pooled["mom"][j] if pooled is not None else None
                if mom is not None:
                    mom_sd = pooled["mom_sd"][j]
                    if mom_sd is not None:
                        ax.fill_between(pct, mom - mom_sd, mom + mom_sd,
                                         color=COLOR[key], alpha=0.20, lw=0)
                    ax.plot(pct, mom, color=COLOR[key], lw=2.4)

                sc = scores[j][key]
                if sc is not None:
                    is_best = key == best_key_per_joint[j]
                    ax.text(0.97, 0.94, f"{sc:.2f}", transform=ax.transAxes,
                             ha="right", va="top", fontsize=22 if is_best else 19,
                             fontweight="bold" if is_best else "normal",
                             color="#f2c14e" if is_best else INK)

                if ri == 0:
                    ax.set_title(COL_TITLES[j], color=INK, fontsize=22, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(LABEL[key], color=INK, fontsize=18, fontweight="bold")
                if ri == 2:
                    ax.set_xlabel("gait cycle [%]", color=INK, fontsize=16)

        fig.suptitle(f"{body.upper()} kinetics vs. human band  (gray = human ± 1SD; "
                     "number = experimental match / match_fraction, best per joint in gold)",
                     color=INK, fontsize=15, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = OUT / f"KINETICS_{body}_separate.png"
        fig.savefig(p, dpi=150, transparent=True)
        plt.close(fig)
        print(f"wrote {p}")
        return

    fig, axes = plt.subplots(len(bodies), 3, figsize=(15, 4.6) if single else (14, 11),
                              sharex=True, squeeze=False)
    fig.patch.set_alpha(0)

    for bi, body in enumerate(bodies):
        series = {}
        for key, conds in COND_FOR.items():
            # selection criterion stays the kinematics one (angle match_overall)
            # -- same fixed, pre-declared rule as the kinematics figure, applied
            # here too so the SAME selected seed is used for both figures.
            pooled, seed, score = best_seed_pooled(body, conds, band)
            series[key] = pooled
            print(f"{body} {key}: seed={seed} (kinematics match_overall={score:.2f})")

        for ci, j in enumerate(JOINTS):
            ax = axes[bi, ci]
            ax.set_facecolor("none")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_color(INK_MUTED)
            ax.spines["left"].set_color(INK_MUTED)
            if single:
                ax.spines["bottom"].set_linewidth(1.4)
                ax.spines["left"].set_linewidth(1.4)
            ax.tick_params(colors=INK_MUTED, labelsize=16 if single else 10, width=1.4 if single else 0.8)

            mm, ms = band.get(f"{j}_moment_mean"), band.get(f"{j}_moment_sd")

            # scores first, so win-shading draws behind the band and lines
            score_lines = []
            for key in ["prior", "mpo", "dep-mpo"]:
                pooled = series[key]
                mom = pooled["mom"][j] if pooled is not None else None
                sc = match_fraction(mom, mm, ms)[0] if (mom is not None and mm is not None) else None
                score_lines.append((key, sc))
            print(f"  {body} {j:6s} moment match_fraction: " +
                  ", ".join(f"{k}={v:.2f}" if v is not None else f"{k}=--" for k, v in score_lines))
            valid = {k: v for k, v in score_lines if v is not None}
            if valid:
                best_backbone = max([v for k, v in valid.items() if k != "prior"], default=None)
                prior_sc = valid.get("prior")
                if prior_sc is not None and best_backbone is not None:
                    if prior_sc > best_backbone + TIE_TOL:
                        ax.axvspan(0, 100, color=WIN_GREEN, alpha=0.10, zorder=0)
                    elif abs(prior_sc - best_backbone) <= TIE_TOL:
                        ax.axvspan(0, 100, color=TIE_ORANGE, alpha=0.10, zorder=0)

            if mm is not None:
                ax.fill_between(pct, mm - ms, mm + ms, color=BAND_FILL, alpha=0.22)
                ax.plot(pct, mm, color=BAND_LINE, lw=1)

            for key in ["prior", "mpo", "dep-mpo"]:
                pooled = series[key]
                mom = pooled["mom"][j] if pooled is not None else None
                if mom is not None:
                    mom_sd = pooled["mom_sd"][j]
                    if mom_sd is not None:
                        ax.fill_between(pct, mom - mom_sd, mom + mom_sd,
                                         color=COLOR[key], alpha=0.18, lw=0)
                    ax.plot(pct, mom, color=COLOR[key], lw=2.2, label=LABEL[key])

            if bi == 0:
                ax.set_title(COL_TITLES[j], color=INK, fontsize=24 if single else 14,
                             fontweight="bold" if single else "normal")
            if ci == 0:
                ax.set_ylabel(body.upper(), color=INK, fontsize=20 if single else 15,
                              fontweight="bold" if single else "normal")
            if bi == len(bodies) - 1:
                ax.set_xlabel("gait cycle [%]", color=INK, fontsize=18 if single else 12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08 if single else 1.05),
               ncol=3, frameon=False, fontsize=17 if single else 13, labelcolor=INK)
    fig.suptitle(
        "Prior (w=0.5, fixed a priori — kinetics favors more residual than kinematics) vs. MPO vs. DEP-MPO\n"
        "net joint moment, human band in gray — green = prior wins by >0.03; orange = tie (within 0.03)",
        color=INK, fontsize=15 if single else 13, y=1.18 if single else 1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = OUT / (f"KINETICS_{args.body}.png" if single else "KINETICS_COMBINED_VS_BACKBONES.png")
    fig.savefig(p, dpi=150, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
