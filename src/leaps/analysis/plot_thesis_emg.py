"""Thesis version of the F2 muscle-activity figure (biomech_fidelity.f2): white background, thesis
variant names, two blocks of four muscles. Same data and functions as F2 (cross_human, pool_condition,
t* map, MIN_ROWS), so the boxes equal the muscle-timing table.

Laid out like the kinematics figures: one row per condition, one column per muscle, peak-normalized
mean activation against the 22-subject EMG mean +-1 SD, badge = mean Pearson r with the 22 subjects
(gold = best condition for that muscle; bold = within the human range, >= 5th pct of subject pairs).

    python plot_thesis_emg.py --body h2190
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from leaps.envs.emg_mapping import MODEL_MAPS
from leaps.scripts.biomech_fidelity import (CAPTION_ORDER, MIN_ROWS, PARK_ABBR, RUNS, cross_human,
                                            LIVE, load_tstar_map, pool_condition, read_sto,
                                            segment_cycles)
from leaps.data import time_normalize_stride
from leaps.paths import FIGURES_DIR

OUT = str(FIGURES_DIR / "thesis")
CONDS = {"h0918": ["mpo", "dep-mpo", "real-LAP", "real-LAP-w05"],
         "h1622": ["mpo", "dep-mpo", "real-LAP-w01-seed0", "real-LAP-w05"],
         "h2190": ["mpo", "dep-mpo", "real-LAP", "real-LAP-w05"]}
LABEL = {"mpo": "MPO", "dep-mpo": "DEP-MPO", "real-LAP-w00": "EMG prior, $w=0$",
         "real-LAP": "EMG prior, $w=0.1$", "real-LAP-w01-seed0": "EMG prior, $w=0.1$ (walking seed)",
         "real-LAP-w05": "EMG prior, $w=0.5$"}
COLOR = {"mpo": "#1f77b4", "dep-mpo": "#9085e9", "real-LAP-w00": "#56b4e9", "real-LAP": "#d95926",
         "real-LAP-w01-seed0": "#d95926", "real-LAP-w05": "#e64545"}
HUMAN_FILL, HUMAN_LINE = "#cfd4da", "#6b6f75"


def stride_sd(run_dirs, muscles, min_rows, tmap):
    """Per-muscle SD over strides, on the same scale as pool_condition's mean (divided by the peak of
    the mean curve). Same checkpoint, fall filter and cycle segmentation as pool_condition."""
    cyc = {m: [] for m in muscles}
    for rd in run_dirs:
        root = LIVE.parent / rd if rd.startswith("not_used/") else LIVE / rd
        cks = sorted(root.glob("*/run_checkpoint_*"))
        if not cks:
            continue
        want = tmap.get(rd)
        pref = [p for p in cks if int(p.name.split("_")[-1]) == want] if want else []
        ck = pref[0] if pref else max(cks, key=lambda p: int(p.name.split("_")[-1]))
        for sto in sorted(ck.glob("*.sto")):
            ep = read_sto(sto)
            if len(ep) < min_rows:
                continue
            g = "leg1_r.grf_y" if "leg1_r.grf_y" in ep.columns else "leg0_r.grf_y"
            cc = segment_cycles(ep[g].to_numpy(), 0.05 * float(np.nanmax(ep[g].to_numpy())))
            for m in muscles:
                if cc and f"{m}.activation" in ep.columns:
                    sig = ep[f"{m}.activation"].to_numpy()
                    cyc[m] += [time_normalize_stride(sig[a:b, None])[:, 0] for a, b in cc]
    out = {}
    for m, c in cyc.items():
        if c:
            A = np.stack(c)
            peak = np.nanmax(np.abs(A.mean(0)))
            out[m] = A.std(0) / (peak if peak > 1e-9 else 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="h2190", choices=CONDS)
    ap.add_argument("--conds", nargs="+", default=None)
    a = ap.parse_args()
    body, conds = a.body, a.conds or CONDS[a.body]
    if body == "h1622":  # pooled w=0.1 on H1622 is mostly hopping seeds -- say so in the label
        LABEL["real-LAP"] = "EMG prior, $w=0.1$ (all seeds, 3 of 4 hop)"
        COLOR["real-LAP-w01-seed0"] = "#f4a582"

    cmap = MODEL_MAPS[body]
    channels = [c for c in CAPTION_ORDER if c in cmap]
    _, sub_means, ch_idx, pair_r, envelope = cross_human(1.2, 0.15)   # F2's reference
    subs = list(sub_means)
    tmap = load_tstar_map()
    muscles = sorted({cmap[c] for c in channels})
    waves = {c: pool_condition(RUNS[body][c], muscles, min_rows=MIN_ROWS.get(body, 2500),
                               prefer_ckpt=tmap)[0] for c in conds}

    sds = {c: stride_sd(RUNS[body][c], muscles, MIN_ROWS.get(body, 2500), tmap) for c in conds}
    r = {(c, ch): np.mean([stats.pearsonr(waves[c][cmap[ch]], sub_means[s][:, ch_idx[ch]])[0]
                            for s in subs]) for c in conds for ch in channels}
    best = {ch: max(conds, key=lambda c: r[c, ch]) for ch in channels}
    fig, axes = plt.subplots(len(conds), len(channels), figsize=(2.5 * len(channels), 2.3 * len(conds)),
                             sharex=True, sharey=True, squeeze=False)
    pct = np.linspace(0, 100, 101)
    for ri, c in enumerate(conds):
        for ci, ch in enumerate(channels):
            ax = axes[ri, ci]
            hm, hs = envelope[ch]
            ax.fill_between(pct, hm - hs, hm + hs, color=HUMAN_FILL, alpha=0.7, lw=0, label="human $\\pm1$ SD")
            ax.plot(pct, hm, color=HUMAN_LINE, lw=1)
            wv, sd = waves[c][cmap[ch]], sds[c][cmap[ch]]
            ax.fill_between(pct, wv - sd, wv + sd, color=COLOR[c], alpha=0.2, lw=0)
            ax.plot(pct, wv, color=COLOR[c], lw=2, label=LABEL[c])
            in_band = r[c, ch] >= np.percentile(pair_r[ch], 5)
            ax.text(0.96, 0.95, f"$r$ = {r[c, ch]:.2f}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, fontweight="bold" if in_band else "normal",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#f2c14e" if best[ch] == c else "#e6e6e6", ec="none"))
            ax.set_ylim(-0.3, 1.6)
            ax.spines[["top", "right"]].set_visible(False)
            if ri == 0:
                ax.set_title(PARK_ABBR[ch], fontsize=13, fontweight="bold")
            if ri == len(conds) - 1:
                ax.set_xlabel("gait cycle [%]")
            if ci == 0:
                ax.set_ylabel(LABEL[c].replace(", ", ",\n"), fontsize=10)
        print(c, {PARK_ABBR[ch]: round(r[c, ch], 2) for ch in channels})
    handles = [axes[0, 0].get_legend_handles_labels()[0][0]] + [axes[i, 0].get_legend_handles_labels()[0][1]
                                                                 for i in range(len(conds))]
    fig.legend(handles, ["human EMG $\\pm1$ SD"] + [LABEL[c] for c in conds], loc="upper center",
               ncol=len(conds) + 1, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{body}_emg_timing{'_' + '-'.join(a.conds) if a.conds else ''}.{ext}"), dpi=300, facecolor="white",
                    bbox_inches="tight")
    print("saved", f"{body}_emg_timing")


if __name__ == "__main__":
    main()
