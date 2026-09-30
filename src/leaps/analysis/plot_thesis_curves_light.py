"""Thesis learning curves: same runs and smoothing as plot_h0918_prior_curves.py, paper style
(white background, no grid, thesis variant names). Untrained decoder: w=0.1/0.5, H0918 only (its
decoder was redrawn per env build, so it is not a fixed content control). Writes PNG + PDF to results/figures/thesis/."""
import os

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leaps.analysis import plot_h0918_prior_curves as c
from leaps.paths import FIGURES_DIR

OUT = str(FIGURES_DIR / "thesis")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.8})

# thesis label -> (defense label, colour)
NAMES = {
    "MPO": ("MPO (backbone)", "#7f7f7f"),
    "DEP-MPO": ("DEP-MPO (backbone)", "#9085e9"),
    "EMG prior, $w=0$": ("only prior (w=0.0)", "#3987e5"),
    "EMG prior, $w=0.1$": ("prior + 0.1 residual", "#d95926"),
    "EMG prior, $w=0.5$": ("prior + 0.5 residual", "#199e70"),
    "Null prior": ("no prior (null)", "#c98500"),
    "Untrained prior, $w=0.1$": ("untrained decoder (w=0.1)", "#2eb8b8"),
    "Untrained prior, $w=0.5$": ("untrained decoder (w=0.5)", "#d55181"),
}
FULL_NAMES = {"MPO": ("MPO (backbone, full)", "#7f7f7f"),
              "EMG prior, $w=0.1$": ("EMG prior (ours, w=0.1, full)", "#d95926")}


def x_limit(results):
    """Convergence (first point within 10% of the best curve) + 1M; full run if most never get there."""
    peak = max(r["test_mean"].max() for r in results.values())
    conv = [r["steps"][np.argmax(r["test_mean"] >= 0.9 * peak)] / 1e6
            for r in results.values() if (r["test_mean"] >= 0.9 * peak).any()]
    full = max(r["steps"].max() for r in results.values()) / 1e6
    return min(full, max(conv) + 1.0) if len(conv) >= len(results) / 2 else full


def plot(runs, names, out_name, y_max=1050):
    results = {lab: c.load_condition(runs[src]) for lab, (src, _) in names.items() if src in runs}
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for lab, r in results.items():
        col = names[lab][1]
        x = r["steps"] / 1e6
        ax.plot(x, r["test_mean"], color=col, lw=1.6, label=lab)
        hit = x[np.argmax(r["test_mean"] >= 950)] if (r["test_mean"] >= 950).any() else None
        print(f"  {out_name} {lab}: n={r['n_seeds']} reach950={hit}")
        ax.fill_between(x, r["test_mean"] - r["test_std"], r["test_mean"] + r["test_std"],
                        color=col, alpha=0.15, lw=0)
    ax.set_xlim(0, x_limit(results))
    ax.set_ylim(0, y_max)
    ax.set_xlabel(r"Timesteps ($\times10^{6}$)")
    ax.set_ylabel("Mean test episode return\n(20 rollouts)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              columnspacing=1.2, handlelength=1.8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{out_name}.{ext}"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", out_name)


# H0918 w=0 / w=0.5 have a finished seed2 the defense plots never picked up; use all three
H0918 = dict(c.H0918_ALL_VARIANTS)
for lab, w in (("only prior (w=0.0)", "00"), ("prior + 0.5 residual", "05")):
    H0918[lab] = [f"{c.LIVE}/emg_lap/h0918/h0918_k6_w{w}_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1, 2)]
variants = {"h0918": H0918, "h1622": c.H1622_ALL_VARIANTS, "h2190": c.H2190_ALL_VARIANTS}
for body, v in variants.items():
    plot({**c.BACKBONE_COMPARISON[body], **v}, NAMES, f"{body}_learning_curves",
         y_max=800 if body == "h2190" else 1050)
for body in ("h0918", "h1622"):
    plot(c.FULL_REWARD_COMPARISON[body], FULL_NAMES, f"{body}_learning_curves_full", y_max=11000)


# H0918 sensitivity, laid out like Hausdoerfer et al. Fig. 9: (a) w at k=6, (b) k at w=0.1.
# H1622/H2190 w-sensitivity goes in a table in the thesis, not here.
SWEEP = f"{c.LIVE}/emg_lap/{{b}}/{{b}}_k{{k}}_w{{w}}_onlyvelrew_AB06_corrected_seed{{s}}"
W_LAB = {"00": "0", "01": "0.1", "02": "0.2", "03": "0.3", "05": "0.5"}


def w_panel(body, seeds):
    return {W_LAB[w]: [SWEEP.format(b=body, k=6, w=w, s=s) for s in ss] for w, ss in seeds.items()}


PANELS = [
    (r"H0918, residual weight $w$ ($k=6$)", "viridis", 5,
     w_panel("h0918", {w: (0, 1, 2) for w in ("00", "01", "02", "03", "05")})),
    (r"H0918, latent dimension $k$ ($w=0.1$)", "plasma", 5,
     {str(k): [SWEEP.format(b="h0918", k=k, w="01", s=s) for s in (0, 1, 2)] for k in (2, 4, 6, 8, 11)}),
]
W_COL = dict(zip(["0", "0.1", "0.2", "0.3", "0.5"], plt.get_cmap("viridis")(np.linspace(0.05, 0.85, 5))))
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
for ax, (title, cmap, xmax, conds) in zip(axes.flat, PANELS):
    cols = (W_COL[l] for l in conds) if cmap == "viridis" else \
        iter(plt.get_cmap(cmap)(np.linspace(0.05, 0.85, len(conds))))
    for (lab, dirs), col in zip(conds.items(), cols):
        r = c.load_condition(dirs)
        x, m, sd = r["steps"] / 1e6, r["test_mean"], r["test_std"]
        ax.plot(x, m, color=col, lw=1.4, label=f"{lab}" + (" ($n=1$)" if len(dirs) == 1 else ""))
        ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        hits = {t: (x[np.argmax(m >= t)] if (m >= t).any() else None) for t in (300, 900, 950)}
        print(f"{title[:6]} {lab:>4} n={len(dirs)} hits={hits} peak={m.max():.0f}@{x[m.argmax()]:.1f}")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1050)
    ax.set_title(title.split(", ", 1)[1][0].upper() + title.split(", ", 1)[1][1:], fontsize=10.5, pad=34)
    ax.set_xlabel(r"Timesteps ($\times10^{6}$)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
              columnspacing=1.0, handlelength=1.5, fontsize=8.5)
for ax, lab in zip(axes.flat, "abcd"):
    ax.text(-0.14, 1.16, lab, transform=ax.transAxes, fontsize=12, fontweight="bold")
for ax in axes[:1]:
    ax.set_ylabel("Mean test episode return\n(20 rollouts)")
fig.tight_layout(h_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"h0918_sensitivity.{ext}"), dpi=300, facecolor="white", bbox_inches="tight")
print("saved h0918_sensitivity")
