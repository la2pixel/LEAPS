"""Pelvis-feet lateral symmetry, DEP-RL (Schumacher et al. 2023, arXiv:2206.00484)
Figure 5 style: "averaged relative pelvis-deviation of the feet" -- lower is
more symmetric/alternating, human reference = 0. Reuses the pelvis_feet_asym
column gait_metrics.py already computes (results/gait_metrics/GAIT_METRICS.csv).

h0918 excluded on principle, not just missing data: it's the 2D sagittal-only
model (no frontal-plane DOF), so lateral pelvis-feet deviation is undefined
there, not merely unmeasured.

    python -m leaps.analysis.pelvis_symmetry --body h2190
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from leaps.paths import RESULTS_DIR

CSV = RESULTS_DIR / "gait_metrics/GAIT_METRICS.csv"
OUT = RESULTS_DIR / "gait_metrics"

ORDER = ["MPO", "DEP-MPO", "EMG prior w=0.0", "EMG prior w=0.1", "EMG prior w=0.5", "null"]
COLOR = {"MPO": "#6fa8dc", "DEP-MPO": "#b35806", "null": "#7f7f7f",
         "EMG prior w=0.0": "#e64545", "EMG prior w=0.1": "#e64545", "EMG prior w=0.5": "#e64545"}
INK = "#ffffff"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="h2190", choices=["h1622", "h2190"])
    args = ap.parse_args()

    rows = {r["method"]: float(r["pelvis_feet_asym"]) for r in csv.DictReader(open(CSV))
            if r["body"] == args.body and r["pelvis_feet_asym"] not in ("", "nan")}
    labels = [m for m in ORDER if m in rows]
    vals = [rows[m] for m in labels]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    bars = ax.bar(labels, vals, color=[COLOR[m] for m in labels], width=0.6)
    ax.axhline(0, ls="--", color=INK, lw=1.5, label="human reference")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", color=INK, fontsize=15, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=13)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", color=INK, fontsize=13)
    ax.set_ylabel("pelvis-feet asymmetry\n(lower = more symmetric)", color=INK, fontsize=15)
    ax.set_title(f"{args.body.upper()}: pelvis-feet gait symmetry (human = 0, dashed)",
                 color=INK, fontsize=15, fontweight="bold")
    fig.tight_layout()
    p = OUT / f"PELVIS_SYMMETRY_{args.body}.png"
    fig.savefig(p, dpi=150, transparent=True)
    plt.close(fig)
    print(f"wrote {p}")
    for m in labels:
        print(f"  {m:20s} {rows[m]:.3f}")


if __name__ == "__main__":
    main()
