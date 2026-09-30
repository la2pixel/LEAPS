"""Test-effort comparison as a dot/lollipop plot instead of zero-baseline bars.

Why: the effort differences between conditions are real but small in absolute
terms (~10-15% relative) -- a zero-baseline bar chart makes them nearly
invisible, but truncating a BAR chart's y-axis to fix that is a classic
misleading-graph move (bar length/area reads as proportional magnitude, so
cutting the baseline distorts that reading on purpose). A dot plot doesn't
carry that convention -- position, not area, encodes the value -- so zooming
the axis to the actual data range is standard, accepted practice here, not a
distortion. Human reference is not plotted (no human "test effort" ground
truth exists for this quantity, unlike the kinematics/EMG comparisons).

    python -m leaps.analysis.effort_dotplot
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from leaps.paths import RESULTS_DIR

CSV = RESULTS_DIR / "gait_metrics/GAIT_METRICS.csv"
OUT = RESULTS_DIR / "gait_metrics"

ORDER = ["MPO", "DEP-MPO", "EMG prior w=0.0", "EMG prior w=0.1", "EMG prior w=0.5"]
COLOR = {"MPO": "#6fa8dc", "DEP-MPO": "#b35806",
         "EMG prior w=0.0": "#e64545", "EMG prior w=0.1": "#e64545", "EMG prior w=0.5": "#e64545"}
INK = "#ffffff"
BODIES = ["h0918", "h1622", "h2190"]


def main():
    data = {}
    for r in csv.DictReader(open(CSV)):
        if r["body"] in BODIES and r["method"] in ORDER:
            data.setdefault(r["body"], {})[r["method"]] = float(r["effort_meanact"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
    fig.patch.set_alpha(0)

    for ax, body in zip(axes, BODIES):
        vals = data[body]
        labels = [m for m in ORDER if m in vals]
        y = list(range(len(labels)))[::-1]
        xs = [vals[m] for m in labels]
        lo, hi = min(xs), max(xs)
        pad = max((hi - lo) * 0.4, 0.01)

        best = min(xs)
        for yi, m, x in zip(y, labels, xs):
            ax.hlines(yi, lo - pad, x, color=COLOR[m], alpha=0.4, lw=2)
            ax.scatter([x], [yi], color=COLOR[m], s=180, zorder=3,
                       edgecolor=INK, linewidth=1.2)
            pct = (x - best) / best * 100
            tag = "  (lowest)" if x == best else f"  (+{pct:.0f}%)"
            ax.text(x + pad * 0.35, yi, f"{x:.3f}{tag}", va="center", ha="left",
                     color=INK, fontsize=13, fontweight="bold" if x == best else "normal")

        ax.set_xlim(lo - pad, hi + pad * 2.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, color=INK, fontsize=13)
        ax.set_xlabel("test effort (mean muscle activation)", color=INK, fontsize=12)
        ax.set_title(body.upper(), color=INK, fontsize=18, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(INK)
        ax.tick_params(colors=INK)
        ax.set_facecolor("none")

    fig.suptitle("Test effort by condition  (axis zoomed to the data range -- "
                  "point position, not bar area, encodes the value)",
                  color=INK, fontsize=14, y=1.03)
    fig.tight_layout()
    p = OUT / "effort_dotplot_all_bodies.png"
    fig.savefig(p, dpi=150, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
