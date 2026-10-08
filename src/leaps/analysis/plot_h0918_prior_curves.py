"""Learning curves per body: mean +/- SD across seeds of test/episode_score/mean
(20 test rollouts per evaluation) against training steps, smoothed with a
3-epoch moving average. Dark-theme transparent PNGs.

The numbers printed per condition are the re-evaluated returns at the selected
checkpoint (results/checkpoint_selection/<label>_reeval.csv) where available.
"""
import csv
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from leaps.paths import FIGURES_DIR, RESULTS_DIR, SCONE_LIVE_DIR

LIVE = str(SCONE_LIVE_DIR / "lalitha/final_experiments")
FIG_DIR = str(FIGURES_DIR)
CHECKPOINT_DIR = str(RESULTS_DIR / "checkpoint_selection")

# condition label -> checkpoint_selection label
_OFFICIAL_LABEL = {
    "MPO (backbone)": "mpo_{body}_onlyvelrew",
    "DEP-MPO (backbone)": "depmpo_{body}_onlyvelrew",
    "EMG prior (ours, w=0.1)": "emg_lap_{body}_w01_onlyvelrew",
    "prior + 0.1 residual": "emg_lap_{body}_w01_onlyvelrew",
    "only prior (w=0.0)": "emg_lap_{body}_w00_onlyvelrew",
    "MPO (backbone, full)": "mpo_{body}_full",
    "EMG prior (ours, w=0.1, full)": "emg_lap_{body}_w01_full",
    "prior + 0.5 residual": "emg_lap_{body}_w05_onlyvelrew",
    "no prior (null)": "null_{body}_onlyvelrew",
    "untrained decoder": "untrainedlap_{body}_w05_onlyvelrew",
    "untrained decoder (w=0.5)": "untrainedlap_{body}_w05_onlyvelrew",
}


def load_official(body: str, condition: str):
    tmpl = _OFFICIAL_LABEL.get(condition)
    if tmpl is None:
        return None
    path = f"{CHECKPOINT_DIR}/{tmpl.format(body=body)}_reeval.csv"
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    means = [float(r["reeval_mean"]) for r in rows]
    falls = [float(r["fall_pct"]) for r in rows]
    return dict(mean=float(np.mean(means)), sd=float(np.std(means)),
                fall=float(np.mean(falls)), n=len(rows))

# one color per condition, the same in every figure
COLOR_BY_LABEL = {
    "only prior (w=0.0)": "#3987e5",
    "prior + 0.1 residual": "#d95926",
    "prior + 0.5 residual": "#199e70",
    "no prior (null)": "#c98500",
    "untrained decoder": "#d55181",
    "untrained decoder (w=0.1)": "#2eb8b8",
    "untrained decoder (w=0.5)": "#d55181",
    "MPO (backbone)": "#a0a0a0",
    "DEP-MPO (backbone)": "#9085e9",
    "EMG prior (ours, w=0.1)": "#d95926",  # same orange as "prior + 0.1 residual" -- same condition
    "MPO (backbone, full)": "#a0a0a0",
    "EMG prior (ours, w=0.1, full)": "#d95926",
}

# full reward: prior vs MPO. No h2190 runs with the current recipe.
FULL_REWARD_COMPARISON = {
    "h0918": {
        "MPO (backbone, full)": [f"{LIVE}/mpo/h0918/h0918_full_seed{s}" for s in (0, 1, 2)],
        "EMG prior (ours, w=0.1, full)": [
            f"{LIVE}/emg_lap/h0918/h0918_k6_w01_full_AB06_corrected_seed{s}" for s in (0, 1, 2)
        ],
    },
    "h1622": {
        "MPO (backbone, full)": [f"{LIVE}/mpo/h1622/h1622_full_seed{s}" for s in (0, 1, 2)],
        "EMG prior (ours, w=0.1, full)": [
            f"{LIVE}/emg_lap/h1622/h1622_k6_w01_full_AB06_corrected_seed{s}" for s in (0, 1, 2)
        ],
    },
}

# backbones vs the w=0.1 EMG prior. h2190 MPO/DEP-MPO seed2 stopped at 5M and is left out.
BACKBONE_COMPARISON = {
    "h0918": {
        "MPO (backbone)": [f"{LIVE}/mpo/h0918/h0918_onlyvelrew_seed{s}" for s in (0, 1, 2)],
        "DEP-MPO (backbone)": [f"{LIVE}/dep-mpo/h0918/h0918_onlyvelrew_dep_seed{s}" for s in (0, 1, 2)],
        "EMG prior (ours, w=0.1)": [
            f"{LIVE}/emg_lap/h0918/h0918_k6_w01_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1, 2)
        ],
        "prior + 0.5 residual": [
            f"{LIVE}/emg_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
        ],
    },
    "h1622": {
        "MPO (backbone)": [f"{LIVE}/mpo/h1622/h1622_onlyvelrew_seed{s}" for s in (0, 1, 2)],
        "DEP-MPO (backbone)": [f"{LIVE}/dep-mpo/h1622/h1622_onlyvelrew_dep_seed{s}" for s in (0, 1, 2)],
        "EMG prior (ours, w=0.1)": [
            f"{LIVE}/emg_lap/h1622/h1622_k6_w01_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1, 2, 3)
        ],
        "prior + 0.5 residual": [
            f"{LIVE}/emg_lap/h1622/h1622_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
        ],
    },
    "h2190": {
        "MPO (backbone)": [f"{LIVE}/mpo/h2190/h2190_onlyvelrew_seed{s}" for s in (0, 1)],
        "DEP-MPO (backbone)": [
            f"{LIVE}/dep-mpo/h2190/h2190_onlyvelrew_dep_seed0",
            f"{LIVE}/dep-mpo/h2190/h2190_noclip_net256_onlyvelrew_seed1",
        ],
        "EMG prior (ours, w=0.1)": [
            f"{LIVE}/emg_lap/h2190/h2190_k6_w01_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1, 2, 3)
        ],
        "prior + 0.5 residual": [
            f"{LIVE}/emg_lap/h2190/h2190_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
        ],
    },
}

# all five prior variants (only h0918 has the untrained decoder at every w)
H0918_ALL_VARIANTS = {
    "only prior (w=0.0)": [
        f"{LIVE}/emg_lap/h0918/h0918_k6_w00_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
    ],
    "prior + 0.1 residual": [
        f"{LIVE}/emg_lap/h0918/h0918_k6_w01_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1, 2)
    ],
    "prior + 0.5 residual": [
        f"{LIVE}/emg_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
    ],
    "no prior (null)": [
        f"{LIVE}/no_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_null_seed{s}" for s in (0, 1, 2)
    ],
    "untrained decoder (w=0.1)": [
        f"{LIVE}/untrained_lap/h0918/h0918_k6_w01_onlyvelrew_AB06_corrected_untrainedprior_seed{s}" for s in (0, 1)
    ],
    "untrained decoder (w=0.5)": [
        f"{LIVE}/untrained_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_untrainedprior_seed{s}" for s in (0, 1)
    ],
}

H1622_ALL_VARIANTS = {
    "only prior (w=0.0)": [
        f"{LIVE}/emg_lap/h1622/h1622_k6_w00_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
    ],
    "prior + 0.1 residual": [
        f"{LIVE}/emg_lap/h1622/h1622_k6_w01_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1, 2, 3)
    ],
    "prior + 0.5 residual": [
        f"{LIVE}/emg_lap/h1622/h1622_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
    ],
    "no prior (null)": [
        f"{LIVE}/no_lap/h1622/h1622_k6_w05_onlyvelrew_AB06_corrected_null_seed{s}" for s in (0, 1, 2)
    ],
    "untrained decoder": [
        f"{LIVE}/untrained_lap/h1622/h1622_k6_w05_onlyvelrew_AB06_corrected_untrainedprior_seed{s}" for s in (0, 1)
    ],
}

H2190_ALL_VARIANTS = {
    "only prior (w=0.0)": [
        f"{LIVE}/emg_lap/h2190/h2190_k6_w00_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
    ],
    "prior + 0.1 residual": [
        f"{LIVE}/emg_lap/h2190/h2190_k6_w01_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1, 2, 3)
    ],
    "prior + 0.5 residual": [
        f"{LIVE}/emg_lap/h2190/h2190_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
    ],
    "no prior (null)": [
        f"{LIVE}/no_lap/h2190/h2190_k6_w05_onlyvelrew_AB06_corrected_null_seed{s}" for s in (0, 1)
    ],
    "untrained decoder": [
        f"{LIVE}/untrained_lap/h2190/h2190_k6_w05_onlyvelrew_AB06_corrected_untrainedprior_seed{s}" for s in (0, 1)
    ],
}

BODIES = {
    "h0918": {
        "only prior (w=0.0)": [
            f"{LIVE}/emg_lap/h0918/h0918_k6_w00_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
        ],
        "prior + 0.1 residual": [
            f"{LIVE}/emg_lap/h0918/h0918_k6_w01_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1, 2)
        ],
        "prior + 0.5 residual": [
            f"{LIVE}/emg_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
        ],
        "no prior (null)": [
            f"{LIVE}/no_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_null_seed{s}" for s in (0, 1, 2)
        ],
    },
    "h1622": {
        "only prior (w=0.0)": [
            f"{LIVE}/emg_lap/h1622/h1622_k6_w00_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
        ],
        "prior + 0.1 residual": [
            f"{LIVE}/emg_lap/h1622/h1622_k6_w01_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1, 2, 3)
        ],
        "prior + 0.5 residual": [
            f"{LIVE}/emg_lap/h1622/h1622_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
        ],
        "no prior (null)": [
            f"{LIVE}/no_lap/h1622/h1622_k6_w05_onlyvelrew_AB06_corrected_null_seed{s}" for s in (0, 1, 2)
        ],
    },
    "h2190": {
        "only prior (w=0.0)": [
            f"{LIVE}/emg_lap/h2190/h2190_k6_w00_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
        ],
        "prior + 0.1 residual": [
            f"{LIVE}/emg_lap/h2190/h2190_k6_w01_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1, 2, 3)
        ],
        "prior + 0.5 residual": [
            f"{LIVE}/emg_lap/h2190/h2190_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)
        ],
        "no prior (null)": [
            f"{LIVE}/no_lap/h2190/h2190_k6_w05_onlyvelrew_AB06_corrected_null_seed{s}" for s in (0, 1)
        ],
    },
}

MAX_STEPS = 20e6      # runs extended past 20M are truncated
SMOOTH_WINDOW = 3      # epochs (0.6M steps)

INK_PRIMARY = "#ffffff"
INK_MUTED = "#898781"
GRID = (1, 1, 1, 0.08)
BASELINE = (1, 1, 1, 0.25)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average; the window shrinks at the edges."""
    out = np.empty_like(x)
    half = window // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        out[i] = x[lo:hi].mean()
    return out


def load_condition(dirs: list[str]) -> dict:
    steps_ref = None
    test_curves = []
    for d in dirs:
        p = glob.glob(f"{d}/*/log.csv")[0]
        df = pd.read_csv(p)
        df = df[df["train/steps"] <= MAX_STEPS]
        steps = df["train/steps"].to_numpy()
        if steps_ref is None:
            steps_ref = steps
        assert len(steps) == len(steps_ref), f"{d} length mismatch"
        test_curves.append(df["test/episode_score/mean"].to_numpy())
    test_arr = np.stack(test_curves)
    mean_raw = test_arr.mean(axis=0)
    std_raw = test_arr.std(axis=0)
    return dict(
        steps=steps_ref,
        test_mean=moving_average(mean_raw, SMOOTH_WINDOW),
        test_std=moving_average(std_raw, SMOOTH_WINDOW),
        n_seeds=len(dirs),
        final_test_mean=test_arr[:, -1].mean(), final_test_std=test_arr[:, -1].std(),
    )


def plot_body(body: str, conditions: dict[str, list[str]], show_legend: bool = True,
              out_name: str = None, title_suffix: str = "w-sensitivity",
              legend_loc: str = "best", legend_below: bool = False,
              y_max: float = 1050, y_label: str = "mean episode return (20 rollouts)") -> None:
    order = list(conditions.keys())
    results = {label: load_condition(dirs) for label, dirs in conditions.items()}

    # trim the x-axis 1M steps after convergence (within 10% of the best
    # smoothed score), but only if most conditions get there
    peak_ref = max(r["test_mean"].max() for r in results.values())
    target = 0.9 * peak_ref
    conv_points = []
    for r in results.values():
        reached = r["test_mean"] >= target
        if reached.any():
            conv_points.append(r["steps"][int(np.argmax(reached))] / 1e6)
    full_range_m = max(r["steps"].max() for r in results.values()) / 1e6
    if conv_points and len(conv_points) >= len(order) / 2:
        x_trim = min(full_range_m, max(conv_points) + 1.0)
    else:
        x_trim = full_range_m

    fig = plt.figure(figsize=(9.5, 6.1 if legend_below else 5.6), dpi=200)
    fig.patch.set_alpha(0)
    ax_bottom = 0.24 if legend_below else 0.13
    ax_height = 0.65 if legend_below else 0.72
    ax_left = 0.13 if y_max > 2000 else 0.09  # wider left margin for 5-digit y-ticks (full reward)
    ax = fig.add_axes([ax_left, ax_bottom, 0.98 - ax_left, ax_height])
    ax.set_facecolor("none")

    for label in order:
        r = results[label]
        x = r["steps"] / 1e6
        c = COLOR_BY_LABEL[label]
        official = load_official(body, label)
        if official:
            legend_label = (f"{label}  ({official['mean']:.0f}±{official['sd']:.1f} @ t*, "
                             f"fall {official['fall']:.0f}%, n={official['n']})")
        else:
            legend_label = f"{label}  ({r['final_test_mean']:.0f}±{r['final_test_std']:.1f} @ 20M, n={r['n_seeds']})"
        ax.plot(x, r["test_mean"], color=c, linewidth=2.0, label=legend_label, zorder=3)
        ax.fill_between(x, r["test_mean"] - r["test_std"], r["test_mean"] + r["test_std"],
                         color=c, alpha=0.15, linewidth=0, zorder=2)

    ax.set_xlim(0, x_trim)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("training steps (millions)", color=INK_PRIMARY, fontsize=16)
    ax.set_ylabel(y_label, color=INK_PRIMARY, fontsize=16)
    ax.tick_params(colors=INK_PRIMARY, labelsize=13)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["left"].set_color(BASELINE)
    ax.set_axisbelow(True)
    if show_legend:
        if legend_below:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                      frameon=False, fontsize=9, labelcolor=INK_PRIMARY, handlelength=1.6)
        else:
            ax.legend(loc=legend_loc, frameon=False, fontsize=9, labelcolor=INK_PRIMARY, handlelength=1.6)
    ax.set_title(
        f"{body.upper()} — EMG-prior {title_suffix} (mean ± 1 SD, {SMOOTH_WINDOW}-epoch moving avg.)",
        color=INK_PRIMARY, fontsize=12.5, loc="left", pad=12, fontweight="bold",
    )
    if x_trim < full_range_m:
        ax.text(
            1.0, 1.0, f"x-axis cut at convergence+1M (full run: {full_range_m:.0f}M)",
            transform=ax.transAxes, ha="right", va="bottom", color=INK_MUTED, fontsize=8,
        )

    out = f"{FIG_DIR}/{out_name or f'{body}_prior_wsensitivity'}.png"
    fig.savefig(out, transparent=True, dpi=200)
    plt.close(fig)
    print(f"saved {out}")
    for label in order:
        r = results[label]
        official = load_official(body, label)
        if official:
            print(f"  {label:24s} n={official['n']}  official={official['mean']:.1f}±{official['sd']:.1f} @ t*"
                  f"  fall={official['fall']:.0f}%")
        else:
            print(f"  {label:24s} n={r['n_seeds']}  final(20M)={r['final_test_mean']:.1f}±{r['final_test_std']:.1f}"
                  f"  [no official t*-reeval for this condition]")


SHOW_LEGEND = False  # per-condition numbers are printed to the console

if __name__ == "__main__":
    for body, conditions in BODIES.items():
        plot_body(body, conditions, show_legend=SHOW_LEGEND)
    # all five prior variants
    plot_body("h0918", H0918_ALL_VARIANTS, out_name="h0918_prior_variants", title_suffix="variants",
              show_legend=SHOW_LEGEND)
    plot_body("h1622", H1622_ALL_VARIANTS, out_name="h1622_prior_variants", title_suffix="variants",
              show_legend=SHOW_LEGEND)
    plot_body("h2190", H2190_ALL_VARIANTS, out_name="h2190_prior_variants", title_suffix="variants",
              show_legend=SHOW_LEGEND)
    # backbones vs the w=0.1 EMG prior
    for body in ("h0918", "h1622", "h2190"):
        plot_body(body, BACKBONE_COMPARISON[body], out_name=f"{body}_vs_backbones",
                  title_suffix="vs. unwrapped backbones", show_legend=SHOW_LEGEND)

    # backbones + prior ladder in one figure per body
    _VARIANTS_BY_BODY = {"h0918": H0918_ALL_VARIANTS, "h1622": H1622_ALL_VARIANTS, "h2190": H2190_ALL_VARIANTS}
    for body in ("h0918", "h1622", "h2190"):
        v = _VARIANTS_BY_BODY[body]
        merged = {
            "MPO (backbone)": BACKBONE_COMPARISON[body]["MPO (backbone)"],
            "DEP-MPO (backbone)": BACKBONE_COMPARISON[body]["DEP-MPO (backbone)"],
            "only prior (w=0.0)": v["only prior (w=0.0)"],
            "prior + 0.1 residual": v["prior + 0.1 residual"],
            "prior + 0.5 residual": v["prior + 0.5 residual"],
        }
        if body == "h0918":
            merged["no prior (null)"] = v["no prior (null)"]
            merged["untrained decoder (w=0.1)"] = v["untrained decoder (w=0.1)"]
        plot_body(body, merged, out_name=f"{body}_all_seven",
                  title_suffix="all conditions", show_legend=SHOW_LEGEND)

    # full reward: prior vs MPO, h0918 and h1622
    for body in ("h0918", "h1622"):
        plot_body(body, FULL_REWARD_COMPARISON[body], out_name=f"{body}_full_vs_backbone",
                  title_suffix="full reward, vs. MPO backbone", show_legend=SHOW_LEGEND,
                  y_max=11000)
