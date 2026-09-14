"""Consolidate every arm's N=100 rollout re-eval into one thesis table + figure.

Reads results/checkpoint_selection/<label>{.csv,_reeval.csv} and emits:
  results/checkpoint_selection/RETURN_TABLE.csv   -- one row per arm
  results/checkpoint_selection/RETURN_TABLE.png   -- grouped by body, per-seed
                                                     points + mean±sd, fall%.
Re-run any time; it picks up new seeds automatically.
"""
import csv
import glob
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

O = "LEAPS/results/checkpoint_selection"
METHOD = {"emg_lap": "EMG prior", "mpo": "MPO", "depmpo": "DEP-MPO", "null": "null"}
MCOL = {"EMG prior": "#d62728", "MPO": "#1f77b4", "DEP-MPO": "#2ca02c", "null": "#7f7f7f"}
MORDER = ["EMG prior", "MPO", "DEP-MPO", "null"]


def parse(label):
    for m in ("emg_lap", "depmpo", "mpo", "null"):
        if label.startswith(m):
            method = METHOD[m]
            break
    else:
        return None
    body = next((b for b in ("h0918", "h1622", "h2190") if b in label), "?")
    regime = "full" if "full" in label else "onlyVelRew"
    return method, body, regime


def main():
    rows = []
    for f in sorted(glob.glob(f"{O}/*_reeval.csv")):
        label = os.path.basename(f)[:-11]
        meta = parse(label)
        if not meta:
            continue
        method, body, regime = meta
        rev = list(csv.DictReader(open(f)))
        sel = {r["seed"]: r for r in csv.DictReader(open(f"{O}/{label}.csv"))}
        means = [float(r["reeval_mean"]) for r in rev]
        falls = [float(r["fall_pct"]) for r in rev]
        seeds = [(r["seed"], float(r["reeval_mean"]), float(r["fall_pct"]),
                  int(r["t_star"])) for r in rev]
        rows.append(dict(
            label=label, method=method, body=body, regime=regime, n=len(rev),
            mean=round(statistics.mean(means), 1),
            sd=round(statistics.pstdev(means) if len(means) > 1 else 0.0, 1),
            sem=round((statistics.pstdev(means) / len(means) ** 0.5) if len(means) > 1 else 0.0, 1),
            fall_pct=round(statistics.mean(falls), 1),
            seeds=seeds))

    # ---- CSV ----
    out = f"{O}/RETURN_TABLE.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["regime", "body", "method", "n_seeds", "rollout_mean",
                    "sd", "sem", "fall_pct", "per_seed(mean/fall%/t*M)"])
        for r in sorted(rows, key=lambda x: (x["regime"], x["body"], MORDER.index(x["method"]))):
            ps = " ; ".join(f"{s}:{m:.0f}/{fp:.0f}%/{ts//10**6}M" for s, m, fp, ts in r["seeds"])
            w.writerow([r["regime"], r["body"], r["method"], r["n"], r["mean"],
                        r["sd"], r["sem"], r["fall_pct"], ps])
    print("wrote", out)

    # ---- printed ----
    print(f"\n{'regime':<11}{'body':<7}{'method':<11}{'n':>2}  {'mean ± sd':>14}  {'fall%':>6}")
    print("-" * 56)
    for r in sorted(rows, key=lambda x: (x["regime"], x["body"], MORDER.index(x["method"]))):
        print(f"{r['regime']:<11}{r['body']:<7}{r['method']:<11}{r['n']:>2}  "
              f"{r['mean']:>7.0f} ± {r['sd']:<4.0f}  {r['fall_pct']:>5.0f}")

    # ---- figure: onlyVelRew, 3 bodies ----
    ov = [r for r in rows if r["regime"] == "onlyVelRew"]
    bodies = ["h0918", "h1622", "h2190"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for ax, body in zip(axes, bodies):
        br = {r["method"]: r for r in ov if r["body"] == body}
        for i, m in enumerate(MORDER):
            if m not in br:
                continue
            r = br[m]
            ax.bar(i, r["mean"], width=0.62, color=MCOL[m], alpha=0.35,
                   yerr=r["sd"], capsize=4, ecolor="black")
            xs = [i + (k - (r["n"] - 1) / 2) * 0.12 for k in range(r["n"])]
            ax.scatter(xs, [s[1] for s in r["seeds"]], color=MCOL[m], s=26, zorder=3,
                       edgecolor="white", linewidth=0.5)
            ax.text(i, 8, f"n={r['n']}\nfall {r['fall_pct']:.0f}%", ha="center",
                    va="bottom", fontsize=7)
        ax.set_xticks(range(len(MORDER)))
        ax.set_xticklabels(MORDER, rotation=30, ha="right", fontsize=8)
        ax.set_title(body, fontsize=11)
        ax.set_ylim(0, 1080 if body != "h2190" else 750)
        ax.grid(axis="y", alpha=0.25)
        if body == "h0918":
            ax.set_ylabel("mean rollout return (N=100 episodes @ t*)")
    fig.suptitle("Return by method — onlyVelRew, deterministic rollout at the "
                 "selected checkpoint (mean ± sd over seeds; dots = seeds)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = f"{O}/RETURN_TABLE.png"
    fig.savefig(png, dpi=140)
    print("wrote", png)

    _trend_fig(rows)
    _curves_fig()


def _label_for(method, body):
    pre = {"EMG prior": f"emg_lap_{body}_w01", "MPO": f"mpo_{body}",
           "DEP-MPO": f"depmpo_{body}", "null": f"null_{body}"}[method]
    return f"{pre}_onlyvelrew"


# pastel, distinct on a dark slide; light "ink" for axes/text so the figure
# can be dropped transparent onto a dark-themed deck.
PASTEL = {"EMG prior": "#ff8a7a", "MPO": "#6fb8ff",
          "DEP-MPO": "#5fd39a", "null": "#8a8f98"}
INK = "#e8e8e8"


def _lighten(hexc, f=0.55):
    """Blend a hex colour f of the way toward white (so an alpha fill still
    reads as *lighter* than a dark slide, not a shadow)."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * f) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


_LIVE_FE = "/media/calc_2/scone_results/live/lalitha/final_experiments"
_LIVE_OLD = "/media/calc_2/scone_results/live/lalitha/not_used/early_tests"
# full-SCONE-reward runs: R4 / R2-under-full. Only MPO exists as a full-reward
# backbone (no DEP-MPO-full, no null-full anywhere). h0918/h1622 are the
# current recipe (k6, AB06_corrected decoder, noclip). h2190 was NEVER run at
# the current recipe -- only the old July recipe (k11, generic pooled decoder,
# clip, net256) under not_used/early_tests/, flagged in the panel title.
# Stub/short runs are dropped by the 0.6*horizon filter.
_FULL_DIRS = {
    "h0918": ({
        "EMG prior": [f"{_LIVE_FE}/emg_lap/h0918/h0918_k6_w01_full_AB06_corrected_seed{s}"
                      for s in (0, 1, 2)],
        "MPO": [f"{_LIVE_FE}/mpo/h0918/h0918_full_seed{s}" for s in (0, 1, 2)],
    }, 10, ""),
    "h1622": ({
        "EMG prior": [f"{_LIVE_FE}/emg_lap/h1622/h1622_k6_w01_full_AB06_corrected_seed{s}"
                      for s in (0, 1, 2)],
        "MPO": [f"{_LIVE_FE}/mpo/h1622/h1622_full_seed{s}" for s in (0, 1, 2)],
    }, 20, ""),
    "h2190": ({
        "EMG prior": [f"{_LIVE_OLD}/emg_lap/h2190/h2190_k11_w01_mirror_clip_net256_full_seed{s}"
                      for s in (0, 1, 2, 3, 4)],
        "MPO": [f"{_LIVE_OLD}/mpo/h2190/h2190_clip_net256_full_seed0"],
    }, 20, "old July recipe (k11 generic decoder, clip) - not comparable"),
}


def _ma(x, w):  # edge-aware centred moving average
    if w < 2:
        return x
    out = np.copy(x)
    for i in range(len(x)):
        lo, hh = max(0, i - w // 2), min(len(x), i + w // 2 + 1)
        out[i] = np.nanmean(x[lo:hh])
    return out


def _one_curve_panel(body, method_dirs, horizon_m, ymax, ylabel, fname,
                     smooth=3, note=""):
    """Mean-over-seeds ±1 SD learning curve for one body. method_dirs =
    {method: [run_dir,...]}. Transparent, dark-slide ink. Convention =
    LAP / DEP-RL / arXiv:2309.02976 (mean + SD band, no best-seed line)."""
    from leaps.scripts.select_checkpoint import load_curve

    hi = horizon_m * 1e6
    grid = np.arange(0, hi + 1, 2e5)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for m in MORDER:
        if m not in method_dirs:
            continue
        curves = []
        for rd in method_dirs[m]:
            got = load_curve(rd)
            if not got:
                continue
            _, steps, scores = got
            s = np.asarray(steps); v = np.asarray(scores)
            keep = s <= hi + 1e5
            if keep.sum() < 3 or s[keep].max() < 0.6 * hi:
                continue
            tail = grid > s[keep].max() + 2e6
            c = np.interp(grid, s[keep], v[keep], left=np.nan, right=v[keep][-1])
            c[tail] = np.nan
            curves.append(_ma(c, smooth))
        if not curves:
            continue
        C = np.vstack(curves)
        mean = np.nanmean(C, axis=0)
        sd = np.nanstd(C, axis=0)
        g = grid / 1e6
        ctrl = (m == "null")
        if len(curves) > 1 and not ctrl:
            ax.fill_between(g, mean - sd, mean + sd,
                            color=_lighten(PASTEL[m], 0.35), alpha=0.20,
                            linewidth=0, zorder=1)
        ax.plot(g, mean, color=PASTEL[m], lw=1.7 if ctrl else 2.6,
                ls=(0, (5, 2)) if ctrl else "-", label=m, zorder=2 if ctrl else 3)

    ax.set_xlim(0, horizon_m)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("training steps [M]", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK, fontsize=11)
    ax.set_title(body if not note else f"{body}\n{note}", color=INK,
                 fontsize=14 if not note else 10, pad=10)
    ax.tick_params(colors=INK, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color(INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15, color=INK)
    leg = ax.legend(loc="lower right", fontsize=10, frameon=False,
                    labelcolor=INK, handlelength=1.6)
    leg.set_title("mean ± 1 SD over seeds", prop={"size": 8})
    leg.get_title().set_color(INK)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(fname, dpi=160, transparent=True)
    plt.close(fig)
    print("wrote", fname)


def _curves_fig():
    """onlyVelRew (4 methods, 3 bodies) + full SCONE reward (2 methods, 2
    bodies). One transparent file per body per regime."""
    for body in ["h0918", "h1622", "h2190"]:
        md = {}
        for m in MORDER:
            sel = f"{O}/{_label_for(m, body)}.csv"
            if os.path.exists(sel):
                md[m] = [r["run_dir"] for r in csv.DictReader(open(sel))]
        _one_curve_panel(body, md, 20, 780 if body == "h2190" else 1060,
                         "test episode return",
                         f"{O}/learning_curve_{body}.png")
    for body, (md, hz, note) in _FULL_DIRS.items():
        _one_curve_panel(body, md, hz, 10500,
                         "test episode return  (full SCONE reward)",
                         f"{O}/learning_curve_full_{body}.png", note=note)


def _trend_fig(rows):
    """One cross-body view: return at t* and fall% vs body, a line per method.
    onlyVelRew only (the common ceiling comparison)."""
    ov = [r for r in rows if r["regime"] == "onlyVelRew"]
    bodies = ["h0918", "h1622", "h2190"]
    x = range(len(bodies))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in MORDER:
        ys, es, fs, xs = [], [], [], []
        for i, b in enumerate(bodies):
            r = next((r for r in ov if r["body"] == b and r["method"] == m), None)
            if not r:
                continue
            xs.append(i); ys.append(r["mean"]); es.append(r["sd"]); fs.append(r["fall_pct"])
        if not xs:
            continue
        axes[0].errorbar(xs, ys, yerr=es, marker="o", lw=2, capsize=3,
                         color=MCOL[m], label=m)
        axes[1].plot(xs, fs, marker="o", lw=2, color=MCOL[m], label=m)
    for ax in axes:
        ax.set_xticks(list(x)); ax.set_xticklabels(bodies)
        ax.set_xlabel("body  (increasing size / muscle redundancy →)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean rollout return @ t*  (N=100)")
    axes[0].set_title("Return vs body")
    axes[0].set_ylim(0, 1080)
    axes[1].set_ylabel("episodes ending in a fall  [%]")
    axes[1].set_title("Training/rollout instability vs body")
    axes[1].set_ylim(-3, 100)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Cross-body trend — locked checkpoint rule, onlyVelRew "
                 "(return @ t*, N=100)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = f"{O}/return_trend.png"
    fig.savefig(png, dpi=140)
    print("wrote", png)


if __name__ == "__main__":
    main()
