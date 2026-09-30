"""Torso oscillation figure, Schumacher et al. 2023 (arXiv:2309.02976) Fig 4
REPLICA: fetched the actual PDF (page 6) on 2026-09-14 rather than going off
the caption text alone -- their Fig 4 is raw time series, NOT a phase-locked
gait-cycle average: one panel per model, ~5 individual 10s rollout traces
(thin overlaid lines, no mean/SD band), dashed 0deg straight-posture line,
x=time[s] 0-10, y=torso angle from vertical [deg]. This version replaces an
earlier gait-cycle-normalized mean+-SD draft that didn't match that style.

Checked directly against their paper: this figure uses NO human reference
data -- straight-posture (0deg) baseline + model-vs-model comparison only
(their H2190 vs MyoLeg). We follow the same precedent: no human band here,
one panel per condition instead of per model (mpo/dep-mpo/w=0.0/w=0.1/w=0.5),
against the same 0deg dashed line.

Two axes, both from torso.ori_{x,z} (SCONE trunk orientation, rad ->deg):
  sagittal (torso.ori_z, r=0.97 vs pelvis_tilt) -- all 3 bodies. This is the
    metric plotted (their "torso angle with the vertical axis" isn't
    formally defined in the text; sagittal trunk lean is the standard gait
    convention and the only one that exists on all 3 of our bodies).
  lateral  (torso.ori_x, r=0.96 vs pelvis_list) -- h1622/h2190 only; h0918 is
    the 2D model (no frontal-plane DOF), so its torso.ori_x is identically 0
    and it's excluded from that panel rather than plotted as a fake flat 0.
    This is our extra column -- their Fig 4's own text singles out lateral
    oscillation as the interesting MyoLeg artifact, so it's worth showing
    separately rather than folding it into one blended angle.
Axis identities verified empirically (correlation against the named pelvis
DOF) on 2026-09-14, not assumed from column-name order.

A separate gait-cycle-normalized summary table (peak-to-peak/RMS/mean-offset
per condition) is still written to TORSO_OSC.csv -- useful quantification,
independent of the plot-style fix.

--style cycle (2026-09-18, added on request): plots that same
gait-cycle-normalized mean+-SD band (x=0-100% stride, already computed by
pool_torso for the CSV, previously discarded after being reduced to the 3
summary scalars) instead of the raw 10s traces. Not a paper replica -- for
h1622 specifically, where w<=0.1 is a stiff-knee bounce rather than a
normal gait, this reads far more clearly: a stable walk gives a narrow
repeatable band, the bounce gives either a much larger amplitude or a wide
ragged band, visible at a glance instead of squinting at 5 overlaid noisy
raw lines. Written to TORSO_OSC_CYCLE_<body>.png, separate from the raw
Fig-4-replica file so neither overwrites the other.

    python -m leaps.analysis.torso_oscillation
    python -m leaps.analysis.torso_oscillation --body h2190
    python -m leaps.analysis.torso_oscillation --body h1622 --style cycle
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leaps.data import time_normalize_stride
from leaps.scripts.biomech_fidelity import RUNS, LIVE, read_sto, load_tstar_map, CKPT_PREF, MIN_ROWS
from leaps.paths import RESULTS_DIR

METHODS = {
    "mpo": "MPO", "dep-mpo": "DEP-MPO",
    "real-LAP-w00": "EMG prior w=0.0", "real-LAP": "EMG prior w=0.1",
    "real-LAP-w01-seed0": "EMG prior w=0.1 (seed0, walking)", "real-LAP-w05": "EMG prior w=0.5",
}
MCOL = {
    "MPO": "#e08214", "DEP-MPO": "#b35806",
    "EMG prior w=0.0": "#9ecae1", "EMG prior w=0.1": "#3182bd",
    "EMG prior w=0.1 (seed0, walking)": "#3182bd", "EMG prior w=0.5": "#08519c",
}
OUT = RESULTS_DIR / "biomech_fidelity"
LATERAL_BODIES = ("h1622", "h2190")
N_TRACES = 5
T_MAX = 10.0


def seg_cycles(grf, dt):
    on = grf > 0.05 * np.nanmax(grf)
    rise_raw = np.where(on[1:] & ~on[:-1])[0] + 1
    if len(rise_raw) == 0:
        return []
    # De-bounce: drop edges <0.4s apart (GRF ripple within one stance phase,
    # not a real new cycle) -- same fix gait_metrics.py's _hs() already has;
    # without it, a genuine ~0.78s stride gets fragmented into two ~0.4s
    # pieces that then fail the 0.6-1.8s stride-length filter below,
    # silently zeroing out otherwise-valid gait (found on h1622 w=0.0).
    rise = [rise_raw[0]]
    for i in rise_raw[1:]:
        if i - rise[-1] >= int(0.4 / dt):
            rise.append(i)
    lo, hi = int(0.6 / dt), int(1.8 / dt)
    return [(rise[i], rise[i + 1]) for i in range(len(rise) - 1)
            if lo <= rise[i + 1] - rise[i] <= hi]


def pool_torso(run_dirs, tmap, body, min_rows):
    sag, lat = [], []
    n_ep = n_fall = 0
    for rd in run_dirs:
        root = LIVE.parent / rd if rd.startswith("not_used/") else LIVE / rd
        want = tmap.get(rd)
        cks = sorted(root.glob("*/run_checkpoint_*"))
        if not cks:
            continue
        pref = [c for c in cks if int(c.name.split("_")[-1]) == want] if want else []
        ck = pref[0] if pref else max(cks, key=lambda c: int(c.name.split("_")[-1]))
        for sto in sorted(ck.glob("*.sto")):
            ep = read_sto(sto)
            if len(ep) < min_rows:
                n_fall += 1
                continue
            n_ep += 1
            dt = float(ep["time"].iloc[1] - ep["time"].iloc[0])
            gcol = "leg1_r.grf_y" if "leg1_r.grf_y" in ep.columns else "leg0_r.grf_y"
            g = ep[gcol].to_numpy()
            for a, b in seg_cycles(g, dt):
                z = np.degrees(ep["torso.ori_z"].to_numpy()[a:b])
                sag.append(time_normalize_stride(z[:, None])[:, 0])
                if body in LATERAL_BODIES:
                    x = np.degrees(ep["torso.ori_x"].to_numpy()[a:b])
                    lat.append(time_normalize_stride(x[:, None])[:, 0])
    if n_ep == 0 or not sag:
        return None
    S = np.stack(sag)
    out = dict(sag_mean=S.mean(0), sag_sd=S.std(0), n_ep=n_ep,
               fall=n_fall / (n_ep + n_fall))
    if lat:
        L = np.stack(lat)
        out["lat_mean"], out["lat_sd"] = L.mean(0), L.std(0)
    return out


def load_raw_traces(run_dirs, tmap, body, min_rows, n=N_TRACES, tmax=T_MAX, drop_falls=False):
    """Up to n individual-episode (t, sag_deg, lat_deg_or_None) traces, raw
    time series (no gait-cycle segmentation/normalization) -- matches Fig 4's
    literal '5 rollouts of 10s' presentation. Falls truncate the trace early
    (kept, not discarded -- that's real behavior, not missing data), UNLESS
    drop_falls=True, in which case falls (< min_rows, same cutoff pool_torso
    uses) are excluded entirely rather than shown truncated -- for when a
    figure is deliberately restricted to a "this seed walks" condition and a
    visible fall in the trace would contradict that framing."""
    traces = []
    for rd in run_dirs:
        root = LIVE.parent / rd if rd.startswith("not_used/") else LIVE / rd
        want = tmap.get(rd)
        cks = sorted(root.glob("*/run_checkpoint_*"))
        if not cks:
            continue
        pref = [c for c in cks if int(c.name.split("_")[-1]) == want] if want else []
        ck = pref[0] if pref else max(cks, key=lambda c: int(c.name.split("_")[-1]))
        for sto in sorted(ck.glob("*.sto")):
            if len(traces) >= n:
                break
            ep = read_sto(sto)
            cutoff = min_rows if drop_falls else min(min_rows, 50)
            if len(ep) < cutoff:  # keep short falls, drop near-instant ones (unless drop_falls)
                continue
            t = ep["time"].to_numpy()
            keep = t <= tmax
            sag = np.degrees(ep["torso.ori_z"].to_numpy()[keep])
            lat = np.degrees(ep["torso.ori_x"].to_numpy()[keep]) if body in LATERAL_BODIES else None
            traces.append((t[keep], sag, lat))
        if len(traces) >= n:
            break
    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=None, choices=["h0918", "h1622", "h2190"])
    ap.add_argument("--style", default="raw", choices=["raw", "cycle"])
    ap.add_argument("--conditions", default=None,
                     help="comma-separated subset of METHODS keys to plot, e.g. "
                          "mpo,dep-mpo,real-LAP-w01-seed0,real-LAP-w05 (default: all)")
    ap.add_argument("--drop-falls", action="store_true",
                     help="raw style only: exclude fall episodes entirely instead of "
                          "showing them truncated (see load_raw_traces docstring)")
    ap.add_argument("--dark", action="store_true",
                     help="cycle style only: dark/transparent deck styling, matching "
                          "biomech_fidelity.py's f2(--dark)")
    ap.add_argument("--separate", action="store_true",
                     help="cycle style only: write sagittal and lateral as two separate "
                          "figures instead of one figure with two columns")
    ap.add_argument("--overlay", action="store_true",
                     help="cycle style only: all conditions on ONE shared axis per figure "
                          "(with a legend) instead of one stacked row per condition")
    args = ap.parse_args()
    methods = ({k: METHODS[k] for k in args.conditions.split(",")}
               if args.conditions else METHODS)
    suffix = f"_{args.conditions}" if args.conditions else ""

    tmap = load_tstar_map()
    bodies = [args.body] if args.body else ["h0918", "h1622", "h2190"]
    x = np.linspace(0, 100, 101)
    rows = []

    for body in bodies:
        has_lat = body in LATERAL_BODIES
        cycle_results = {}
        raw_results = {}
        for cond, label in methods.items():
            rd = RUNS.get(body, {}).get(cond)
            if not rd:
                continue
            r = pool_torso(rd, tmap, body, MIN_ROWS.get(body, 2500))
            if r is None:
                print(f"  !! {body} {cond}: no usable cycles, skipped")
                continue
            cycle_results[label] = r
            rows.append(dict(
                body=body, method=label,
                sag_ptp=float(r["sag_mean"].ptp()), sag_rms=float(np.sqrt((r["sag_mean"] ** 2).mean())),
                sag_offset=float(r["sag_mean"].mean()),
                lat_ptp=float(r["lat_mean"].ptp()) if "lat_mean" in r else None,
                lat_rms=float(np.sqrt((r["lat_mean"] ** 2).mean())) if "lat_mean" in r else None,
                lat_offset=float(r["lat_mean"].mean()) if "lat_mean" in r else None,
                n_ep=r["n_ep"], fall=r["fall"],
            ))
            if args.style == "raw":
                raw_results[label] = load_raw_traces(rd, tmap, body, MIN_ROWS.get(body, 2500),
                                                       drop_falls=args.drop_falls)

        if args.style == "cycle":
            labels = [lb for lb in cycle_results]
            if not labels:
                continue
            INK, INK_MUTED = "#e8e8e8", "#9aa0a6"

            def style_axis(ax, title, is_dark):
                ax.axhline(0, ls="--", c=(INK_MUTED if is_dark else "k"), lw=1, alpha=0.6)
                ax.set_title(title, color=(INK if is_dark else "black"))
                if is_dark:
                    ax.set_facecolor("none")
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.spines["bottom"].set_visible(True)
                    ax.spines["left"].set_visible(True)
                    ax.spines["bottom"].set_color(INK_MUTED)
                    ax.spines["left"].set_color(INK_MUTED)
                    ax.tick_params(colors=INK_MUTED, labelsize=9)

            def make_fig(axis_key, sd_key, axis_name):
                fig, axes = plt.subplots(len(labels), 1, figsize=(5.5, 2.0 * len(labels)),
                                          squeeze=False, sharex=True)
                if args.dark:
                    fig.patch.set_alpha(0)
                any_data = False
                for i, label in enumerate(labels):
                    ax = axes[i][0]
                    r = cycle_results[label]
                    if axis_key not in r:
                        ax.axis("off")
                        continue
                    any_data = True
                    c = MCOL[label]
                    ax.plot(x, r[axis_key], color=c, lw=1.3)
                    ax.fill_between(x, r[axis_key] - r[sd_key], r[axis_key] + r[sd_key],
                                     color=c, alpha=0.25, lw=0)
                    style_axis(ax, "", args.dark)
                    ax.set_ylabel(f"{label}\ntorso ang. [deg]", fontsize=8,
                                  color=(INK if args.dark else "black"))
                    if args.dark:
                        ax.yaxis.label.set_color(INK)
                if not any_data:
                    plt.close(fig)
                    return None
                axes[0][0].set_title(f"{body}: {axis_name} (mean ± SD over strides)",
                                      color=(INK if args.dark else "black"))
                axes[-1][0].set_xlabel("% gait cycle", color=(INK if args.dark else "black"))
                axes[-1][0].set_xlim(0, 100)
                fig.tight_layout()
                return fig

            def make_overlay_fig(axis_key, sd_key, axis_name):
                fig, ax = plt.subplots(figsize=(6.0, 4.2))
                if args.dark:
                    fig.patch.set_alpha(0)
                any_data = False
                for label in labels:
                    r = cycle_results[label]
                    if axis_key not in r:
                        continue
                    any_data = True
                    c = MCOL[label]
                    ax.plot(x, r[axis_key], color=c, lw=1.6, label=label)
                    ax.fill_between(x, r[axis_key] - r[sd_key], r[axis_key] + r[sd_key],
                                     color=c, alpha=0.15, lw=0)
                if not any_data:
                    plt.close(fig)
                    return None
                style_axis(ax, f"{body}: {axis_name} (mean ± SD over strides)", args.dark)
                ax.set_xlabel("% gait cycle", color=(INK if args.dark else "black"))
                ax.set_ylabel("torso ang. [deg]", color=(INK if args.dark else "black"))
                ax.set_xlim(0, 100)
                leg = ax.legend(fontsize=9, loc="best",
                                 labelcolor=(INK if args.dark else None),
                                 framealpha=(0.0 if args.dark else None))
                fig.tight_layout()
                return fig

            dark_tag = "_dark" if args.dark else ""
            if args.overlay:
                for axis_key, sd_key, axis_name in [("sag_mean", "sag_sd", "sagittal"),
                                                     ("lat_mean", "lat_sd", "lateral")]:
                    if axis_key == "lat_mean" and not has_lat:
                        continue
                    fig = make_overlay_fig(axis_key, sd_key, axis_name)
                    if fig is None:
                        continue
                    outp = OUT / f"TORSO_OSC_OVERLAY_{body}_{axis_name}{suffix}{dark_tag}.png"
                    fig.savefig(outp, dpi=150, transparent=args.dark)
                    plt.close(fig)
                    print(f"wrote {outp}")
                continue

            if args.separate:
                for axis_key, sd_key, axis_name in [("sag_mean", "sag_sd", "sagittal"),
                                                     ("lat_mean", "lat_sd", "lateral")]:
                    if axis_key == "lat_mean" and not has_lat:
                        continue
                    fig = make_fig(axis_key, sd_key, axis_name)
                    if fig is None:
                        continue
                    outp = OUT / f"TORSO_OSC_CYCLE_{body}_{axis_name}{suffix}{dark_tag}.png"
                    fig.savefig(outp, dpi=150, transparent=args.dark)
                    plt.close(fig)
                    print(f"wrote {outp}")
                continue

            # combined sagittal+lateral, one figure, two columns (original behavior)
            ncols = 2 if has_lat else 1
            fig, axes = plt.subplots(len(labels), ncols, figsize=(5.0 * ncols, 2.0 * len(labels)),
                                      squeeze=False, sharex=True)
            for i, label in enumerate(labels):
                c = MCOL[label]
                r = cycle_results[label]
                axes[i][0].plot(x, r["sag_mean"], color=c, lw=1.3)
                axes[i][0].fill_between(x, r["sag_mean"] - r["sag_sd"], r["sag_mean"] + r["sag_sd"],
                                         color=c, alpha=0.25, lw=0)
                axes[i][0].axhline(0, ls="--", c="k", lw=1, alpha=0.6)
                axes[i][0].set_ylabel(f"{label}\ntorso ang. [deg]", fontsize=8)
                if has_lat and "lat_mean" in r:
                    axes[i][1].plot(x, r["lat_mean"], color=c, lw=1.3)
                    axes[i][1].fill_between(x, r["lat_mean"] - r["lat_sd"], r["lat_mean"] + r["lat_sd"],
                                             color=c, alpha=0.25, lw=0)
                    axes[i][1].axhline(0, ls="--", c="k", lw=1, alpha=0.6)
                if i == 0:
                    axes[i][0].set_title(f"{body}: sagittal (mean ± SD over strides)")
                    if has_lat:
                        axes[i][1].set_title(f"{body}: lateral (mean ± SD over strides)")
            axes[-1][0].set_xlabel("% gait cycle")
            axes[-1][0].set_xlim(0, 100)
            if has_lat:
                axes[-1][1].set_xlabel("% gait cycle")
            fig.tight_layout()
            outp = OUT / f"TORSO_OSC_CYCLE_{body}{suffix}.png"
            fig.savefig(outp, dpi=150)
            plt.close(fig)
            print(f"wrote {outp}")
            continue

        labels = [lb for lb in raw_results if raw_results[lb]]
        if not labels:
            continue
        ncols = 2 if has_lat else 1
        fig, axes = plt.subplots(len(labels), ncols, figsize=(5.0 * ncols, 2.0 * len(labels)),
                                  squeeze=False, sharex=True)
        for i, label in enumerate(labels):
            c = MCOL[label]
            for t, sag, lat in raw_results[label]:
                axes[i][0].plot(t, sag, color=c, lw=0.9, alpha=0.7)
            axes[i][0].axhline(0, ls="--", c="k", lw=1, alpha=0.6)
            axes[i][0].set_ylabel(f"{label}\ntorso ang. [deg]", fontsize=8)
            if has_lat:
                for t, sag, lat in raw_results[label]:
                    axes[i][1].plot(t, lat, color=c, lw=0.9, alpha=0.7)
                axes[i][1].axhline(0, ls="--", c="k", lw=1, alpha=0.6)
            if i == 0:
                axes[i][0].set_title(f"{body}: sagittal")
                if has_lat:
                    axes[i][1].set_title(f"{body}: lateral")
        axes[-1][0].set_xlabel("time [s]")
        axes[-1][0].set_xlim(0, T_MAX)
        if has_lat:
            axes[-1][1].set_xlabel("time [s]")
        fig.tight_layout()
        outp = OUT / f"TORSO_OSC_{body}{suffix}.png"
        fig.savefig(outp, dpi=150)
        plt.close(fig)
        print(f"wrote {outp}")

    if args.body is None and args.conditions is None:
        csvp = OUT / "TORSO_OSC.csv"
        import csv as csvmod
        with open(csvp, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {csvp}")
    else:
        print("(partial --body/--conditions run -- not overwriting the shared TORSO_OSC.csv)")

    print(f"\n{'body':6} {'method':18} {'sagPTP':>7} {'sagRMS':>7} {'sagOff':>7} "
          f"{'latPTP':>7} {'latRMS':>7} {'latOff':>7}  {'n_ep':>4} {'fall':>5}")
    for r in rows:
        def f(v):
            return f"{v:7.2f}" if v is not None else "     --"
        print(f"{r['body']:6} {r['method']:18} {f(r['sag_ptp'])} {f(r['sag_rms'])} {f(r['sag_offset'])} "
              f"{f(r['lat_ptp'])} {f(r['lat_rms'])} {f(r['lat_offset'])}  {r['n_ep']:4d} {r['fall']:5.2f}")


if __name__ == "__main__":
    main()
