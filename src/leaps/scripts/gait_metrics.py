"""Consolidated gait-mechanics battery for the trained RL policies -- one row
per (body, method), pooled over the t* rollout episodes. Replaces the
scattered gait_scorecard.py + experimental_match.py.

Metrics
  spatiotemporal : speed [m/s], cadence [steps/min], stride time [s],
                   stance fraction, double-support fraction, flight fraction
                   (flight > 0 => running, not walking)
  kinematics     : hip / knee / ankle ROM [deg]  (per-cycle excursion)
  kinetics       : peak vertical GRF per leg [BW]; peak NET knee & ankle
                   joint moment [Nm/kg]  (sum of muscle contributions)
  robustness     : mean time-to-fall [s]; fall rate (< 95% of episode cap)
  coordination   : agonist-antagonist zero-lag activation correlation
                   (gastroc-TA, soleus-TA, vasti-hamstrings; < 0 = reciprocal,
                   physiological; ~0 / > 0 = co-contraction)
                   + mean activation (effort)
  symmetry       : L/R symmetry index  SI = 100*(R-L)/(0.5*(R+L))  (0 = perfect)
                   for step time, stance time, peak GRF, knee & ankle ROM;
                   plus a DEP-RL-style pelvis-vs-feet lateral asymmetry
                   (Schumacher et al. 2023 report gait symmetry as the
                   "averaged relative pelvis-deviation of the feet").

A 'human' row (treadmill ~1.2 m/s: literature values + the 22-subject EMG
pool for the A-A correlations) is written to the CSV and drawn as a reference
line on the figure.

    python -m leaps.scripts.gait_metrics
    python -m leaps.scripts.gait_metrics --bodies h1622

Outputs: results/gait_metrics/GAIT_METRICS.csv  +  GAIT_METRICS.png
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy import stats

from leaps.data.processing import time_normalize_stride
from leaps.scripts.biomech_fidelity import (RUNS, LIVE, MIN_ROWS, CKPT_PREF,
                                            read_sto, cross_human, load_tstar_map)

OUT = Path(__file__).resolve().parents[3] / "results" / "gait_metrics"
METHODS = {"real-LAP": "EMG prior", "mpo": "MPO", "dep-mpo": "DEP-MPO",
           "null": "null"}
MCOL = {"EMG prior": "#ff8a7a", "MPO": "#6fb8ff", "DEP-MPO": "#5fd39a",
        "null": "#8a8f98"}
INK = "#e8e8e8"

# muscle groups -> candidate model-muscle prefixes (lumped h0918/h1622 and
# un-lumped h2190 both covered).
GRP = {
    "gastroc": ["gastroc", "gas_med", "gas_lat"],
    "soleus": ["soleus"],
    "tib_ant": ["tib_ant"],
    "vasti": ["vasti", "vas_med", "vas_lat", "vas_int"],
    "hamstr": ["hamstrings", "bifemlh", "semimem", "semiten"],
}
# healthy treadmill walking, ~1.2 m/s (literature; A-A filled in from EMG pool)
HUMAN = dict(speed_ms=1.20, cadence_spm=112, stride_time_s=1.07,
             stance_frac=0.62, double_support_frac=0.12, flight_frac=0.0,
             rom_hip_deg=43, rom_knee_deg=60, rom_ankle_deg=28,
             peak_grf_R_bw=1.15, peak_grf_L_bw=1.15,
             peak_knee_moment_nmkg=0.45, peak_ankle_moment_nmkg=1.50,
             time_to_fall_s=np.nan, fall_rate=0.0, effort_meanact=0.10,
             si_steptime=0.0, si_stancetime=0.0, si_peakgrf=0.0,
             si_rom_knee=0.0, si_rom_ankle=0.0, pelvis_feet_asym=0.0)

COORD = {"hip": "hip_flexion", "knee": "knee_angle", "ankle": "ankle_angle"}


# --------------------------------------------------------------------- helpers
def _rising(mask):
    return np.where(mask[1:] & ~mask[:-1])[0] + 1


def _cycles(idx, lo, hi):
    return [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)
            if lo <= idx[i + 1] - idx[i] <= hi]


def _grp_signal(ep, group, leg):
    cols = [f"{p}_{leg}.activation" for pre in GRP[group] for p in [pre]
            if f"{pre}_{leg}.activation" in ep.columns]
    if not cols:
        return None
    return ep[cols].to_numpy().mean(axis=1)


def _norm_mean(sig, cyc):
    if not cyc:
        return None
    S = np.stack([time_normalize_stride(sig[a:b, None])[:, 0] for a, b in cyc])
    return S.mean(axis=0)


def _ckpt_dir(rd):
    root = LIVE.parent / rd if rd.startswith("not_used/") else LIVE / rd
    cks = sorted(root.glob("*/run_checkpoint_*"))
    if not cks:
        return None
    want = CKPT_PREF.get(rd.split("/")[1]) if "/" in rd else None
    pref = [c for c in cks if int(c.name.split("_")[-1]) == want] if want else []
    return pref[0] if pref else max(cks, key=lambda c: int(c.name.split("_")[-1]))


# --------------------------------------------------------------- per condition
def condition_metrics(body, run_dirs):
    min_gait = MIN_ROWS.get(body, 2500) * 0.4          # "enough gait to score"
    dur_all, dur_cap = [], []
    acc = {k: [] for k in ("speed", "cadence", "stride_t", "ds", "flight",
                           "effort", "aa_gas_ta", "aa_sol_ta", "aa_vas_ham",
                           "pelvis_feet_asym")}
    per_leg = {leg: {k: [] for k in ("stance_frac", "stance_t", "step_t",
                                     "grf_peak", "rom_hip", "rom_knee",
                                     "rom_ankle", "M_knee", "M_ankle")}
               for leg in ("r", "l")}

    for rd in run_dirs:
        ck = _ckpt_dir(rd)
        if ck is None:
            continue
        for sto in sorted(ck.glob("[0-9]*.sto")):
            ep = read_sto(sto)
            n = len(ep)
            t = ep["time"].to_numpy()
            dt = float(t[1] - t[0])
            dur = float(t[-1] - t[0])
            dur_all.append(dur)
            dur_cap.append(dur)
            if n < min_gait:
                continue
            gr = {"r": ep["leg1_r.grf_y"].to_numpy(),
                  "l": ep["leg0_l.grf_y"].to_numpy()}
            core = slice(int(0.05 * n), int(0.95 * n))
            bw = float(np.mean(gr["r"][core] + gr["l"][core]))
            if bw < 1e-6:
                continue
            mass = bw / 9.81
            thr = 0.05 * bw
            contact = {lg: gr[lg] > thr for lg in "rl"}
            acc["ds"].append(float(np.mean(contact["r"] & contact["l"])))
            acc["flight"].append(float(np.mean(~contact["r"] & ~contact["l"])))
            act_cols = [c for c in ep.columns if c.endswith(".activation")]
            acc["effort"].append(float(np.mean(ep[act_cols].to_numpy()[core])))

            x = ep["pelvis.com_pos_x"].to_numpy()
            acc["speed"].append((x[core][-1] - x[core][0]) / (t[core][-1] - t[core][0]))

            # DEP-style: pelvis lateral position relative to the feet midline,
            # / half the stance width. 0 = centred (symmetric), ->1 = one-sided.
            # Skipped for a sagittal-plane model (h0918 has no lateral DoF).
            pz = ep.get("pelvis.com_pos_z")
            if pz is not None and np.ptp(pz.to_numpy()[core]) > 1e-3 \
                    and {"calcn_r.com_pos_z", "calcn_l.com_pos_z"}.issubset(ep.columns):
                fr = ep["calcn_r.com_pos_z"].to_numpy()[core] - pz.to_numpy()[core]
                fl = ep["calcn_l.com_pos_z"].to_numpy()[core] - pz.to_numpy()[core]
                width = np.mean(np.abs(fr - fl)) / 2
                if width > 1e-4:
                    acc["pelvis_feet_asym"].append(float(np.mean(np.abs(fr + fl)) / (2 * width)))

            lo, hi = int(0.6 / dt), int(1.8 / dt)
            # heel strikes, de-bounced (drop edges < 0.4 s apart = GRF ripple)
            def _hs(lg):
                idx = _rising(contact[lg])
                if len(idx) == 0:
                    return idx
                keep = [idx[0]]
                for i in idx[1:]:
                    if i - keep[-1] >= int(0.4 / dt):
                        keep.append(i)
                return np.array(keep)
            hs = {lg: _hs(lg) for lg in "rl"}

            rc = _cycles(hs["r"], lo, hi)      # valid right strides (0.6-1.8 s)
            if rc:
                st = np.mean([b - a for a, b in rc]) * dt
                acc["stride_t"].append(float(st))
                acc["cadence"].append(120.0 / st)

            # step time R = R-HS -> next L-HS, and vice versa
            for a, b in (("r", "l"), ("l", "r")):
                sts = [(hs[b][hs[b] > h][0] - h) * dt for h in hs[a]
                       if np.any(hs[b] > h)]
                if sts:
                    per_leg[a]["step_t"].append(float(np.mean(sts)))

            for lg in "rl":
                per_leg[lg]["stance_frac"].append(float(np.mean(contact[lg])))
                # stance duration per cycle
                cyc = _cycles(hs[lg], lo, hi)
                if cyc:
                    st = [np.sum(contact[lg][a:b]) * dt for a, b in cyc]
                    per_leg[lg]["stance_t"].append(float(np.mean(st)))
                    per_leg[lg]["grf_peak"].append(
                        float(np.mean([gr[lg][a:b].max() for a, b in cyc])) / bw)
                    for j, coord in COORD.items():
                        ang = np.degrees(ep[f"{coord}_{lg}"].to_numpy())
                        rom = np.mean([np.ptp(ang[a:b]) for a, b in cyc])
                        per_leg[lg][f"rom_{j}"].append(float(rom))
                    for jm, cd in (("knee", "knee_angle"), ("ankle", "ankle_angle")):
                        mcols = [c for c in ep.columns
                                 if re.match(rf".+_{lg}\.{cd}_{lg}\.moment$", c)]
                        if mcols:
                            net = ep[mcols].to_numpy().sum(axis=1)
                            pk = np.mean([np.abs(net[a:b]).max() for a, b in cyc])
                            per_leg[lg][f"M_{jm}"].append(float(pk) / mass)

            # A-A on right-leg cycles
            mc = {}
            for g in GRP:
                s = _grp_signal(ep, g, "r")
                nm = _norm_mean(s, rc) if s is not None else None
                if nm is not None:
                    mc[g] = nm

            def _r(a, c):
                if a in mc and c in mc:
                    return float(stats.pearsonr(mc[a], mc[c])[0])
                return None
            for key, (a, c) in (("aa_gas_ta", ("gastroc", "tib_ant")),
                                ("aa_sol_ta", ("soleus", "tib_ant")),
                                ("aa_vas_ham", ("vasti", "hamstr"))):
                v = _r(a, c)
                if v is not None:
                    acc[key].append(v)

    if not dur_all:
        return None
    cap = max(dur_cap) if dur_cap else 25.0
    m = lambda xs: float(np.mean(xs)) if xs else np.nan          # noqa: E731

    def si(rv, lv):
        r, l = m(per_leg["r"][rv]), m(per_leg["l"][lv if lv else rv])
        if np.isnan(r) or np.isnan(l) or abs(r + l) < 1e-9:
            return np.nan
        return 100.0 * (r - l) / (0.5 * (r + l))

    both = lambda k: m(per_leg["r"][k] + per_leg["l"][k])        # noqa: E731
    return dict(
        n_ep=len(dur_all),
        time_to_fall_s=round(m(dur_all), 2),
        fall_rate=round(float(np.mean([d < 0.95 * cap for d in dur_all])), 2),
        speed_ms=round(m(acc["speed"]), 2),
        cadence_spm=round(m(acc["cadence"]), 1),
        stride_time_s=round(m(acc["stride_t"]), 3),
        stance_frac=round(both("stance_frac"), 3),
        double_support_frac=round(m(acc["ds"]), 3),
        flight_frac=round(m(acc["flight"]), 3),
        rom_hip_deg=round(both("rom_hip"), 1),
        rom_knee_deg=round(both("rom_knee"), 1),
        rom_ankle_deg=round(both("rom_ankle"), 1),
        peak_grf_R_bw=round(m(per_leg["r"]["grf_peak"]), 2),
        peak_grf_L_bw=round(m(per_leg["l"]["grf_peak"]), 2),
        peak_knee_moment_nmkg=round(both("M_knee"), 2),
        peak_ankle_moment_nmkg=round(both("M_ankle"), 2),
        effort_meanact=round(m(acc["effort"]), 3),
        aa_gas_ta=round(m(acc["aa_gas_ta"]), 2),
        aa_sol_ta=round(m(acc["aa_sol_ta"]), 2),
        aa_vas_ham=round(m(acc["aa_vas_ham"]), 2),
        si_steptime=round(si("step_t", "step_t"), 1),
        si_stancetime=round(si("stance_t", "stance_t"), 1),
        si_peakgrf=round(si("grf_peak", "grf_peak"), 1),
        si_rom_knee=round(si("rom_knee", "rom_knee"), 1),
        si_rom_ankle=round(si("rom_ankle", "rom_ankle"), 1),
        pelvis_feet_asym=round(m(acc["pelvis_feet_asym"]), 3),
    )


def human_aa():
    """gastroc-TA, soleus-TA, vasti-hamstrings zero-lag r from the 22-subject
    EMG pool (per-subject, then averaged)."""
    chn, sub_means, ch_idx, _, _ = cross_human()
    def pair(a, b):
        rs = []
        for s in sub_means:
            wa = sub_means[s][:, ch_idx[a]]
            wb = np.mean([sub_means[s][:, ch_idx[c]] for c in b], axis=0)
            rs.append(stats.pearsonr(wa, wb)[0])
        return round(float(np.mean(rs)), 2)
    return dict(aa_gas_ta=pair("gastrocmed", ["tibialisanterior"]),
                aa_sol_ta=pair("soleus", ["tibialisanterior"]),
                aa_vas_ham=pair("vastuslateralis",
                                ["bicepsfemoris", "semitendinosus"]))


# --------------------------------------------------------------------- figure
FIG_METRICS = [
    ("rom_hip_deg", "hip ROM [deg]"), ("rom_knee_deg", "knee ROM [deg]"),
    ("rom_ankle_deg", "ankle ROM [deg]"), ("cadence_spm", "cadence [/min]"),
    ("peak_grf_R_bw", "peak GRF R [BW]"), ("peak_grf_L_bw", "peak GRF L [BW]"),
    ("peak_knee_moment_nmkg", "peak knee moment [Nm/kg]"),
    ("peak_ankle_moment_nmkg", "peak ankle moment [Nm/kg]"),
    ("time_to_fall_s", "time to fall [s]"),
    ("aa_vas_ham", "vasti-hamstr corr"), ("aa_gas_ta", "gastroc-TA corr"),
    ("si_peakgrf", "GRF symmetry index [%]"),
]


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bodies = sorted({r["body"] for r in rows if r["body"] != "human"})
    order = ["EMG prior", "MPO", "DEP-MPO", "null"]
    hu = next((r for r in rows if r["body"] == "human"), {})
    nrow = len(FIG_METRICS)
    fig, axes = plt.subplots(nrow, len(bodies), figsize=(3.1 * len(bodies),
                             1.8 * nrow), squeeze=False)
    for ri, (key, label) in enumerate(FIG_METRICS):
        for ci, body in enumerate(bodies):
            ax = axes[ri, ci]
            br = {r["method"]: r for r in rows if r["body"] == body}
            for i, mth in enumerate(order):
                if mth in br and not np.isnan(br[mth].get(key, np.nan)):
                    ax.bar(i, br[mth][key], color=MCOL[mth], width=0.7)
            hv = hu.get(key, np.nan)
            if hv is not None and not (isinstance(hv, float) and np.isnan(hv)):
                ax.axhline(hv, color=INK, ls="--", lw=1)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order if ri == nrow - 1 else [], rotation=35,
                               ha="right", fontsize=7.5, color=INK)
            ax.tick_params(colors=INK, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(INK)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.15, color=INK)
            if ri == 0:
                ax.set_title(body, color=INK, fontsize=12, pad=8)
            if ci == 0:
                ax.set_ylabel(label, color=INK, fontsize=8.5)
        fig.patch.set_alpha(0)
    for ax in axes.flat:
        ax.patch.set_alpha(0)
    fig.suptitle("Gait-mechanics battery at t*  (dashed = healthy-human "
                 "reference)", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    png = OUT / "GAIT_METRICS.png"
    fig.savefig(png, dpi=150, transparent=True)
    print("wrote", png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", nargs="+", default=["h0918", "h1622", "h2190"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    load_tstar_map()  # warm the checkpoint-selection cache / validate paths

    rows = []
    for body in a.bodies:
        for cond, mlabel in METHODS.items():
            if cond not in RUNS.get(body, {}):
                continue
            r = condition_metrics(body, RUNS[body][cond])
            if r is None:
                print(f"  {body} {mlabel}: no rollouts")
                continue
            rows.append(dict(body=body, method=mlabel, **r))
            print(f"  {body:6s} {mlabel:9s}: speed {r['speed_ms']}  "
                  f"cad {r['cadence_spm']}  ROM h/k/a "
                  f"{r['rom_hip_deg']}/{r['rom_knee_deg']}/{r['rom_ankle_deg']}  "
                  f"GRF {r['peak_grf_R_bw']}/{r['peak_grf_L_bw']}  "
                  f"Mk/Ma {r['peak_knee_moment_nmkg']}/{r['peak_ankle_moment_nmkg']}  "
                  f"ttf {r['time_to_fall_s']}s fall {r['fall_rate']}  "
                  f"AA vh/gt {r['aa_vas_ham']}/{r['aa_gas_ta']}  "
                  f"SI grf {r['si_peakgrf']}  pf-asym {r['pelvis_feet_asym']}")

    rows.append(dict(body="human", method="reference", n_ep=0, **HUMAN,
                     **{k: v for k, v in human_aa().items()}))

    fields = list(rows[0])
    for r in rows:
        for f in fields:
            r.setdefault(f, "")
    csvp = OUT / "GAIT_METRICS.csv"
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", csvp)
    make_figure(rows)


if __name__ == "__main__":
    main()
