"""Experimental-match figure + table, Park/Hausdorfer "Natural walking" style.

experimental_match = fraction of the gait cycle for which the mean simulated
joint-angle trajectory lies within +/-1 SD of the experimental human band
(hip, knee, ankle; both legs pooled). Reference band = 6 morphology-matched
Camargo subjects (AB06/08/10/11/23/30), treadmill walking, pooled gait cycles.

Also plots joint angles + joint velocities + vertical GRF over the cycle,
one line per method, human band shaded -- the paper's Figure 2.

    python -m leaps.scripts.kinematic_match
    python -m leaps.scripts.kinematic_match --speed 1.35 --tol 0.25
"""
import argparse
import glob
import os
from pathlib import Path

os.environ.setdefault("LEAPS_EMG_H5",
                      "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leaps.data.loaders import load_mat_table
from leaps.data.processing import find_stride_intervals, time_normalize_stride
from leaps.data.metadata import SUBJECTS
from leaps.scripts.biomech_fidelity import RUNS, load_tstar_map, LIVE, read_sto

SUBJ = ["AB06", "AB08", "AB10", "AB11", "AB23", "AB30"]
JOINTS = ["hip", "knee", "ankle"]
GON = {"hip": "hip_sagittal", "knee": "knee_sagittal", "ankle": "ankle_sagittal"}
# Camargo gon conventions (verified against raw data 2026-09-08): hip flexion +,
# knee flexion -, ankle dorsiflexion + -- all three match SCONE's native signs,
# so no flips. gon has a subject-specific DC offset (relative goniometers), so
# the match metric aligns each simulated mean curve to the human mean by
# removing the per-joint cycle-mean difference (compares shape/phase, not the
# goniometer's absolute zero); raw offsets reported separately.
SCONE_SIGN = {"hip": 1.0, "knee": 1.0, "ankle": 1.0}
ALIGN_OFFSET = True
METHODS = {"real-LAP": "EMG prior", "mpo": "MPO", "dep-mpo": "DEP-MPO", "null": "null"}
MCOL = {"EMG prior": "#d62728", "MPO": "#1f77b4", "DEP-MPO": "#2ca02c", "null": "#7f7f7f"}
OUT = Path("/home/nadinebadie/lalitha/LEAPS/results/biomech_fidelity")
CACHE = OUT / "human_gait_band.npz"


# ----------------------------------------------------------------- human band
def build_human_band(speed, tol, force=False):
    if CACHE.exists() and not force:
        d = np.load(CACHE)
        if abs(float(d["speed"]) - speed) < 1e-6 and abs(float(d["tol"]) - tol) < 1e-6:
            return {k: d[k] for k in d.files}
    ang = {j: [] for j in JOINTS}
    grf = []
    for s in SUBJ:
        base = glob.glob(f"/home/nadinebadie/lalitha/datasets/camargo/{s}/*/treadmill")[0]
        bw = SUBJECTS[s].mass * 9.81
        for gon_f in sorted(glob.glob(f"{base}/gon/*.mat")):
            stem = Path(gon_f).name
            try:
                g = load_mat_table(gon_f)
                cnd = load_mat_table(f"{base}/conditions/{stem}")
                gc = load_mat_table(f"{base}/gcRight/{stem}")
                fp = load_mat_table(f"{base}/fp/{stem}")
            except Exception:
                continue
            iv = find_stride_intervals(gc["Header"], gc["HeelStrike"])
            for t0, t1 in iv:
                sp = cnd["Speed"][(cnd["Header"] >= t0) & (cnd["Header"] < t1)]
                if len(sp) == 0 or abs(float(np.mean(sp)) - speed) > tol:
                    continue
                m = (g["Header"] >= t0) & (g["Header"] < t1)
                if m.sum() < 20:
                    continue
                for j in JOINTS:
                    ang[j].append(time_normalize_stride(g[GON[j]][m][:, None])[:, 0])
                mf = (fp["Header"] >= t0) & (fp["Header"] < t1)
                grf.append(time_normalize_stride(
                    (fp["Treadmill_R_vy"][mf] / bw)[:, None])[:, 0])
    out = {"speed": speed, "tol": tol}
    for j in JOINTS:
        A = np.stack(ang[j])
        out[f"{j}_mean"], out[f"{j}_sd"], out[f"{j}_n"] = A.mean(0), A.std(0), len(A)
    G = np.stack(grf)
    out["grf_mean"], out["grf_sd"], out["grf_n"] = G.mean(0), G.std(0), len(G)
    np.savez(CACHE, **out)
    return out


# -------------------------------------------------------------------- sim side
def seg_cycles(grf, dt):
    on = grf > 0.05 * np.nanmax(grf)
    rise = np.where(on[1:] & ~on[:-1])[0] + 1
    lo, hi = int(0.6 / dt), int(1.8 / dt)
    return [(rise[i], rise[i + 1]) for i in range(len(rise) - 1)
            if lo <= rise[i + 1] - rise[i] <= hi]


def pool_sim(run_dirs, tmap):
    ang = {j: [] for j in JOINTS}
    vel = {j: [] for j in JOINTS}
    grf = []
    n_ep = n_fall = 0
    for rd in run_dirs:
        root = LIVE / rd
        want = tmap.get(rd)
        cks = sorted(root.glob("*/run_checkpoint_*"))
        if not cks:
            continue
        pref = [c for c in cks if int(c.name.split("_")[-1]) == want] if want else []
        ck = pref[0] if pref else max(cks, key=lambda c: int(c.name.split("_")[-1]))
        for sto in sorted(ck.glob("*.sto")):
            ep = read_sto(sto)
            if len(ep) < 900:
                n_fall += 1
                continue
            n_ep += 1
            dt = float(ep["time"].iloc[1] - ep["time"].iloc[0])
            for leg, gcol in (("r", "leg1_r.grf_norm_y"), ("l", "leg0_l.grf_norm_y")):
                if gcol not in ep:
                    continue
                g = ep[gcol].to_numpy()
                cyc = seg_cycles(g, dt)
                for a, b in cyc:
                    for j in JOINTS:
                        col = {"hip": f"hip_flexion_{leg}", "knee": f"knee_angle_{leg}",
                               "ankle": f"ankle_angle_{leg}"}[j]
                        s = np.degrees(ep[col].to_numpy()[a:b]) * SCONE_SIGN[j]
                        ang[j].append(time_normalize_stride(s[:, None])[:, 0])
                        sv = np.degrees(ep[col + "_u"].to_numpy()[a:b]) * SCONE_SIGN[j]
                        vel[j].append(time_normalize_stride(sv[:, None])[:, 0])
                    grf.append(time_normalize_stride(g[a:b][:, None])[:, 0])
    if n_ep == 0:
        return None
    A = {j: np.stack(ang[j]) for j in JOINTS}
    V = {j: np.stack(vel[j]) for j in JOINTS}
    return dict(ang={j: A[j].mean(0) for j in JOINTS},
                vel={j: V[j].mean(0) for j in JOINTS},
                grf=np.stack(grf).mean(0), n_ep=n_ep,
                fall=n_fall / (n_ep + n_fall))


def match_fraction(sim_mean, hmean, hsd):
    s = sim_mean - (sim_mean - hmean).mean() if ALIGN_OFFSET else sim_mean
    lo, hi = hmean - hsd, hmean + hsd
    return float(np.mean((s >= lo) & (s <= hi))), float((sim_mean - hmean).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=1.35)
    ap.add_argument("--tol", type=float, default=0.25)
    ap.add_argument("--rebuild-band", action="store_true")
    args = ap.parse_args()

    band = build_human_band(args.speed, args.tol, args.rebuild_band)
    print(f"human band: speed {args.speed}+/-{args.tol} m/s   "
          f"n cycles hip/knee/ankle/grf = "
          f"{band['hip_n']}/{band['knee_n']}/{band['ankle_n']}/{band['grf_n']}\n")

    tmap = load_tstar_map()
    bodies = ["h0918", "h1622", "h2190"]
    rows = []
    simdata = {}
    for body in bodies:
        for cond, ml in METHODS.items():
            if cond not in RUNS[body]:
                continue
            sd = pool_sim(RUNS[body][cond], tmap)
            if sd is None:
                continue
            simdata[(body, ml)] = sd
            mf = {j: match_fraction(sd["ang"][j], band[f"{j}_mean"], band[f"{j}_sd"])[0]
                  for j in JOINTS}
            off = {j: match_fraction(sd["ang"][j], band[f"{j}_mean"], band[f"{j}_sd"])[1]
                   for j in JOINTS}
            gm = match_fraction(sd["grf"], band["grf_mean"], band["grf_sd"])[0]
            rows.append(dict(body=body, method=ml, n_ep=sd["n_ep"],
                             fall=round(sd["fall"], 2),
                             match_hip=round(mf["hip"], 2), match_knee=round(mf["knee"], 2),
                             match_ankle=round(mf["ankle"], 2),
                             match_overall=round(np.mean(list(mf.values())), 2),
                             grf_match=round(gm, 2),
                             peak_grf_bw=round(float(np.max(sd["grf"])), 2),
                             offset_hip=round(off["hip"], 1), offset_knee=round(off["knee"], 1),
                             offset_ankle=round(off["ankle"], 1)))

    import csv
    with open(OUT / "KINEMATIC_MATCH.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{'body':<7}{'method':<11}{'hip':>6}{'knee':>6}{'ankle':>6}{'OVERALL':>9}{'GRF':>6}{'pkGRF':>7}{'fall':>6}")
    for r in rows:
        print(f"{r['body']:<7}{r['method']:<11}{r['match_hip']:>6.2f}{r['match_knee']:>6.2f}"
              f"{r['match_ankle']:>6.2f}{r['match_overall']:>9.2f}{r['grf_match']:>6.2f}"
              f"{r['peak_grf_bw']:>7.2f}{r['fall']:>6.2f}")
    print(f"\nwrote {OUT/'KINEMATIC_MATCH.csv'}")

    # ---- Figure 2 style: angles ----
    pct = np.linspace(0, 100, 101)
    for what, ylab in [("ang", "angle (deg)"), ("vel", "angular velocity (deg/s)")]:
        fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
        for bi, body in enumerate(bodies):
            for ji, j in enumerate(JOINTS):
                ax = axes[bi, ji]
                if what == "ang":
                    ax.fill_between(pct, band[f"{j}_mean"] - band[f"{j}_sd"],
                                    band[f"{j}_mean"] + band[f"{j}_sd"],
                                    color="0.7", alpha=0.5, label="human ±1SD")
                    ax.plot(pct, band[f"{j}_mean"], color="0.35", lw=1)
                for ml in METHODS.values():
                    sd = simdata.get((body, ml))
                    if not sd:
                        continue
                    y = sd[what][j]
                    if what == "ang" and ALIGN_OFFSET:
                        y = y - (y - band[f"{j}_mean"]).mean()
                    ax.plot(pct, y, color=MCOL[ml], lw=1.6, label=ml)
                if bi == 0:
                    ax.set_title(j)
                if ji == 0:
                    ax.set_ylabel(f"{body}\n{ylab}", fontsize=9)
                ax.grid(alpha=0.25)
            axg = axes[bi, 3]
            if what == "ang":
                axg.fill_between(pct, band["grf_mean"] - band["grf_sd"],
                                 band["grf_mean"] + band["grf_sd"], color="0.7", alpha=0.5)
                axg.plot(pct, band["grf_mean"], color="0.35", lw=1)
                for ml in METHODS.values():
                    sd = simdata.get((body, ml))
                    if sd:
                        axg.plot(pct, sd["grf"], color=MCOL[ml], lw=1.6)
                if bi == 0:
                    axg.set_title("vertical GRF (BW)")
                axg.grid(alpha=0.25)
            else:
                axg.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("gait cycle [%]")
        axes[0, 0].legend(fontsize=7, loc="upper right")
        ttl = ("Joint angles + vertical GRF vs experimental human band (±1 SD, "
               f"{len(SUBJ)} subjects)" if what == "ang"
               else "Joint angular velocities by method")
        fig.suptitle(ttl + "  —  EMG prior vs backbones, at the selected checkpoint",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = OUT / (f"KINEMATIC_MATCH_{'angles' if what == 'ang' else 'velocities'}.png")
        fig.savefig(p, dpi=140)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
