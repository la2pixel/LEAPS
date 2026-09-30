"""Experimental-match figure + table, Schumacher et al. 2023 (arXiv:2309.02976)
"Natural and Robust Walking..." Table II style.

experimental_match = fraction of the gait cycle for which the mean simulated
joint-angle trajectory lies within +/-1 SD of the experimental human band
(hip, knee, ankle; both legs pooled). Their exact wording: "the fraction of
the gait cycle for which the average simulated trajectory overlaps within
the standard deviation of experimental data" -- positions only, no
velocities, reference = Bovi et al. data bundled with SCONE (not available
on this machine; we use 6 morphology-matched Camargo subjects instead, same
methodology, different reference cohort -- their H0918/H1622/H2190 = 0.67/
0.73/0.50 is a directionally comparable target, not a strict number-for-
number replication).

Velocity match_fraction is OUR extension, not in their paper -- they don't
score velocities at all (checked directly against the paper text). Human
velocity band = np.gradient of the raw (pre-time-normalization) Camargo
angle trace at native 1000 Hz, then time-normalized the same way as angle.

Joint-moment match_fraction (kinetics) and muscle-activity match_fraction
are ALSO our extension -- not in their paper. Muscle activity reuses
biomech_fidelity.py's cross_human()/pool_condition() (same per-channel
human envelope F2 is built from) and scores it with the identical
match_fraction band-membership test used for kinematics/GRF, so all 5
domains (position, velocity, GRF, moment, muscle activity) land on the
same [0,1] scale and are directly comparable -- reported side by side as
separate columns, deliberately NOT averaged into one number: a blended
score would hide exactly the per-domain/per-muscle dissociation this
project's results are actually about (the same reason experimental_match.py
was deleted and folded into this + gait_metrics + F2 on 2026-09-10 rather
than kept as one all-in-one scorecard).

Muscle-activity match_fraction calls cross_human()/pool_condition() with
normalize_per_cycle=True (fixed 2026-09-14): match_fraction tests absolute
band membership, unlike F2's Pearson r it is NOT invariant to normalization
order, so it needs the standard per-cycle-then-average RVC convention
rather than F2's average-then-normalize (which is fine for F2 only because
correlation doesn't care). F2 itself is untouched by this fix -- verified
the old order changes zero F2 r values, so f2()'s own pool_condition()/
cross_human() calls keep the default normalize_per_cycle=False.

Also plots joint angles + joint velocities + vertical GRF over the cycle,
one line per method, human band shaded.

    python -m leaps.scripts.kinematic_match
    python -m leaps.scripts.kinematic_match --speed 1.35 --tol 0.25
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leaps.data import load_mat_table, find_stride_intervals, time_normalize_stride
from leaps.data.metadata import SUBJECTS
from leaps.scripts.biomech_fidelity import (
    RUNS, RUNS_FULL, load_tstar_map, LIVE, read_sto,
    cross_human, pool_condition, CAPTION_ORDER, CKPT_PREF, MIN_ROWS,
)
from leaps.envs.emg_mapping import MODEL_MAPS
from leaps.paths import RESULTS_DIR
from leaps.data.metadata import CAMARGO_DATA_ROOT

# full-reward regime: only h0918 has current-recipe full-reward data (h1622/h2190's
# RUNS_FULL entries are old k11/generic-decoder recipe, not comparable -- see
# biomech_fidelity.py's own RUNS_FULL comment). This is the one regime where a
# genuine Schumacher et al. Table II comparison is possible (their naturalism
# reward is the analog of our `full`, not `onlyvel`).
METHODS_FULL = {"mpo-full": "MPO (full)", "real-LAP-noclip": "EMG prior (full, noclip)",
                "real-LAP-clip": "EMG prior (full, clip)"}
MCOL_FULL = {"MPO (full)": "#1f77b4", "EMG prior (full, noclip)": "#d62728",
             "EMG prior (full, clip)": "#e377c2"}

SUBJ = ["AB06", "AB08", "AB10", "AB11", "AB23", "AB30"]
JOINTS = ["hip", "knee", "ankle"]
GON = {"hip": "hip_sagittal", "knee": "knee_sagittal", "ankle": "ankle_sagittal"}
MOMENT_COL = {"hip": "hip_flexion_r_moment", "knee": "knee_angle_r_moment", "ankle": "ankle_angle_r_moment"}
POLICY_MASS_KG = 74.53140258789062  # same for h0918/h1622/h2190, confirmed via model.mass()
# Camargo gon conventions (verified against raw data 2026-09-08): hip flexion +,
# knee flexion -, ankle dorsiflexion + -- all three match SCONE's native signs,
# so no flips. gon has a subject-specific DC offset (relative goniometers), so
# the match metric aligns each simulated mean curve to the human mean by
# removing the per-joint cycle-mean difference (compares shape/phase, not the
# goniometer's absolute zero); raw offsets reported separately.
SCONE_SIGN = {"hip": 1.0, "knee": 1.0, "ankle": 1.0}
ALIGN_OFFSET = True
METHODS = {
    "mpo": "MPO", "dep-mpo": "DEP-MPO",
    "real-LAP-w00": "EMG prior w=0.0", "real-LAP": "EMG prior w=0.1",
    "real-LAP-w01-seed0": "EMG prior w=0.1 (seed0, walking)",
    "real-LAP-w05": "EMG prior w=0.5",
    "null": "null",
}
MCOL = {
    "MPO": "#e08214", "DEP-MPO": "#b35806",
    "EMG prior w=0.0": "#9ecae1", "EMG prior w=0.1": "#3182bd",
    "EMG prior w=0.1 (seed0, walking)": "#3182bd", "EMG prior w=0.5": "#08519c",
    "null": "#7f7f7f",
}

# Prior-ladder only -- no MPO/DEP-MPO backbones, adds untrained decoder.
# Colors match plot_h0918_prior_curves.py's COLOR_BY_LABEL exactly, so the
# same condition reads as the same color across every figure in the deck.
METHODS_VARIANTS = {
    "real-LAP-w00": "only prior (w=0.0)", "real-LAP": "prior + 0.1 residual",
    "real-LAP-w05": "prior + 0.5 residual", "null": "no prior (null)",
    "untrained": "untrained decoder",
}
MCOL_VARIANTS = {
    "only prior (w=0.0)": "#3987e5", "prior + 0.1 residual": "#d95926",
    "prior + 0.5 residual": "#199e70", "no prior (null)": "#c98500",
    "untrained decoder": "#d55181",
}
OUT = RESULTS_DIR / "biomech_fidelity"
CACHE = OUT / "human_gait_band.npz"


# ----------------------------------------------------------------- human band
def build_human_band(speed, tol, force=False):
    if CACHE.exists() and not force:
        d = np.load(CACHE)
        if abs(float(d["speed"]) - speed) < 1e-6 and abs(float(d["tol"]) - tol) < 1e-6:
            return {k: d[k] for k in d.files}
    ang = {j: [] for j in JOINTS}
    vel = {j: [] for j in JOINTS}
    mom = {j: [] for j in JOINTS}
    grf = []
    GON_FS = 1000.0  # Hz, leaps.data.metadata.SAMPLE_RATES["gon"]
    for s in SUBJ:
        base = glob.glob(f"{CAMARGO_DATA_ROOT}/{s}/*/treadmill")[0]
        mass = SUBJECTS[s].mass
        bw = mass * 9.81
        for gon_f in sorted(glob.glob(f"{base}/gon/*.mat")):
            stem = Path(gon_f).name
            try:
                g = load_mat_table(gon_f)
                cnd = load_mat_table(f"{base}/conditions/{stem}")
                gc = load_mat_table(f"{base}/gcRight/{stem}")
                fp = load_mat_table(f"{base}/fp/{stem}")
                idd = load_mat_table(f"{base}/id/{stem}")
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
                    raw = _lowpass(g[GON[j]][m], GON_FS)
                    ang[j].append(time_normalize_stride(raw[:, None])[:, 0])
                    # velocity = gradient of the RAW (pre-time-normalized) trace at
                    # native sample rate, deg/s, then time-normalized the same way
                    # -- differentiating after time-normalization would distort the
                    # magnitude since the 101-point grid isn't evenly spaced in time.
                    # Gradient taken on the already-filtered angle so velocity and
                    # position stay consistent with each other.
                    dv = np.gradient(raw, 1.0 / GON_FS)
                    vel[j].append(time_normalize_stride(dv[:, None])[:, 0])
                mi = (idd["Header"] >= t0) & (idd["Header"] < t1)
                if mi.sum() >= 20:
                    for j in JOINTS:
                        col = MOMENT_COL[j]
                        if col in idd:
                            mv = _lowpass(idd[col][mi] / mass, GON_FS)  # Nm -> Nm/kg
                            mom[j].append(time_normalize_stride(mv[:, None])[:, 0])
                mf = (fp["Header"] >= t0) & (fp["Header"] < t1)
                grf.append(time_normalize_stride(
                    _lowpass(fp["Treadmill_R_vy"][mf] / bw, GON_FS)[:, None])[:, 0])
    out = {"speed": speed, "tol": tol}
    for j in JOINTS:
        A = np.stack(ang[j])
        out[f"{j}_mean"], out[f"{j}_sd"], out[f"{j}_n"] = A.mean(0), A.std(0), len(A)
        V = np.stack(vel[j])
        out[f"{j}_vel_mean"], out[f"{j}_vel_sd"] = V.mean(0), V.std(0)
        M = np.stack(mom[j]) if mom[j] else np.zeros((1, 101))
        out[f"{j}_moment_mean"], out[f"{j}_moment_sd"], out[f"{j}_moment_n"] = M.mean(0), M.std(0), len(M)
    G = np.stack(grf)
    out["grf_mean"], out["grf_sd"], out["grf_n"] = G.mean(0), G.std(0), len(G)
    np.savez(CACHE, **out)
    return out


# -------------------------------------------------------------------- sim side
KINEMATICS_LOWPASS_HZ = 20.0  # standard gait-analysis convention (matches the
# already-defined-but-never-applied leaps.data.metadata.GON_FILTER spec).
# Added 2026-09-18: raw sim (.sto, 100Hz) and raw human (gon, 1000Hz) signals
# were both being pooled completely unfiltered -- the human band still looked
# smooth only because it pools many more strides (noise averages out over
# volume), not because it was filtered. Applied to the RAW per-cycle segment,
# before time_normalize_stride, on both sides, same cutoff -- so neither side
# is filtered more than the other. Cycles are 0.6-1.8s (60-180 samples at
# 100Hz, 600-1800 at 1000Hz), comfortably long enough for a stable order-4
# filtfilt at this cutoff on either sample rate.


def _lowpass(x, fs, cutoff=KINEMATICS_LOWPASS_HZ, order=4):
    b, a = butter(order, cutoff / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def seg_cycles(grf, dt):
    """Heel-strike segmentation from vertical GRF, de-bounced the same way
    gait_metrics.py's _hs() already is (fixed there 2026-08-26): a mid-stance
    GRF ripple/double-bump can cross the 5%-of-max threshold twice in quick
    succession, registering as a spurious extra heel strike. Without this,
    every real ~0.8-1.1s stride gets seen as two ~0.4s pseudo-strides, both
    outside the [0.6, 1.8]s window below -- silently zeroing out otherwise-
    good data (found 2026-09-15: h1622 real-LAP-w00, both seeds, near-ceiling
    episode scores, zero cycles detected before this fix)."""
    on = grf > 0.05 * np.nanmax(grf)
    raw_rise = np.where(on[1:] & ~on[:-1])[0] + 1
    min_gap = int(0.4 / dt)
    rise = []
    for i in raw_rise:
        if not rise or i - rise[-1] >= min_gap:
            rise.append(i)
    rise = np.array(rise)
    lo, hi = int(0.6 / dt), int(1.8 / dt)
    return [(rise[i], rise[i + 1]) for i in range(len(rise) - 1)
            if lo <= rise[i + 1] - rise[i] <= hi]


def pool_sim(run_dirs, tmap, min_rows=None):
    """min_rows: fall cutoff. Was hardcoded to 900 (never used the shared
    MIN_ROWS convention pool_condition/pool_torso use, 2500 by default) --
    found 2026-09-18 while sanity-checking a suspiciously high h1622 ankle
    match_fraction: with the 900 cutoff, 3 of seed0's 5 episodes (which F2's
    stricter 2500 threshold correctly treats as falls) were being counted as
    complete, pre-fall-truncated strides polluting the pooled cycles."""
    if min_rows is None:
        min_rows = 2500  # callers should pass MIN_ROWS.get(body, 2500) explicitly
    ang = {j: [] for j in JOINTS}
    vel = {j: [] for j in JOINTS}
    mom = {j: [] for j in JOINTS}
    grf = []
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
            for leg, gcol in (("r", "leg1_r.grf_norm_y"), ("l", "leg0_l.grf_norm_y")):
                if gcol not in ep:
                    continue
                g = ep[gcol].to_numpy()
                cyc = seg_cycles(g, dt)
                for a, b in cyc:
                    for j in JOINTS:
                        col = {"hip": f"hip_flexion_{leg}", "knee": f"knee_angle_{leg}",
                               "ankle": f"ankle_angle_{leg}"}[j]
                        s = _lowpass(np.degrees(ep[col].to_numpy()[a:b]) * SCONE_SIGN[j], 1.0 / dt)
                        ang[j].append(time_normalize_stride(s[:, None])[:, 0])
                        sv = _lowpass(np.degrees(ep[col + "_u"].to_numpy()[a:b]) * SCONE_SIGN[j], 1.0 / dt)
                        vel[j].append(time_normalize_stride(sv[:, None])[:, 0])
                        # col already has the leg suffix baked in (e.g. "hip_flexion_r"),
                        # matching real columns like "hamstrings_r.hip_flexion_r.moment"
                        # -- do NOT append _{leg} again (that was the bug: produced
                        # "..._r_r.moment", which matches nothing).
                        mcols = [c for c in ep.columns
                                 if re.match(rf".+_{leg}\.{col}\.moment$", c)]
                        if mcols:
                            net = _lowpass(ep[mcols].to_numpy()[a:b].sum(axis=1) / POLICY_MASS_KG, 1.0 / dt)
                            mom[j].append(time_normalize_stride(net[:, None])[:, 0])
                    grf.append(time_normalize_stride(_lowpass(g[a:b], 1.0 / dt)[:, None])[:, 0])
    if n_ep == 0 or not grf or any(not ang[j] for j in JOINTS):
        # n_ep counts non-falling episodes, but a full-length episode can still
        # yield zero detected GRF cycles (e.g. a genuinely bad rollout) -- treat
        # that the same as "no usable data" instead of crashing on np.stack.
        return None
    A = {j: np.stack(ang[j]) for j in JOINTS}
    V = {j: np.stack(vel[j]) for j in JOINTS}
    M = {j: np.stack(mom[j]) if mom[j] else None for j in JOINTS}
    G = np.stack(grf)
    return dict(ang={j: A[j].mean(0) for j in JOINTS},
                ang_sd={j: A[j].std(0) for j in JOINTS},
                vel={j: V[j].mean(0) for j in JOINTS},
                mom={j: (M[j].mean(0) if M[j] is not None else None) for j in JOINTS},
                mom_sd={j: (M[j].std(0) if M[j] is not None else None) for j in JOINTS},
                grf=G.mean(0), grf_sd=G.std(0), n_ep=n_ep, n_cyc=len(G),
                fall=n_fall / (n_ep + n_fall))


def match_fraction(sim_mean, hmean, hsd):
    s = sim_mean - (sim_mean - hmean).mean() if ALIGN_OFFSET else sim_mean
    lo, hi = hmean - hsd, hmean + hsd
    return float(np.mean((s >= lo) & (s <= hi))), float((sim_mean - hmean).mean())


def band_overlap_fraction(sim_mean, sim_sd, hmean, hsd):
    """Symmetric alternative to match_fraction: fraction of the cycle where
    the model's OWN +-1SD band overlaps the human's +-1SD band, instead of
    only testing whether the model's mean line falls inside the human band
    alone. Gives a model credit for its own plausible variability -- a model
    with a wide-but-real spread that brackets the human band scores well
    here even if its single mean line drifts outside the (narrow) human
    band at some points, which is exactly the gap match_fraction can't see.
    Same offset-alignment convention as match_fraction (removes constant
    bias, doesn't affect either SD since a shift doesn't change spread)."""
    s = sim_mean - (sim_mean - hmean).mean() if ALIGN_OFFSET else sim_mean
    m_lo, m_hi = s - sim_sd, s + sim_sd
    h_lo, h_hi = hmean - hsd, hmean + hsd
    overlap = (m_lo <= h_hi) & (m_hi >= h_lo)
    return float(np.mean(overlap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=1.35)
    ap.add_argument("--tol", type=float, default=0.25)
    ap.add_argument("--rebuild-band", action="store_true")
    ap.add_argument("--dark", action="store_true",
                    help="dark/transparent slide styling, independent of --regime "
                         "(previously only available via regime=variants, which also "
                         "forced backbones out -- this decouples the two)")
    ap.add_argument("--regime", default="onlyvel", choices=["onlyvel", "full", "variants"],
                    help="onlyvel = velocity-ceiling runs (5-condition w-sweep); "
                         "full = current-recipe full-reward runs, h0918 only -- "
                         "the regime genuinely comparable to Schumacher et al.'s "
                         "Table II naturalism numbers (0.67/0.73/0.50); "
                         "variants = prior ladder only (w=0.0/0.1/0.5, no prior, "
                         "untrained decoder), no MPO/DEP-MPO backbones, dark/"
                         "transparent styled for slides")
    args = ap.parse_args()

    band = build_human_band(args.speed, args.tol, args.rebuild_band)
    print(f"human band: speed {args.speed}+/-{args.tol} m/s   "
          f"n cycles hip/knee/ankle/grf = "
          f"{band['hip_n']}/{band['knee_n']}/{band['ankle_n']}/{band['grf_n']}\n")

    if args.regime == "full":
        runs, methods, mcol, bodies, tmap = RUNS_FULL, METHODS_FULL, MCOL_FULL, ["h0918"], {}
    elif args.regime == "variants":
        runs, methods, mcol, bodies, tmap = RUNS, METHODS_VARIANTS, MCOL_VARIANTS, \
            ["h0918", "h1622", "h2190"], load_tstar_map()
    else:
        runs, methods, mcol, bodies, tmap = RUNS, METHODS, MCOL, ["h0918", "h1622", "h2190"], load_tstar_map()
    # cross-human EMG envelope, same benchmark F2 uses -- computed once per body
    # (channel set is body-specific via MODEL_MAPS), reused across conditions.
    emg_bands = {}

    rows = []
    simdata = {}
    for body in bodies:
        cmap = MODEL_MAPS[body]
        channels = [c for c in CAPTION_ORDER if c in cmap]
        model_muscles = sorted({cmap[c] for c in channels})
        if body not in emg_bands:
            # normalize_per_cycle=True: match_fraction tests absolute band
            # membership (not scale-invariant like F2's Pearson r), so unlike
            # F2 this domain needs the standard per-cycle-then-average
            # normalization order -- see cross_human()'s docstring.
            _, sub_means, ch_idx, pair_r, envelope = cross_human(
                args.speed, args.tol, normalize_per_cycle=True)
            emg_bands[body] = (channels, cmap, envelope, ch_idx)
        for cond, ml in methods.items():
            if cond not in runs[body]:
                continue
            sd = pool_sim(runs[body][cond], tmap, min_rows=MIN_ROWS.get(body, 2500))
            if sd is None:
                print(f"  !! {body} {ml}: no usable cycles, skipped")
                continue
            simdata[(body, ml)] = sd
            mf = {j: match_fraction(sd["ang"][j], band[f"{j}_mean"], band[f"{j}_sd"])[0]
                  for j in JOINTS}
            off = {j: match_fraction(sd["ang"][j], band[f"{j}_mean"], band[f"{j}_sd"])[1]
                   for j in JOINTS}
            # velocity match: same band-membership test, on angular velocity instead of
            # angle. ALIGN_OFFSET doesn't apply here (Camargo gon has a per-subject DC
            # offset on angle, not on its derivative) -- match_fraction's offset removal
            # would just subtract the (near-zero) mean velocity difference, harmless.
            vf = {j: match_fraction(sd["vel"][j], band[f"{j}_vel_mean"], band[f"{j}_vel_sd"])[0]
                  for j in JOINTS}
            # moment match: same test, on net per-DOF muscle moment (Nm/kg). Missing
            # entirely for some conditions if their .sto lacks per-muscle .moment
            # columns -- reported as None, not silently dropped to 0.
            mom_f = {}
            for j in JOINTS:
                if sd["mom"][j] is None:
                    mom_f[j] = None
                else:
                    mom_f[j] = match_fraction(sd["mom"][j], band[f"{j}_moment_mean"],
                                              band[f"{j}_moment_sd"])[0]
            mom_vals = [v for v in mom_f.values() if v is not None]
            gm = match_fraction(sd["grf"], band["grf_mean"], band["grf_sd"])[0]
            # muscle-activity match: same test, per mapped muscle, against the F2
            # cross-human envelope -- the whole point of this extension.
            channels, cmap, envelope, ch_idx = emg_bands[body]
            ckpt_pref = tmap if tmap else CKPT_PREF.get(body)
            waves, _, _, _ = pool_condition(runs[body][cond], model_muscles,
                                            min_rows=MIN_ROWS.get(body, 2500),
                                            prefer_ckpt=ckpt_pref,
                                            normalize_per_cycle=True)
            musc_f = {}
            for ch in channels:
                mm = cmap[ch]
                wv = waves.get(mm)
                if wv is None:
                    continue
                em, es = envelope[ch]
                musc_f[ch] = match_fraction(wv, em, es)[0]
            musc_vals = list(musc_f.values())
            rows.append(dict(body=body, method=ml, n_ep=sd["n_ep"],
                             fall=round(sd["fall"], 2),
                             match_hip=round(mf["hip"], 2), match_knee=round(mf["knee"], 2),
                             match_ankle=round(mf["ankle"], 2),
                             match_overall=round(np.mean(list(mf.values())), 2),
                             vel_match_hip=round(vf["hip"], 2), vel_match_knee=round(vf["knee"], 2),
                             vel_match_ankle=round(vf["ankle"], 2),
                             vel_match_overall=round(np.mean(list(vf.values())), 2),
                             moment_match_hip=(round(mom_f["hip"], 2) if mom_f["hip"] is not None else None),
                             moment_match_knee=(round(mom_f["knee"], 2) if mom_f["knee"] is not None else None),
                             moment_match_ankle=(round(mom_f["ankle"], 2) if mom_f["ankle"] is not None else None),
                             moment_match_overall=(round(float(np.mean(mom_vals)), 2) if mom_vals else None),
                             grf_match=round(gm, 2),
                             muscle_match_overall=(round(float(np.mean(musc_vals)), 2) if musc_vals else None),
                             muscle_match_n=len(musc_vals),
                             peak_grf_bw=round(float(np.max(sd["grf"])), 2),
                             offset_hip=round(off["hip"], 1), offset_knee=round(off["knee"], 1),
                             offset_ankle=round(off["ankle"], 1)))

    import csv
    with open(OUT / f"KINEMATIC_MATCH_{args.regime}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    def fmt(v, w=8):
        return f"{v:>{w}.2f}" if v is not None else f"{'--':>{w}}"

    print(f"{'body':<7}{'method':<17}{'ANG':>8}{'VEL':>8}{'MOMENT':>8}{'GRF':>8}{'MUSCLE':>9}{'pkGRF':>7}{'fall':>6}")
    for r in rows:
        print(f"{r['body']:<7}{r['method']:<17}{fmt(r['match_overall'])}{fmt(r['vel_match_overall'])}"
              f"{fmt(r['moment_match_overall'])}{fmt(r['grf_match'])}"
              f"{fmt(r['muscle_match_overall'], 9)}"
              f"{r['peak_grf_bw']:>7.2f}{r['fall']:>6.2f}")
    print(f"\nwrote {OUT/f'KINEMATIC_MATCH_{args.regime}.csv'}")

    # ---- Figure 2 style: angles ----
    is_dark = args.regime == "variants" or args.dark
    ink = "#e8e8e8" if is_dark else "0.0"
    ink_muted = "#9aa0a6" if is_dark else "0.35"
    band_fill = "#c9cdd1" if is_dark else "0.7"
    band_line = "#e8e8e8" if is_dark else "0.35"
    band_alpha = 0.22 if is_dark else 0.5
    fs_title, fs_label, fs_tick, fs_legend, fs_suptitle = (15, 13, 11, 11, 15) if is_dark else (10, 9, 8, 7, 11)

    pct = np.linspace(0, 100, 101)
    for what, ylab in [("ang", "angle (deg)"), ("vel", "angular velocity (deg/s)"),
                       ("mom", "net moment (Nm/kg)")]:
        fig, axes = plt.subplots(3, 4, figsize=(18, 10.5) if is_dark else (15, 9), sharex=True)
        if is_dark:
            fig.patch.set_alpha(0)
        for bi, body in enumerate(bodies):
            for ji, j in enumerate(JOINTS):
                ax = axes[bi, ji]
                if is_dark:
                    ax.set_facecolor("none")
                if what == "ang":
                    ax.fill_between(pct, band[f"{j}_mean"] - band[f"{j}_sd"],
                                    band[f"{j}_mean"] + band[f"{j}_sd"],
                                    color=band_fill, alpha=band_alpha, label="human ±1SD")
                    ax.plot(pct, band[f"{j}_mean"], color=band_line, lw=1)
                elif what == "mom":
                    ax.fill_between(pct, band[f"{j}_moment_mean"] - band[f"{j}_moment_sd"],
                                    band[f"{j}_moment_mean"] + band[f"{j}_moment_sd"],
                                    color=band_fill, alpha=band_alpha, label="human ±1SD")
                    ax.plot(pct, band[f"{j}_moment_mean"], color=band_line, lw=1)
                for ml in methods.values():
                    sd = simdata.get((body, ml))
                    if not sd:
                        continue
                    y = sd["mom"][j] if what == "mom" else sd[what][j]
                    if y is None:
                        continue  # some conditions lack per-muscle .moment columns
                    if what == "ang" and ALIGN_OFFSET:
                        y = y - (y - band[f"{j}_mean"]).mean()
                    ax.plot(pct, y, color=mcol[ml], lw=2.2 if is_dark else 1.6, label=ml)
                if bi == 0:
                    ax.set_title(j, color=ink, fontsize=fs_title)
                if ji == 0:
                    ax.set_ylabel(f"{body}\n{ylab}", fontsize=fs_label, color=ink)
                if is_dark:
                    ax.tick_params(colors=ink_muted, labelsize=fs_tick)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.spines["bottom"].set_visible(True)
                    ax.spines["left"].set_visible(True)
                    ax.spines["bottom"].set_color(ink_muted)
                    ax.spines["left"].set_color(ink_muted)
                else:
                    ax.grid(alpha=0.25)
            axg = axes[bi, 3]
            if is_dark:
                axg.set_facecolor("none")
            if what == "ang":
                axg.fill_between(pct, band["grf_mean"] - band["grf_sd"],
                                 band["grf_mean"] + band["grf_sd"], color=band_fill, alpha=band_alpha)
                axg.plot(pct, band["grf_mean"], color=band_line, lw=1)
                for ml in methods.values():
                    sd = simdata.get((body, ml))
                    if sd:
                        axg.plot(pct, sd["grf"], color=mcol[ml], lw=2.2 if is_dark else 1.6)
                if bi == 0:
                    axg.set_title("vertical GRF (BW)", color=ink, fontsize=fs_title)
                if is_dark:
                    axg.tick_params(colors=ink_muted, labelsize=fs_tick)
                    for spine in axg.spines.values():
                        spine.set_visible(False)
                    axg.spines["bottom"].set_visible(True)
                    axg.spines["left"].set_visible(True)
                    axg.spines["bottom"].set_color(ink_muted)
                    axg.spines["left"].set_color(ink_muted)
                else:
                    axg.grid(alpha=0.25)
            else:
                axg.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("gait cycle [%]", color=ink, fontsize=fs_label)
        leg = axes[0, 0].legend(fontsize=fs_legend, loc="upper right",
                                 frameon=False if is_dark else True,
                                 labelcolor=ink if is_dark else None)
        ttl_suffix = "prior ladder vs. real human band" if is_dark else "EMG prior vs backbones, at the selected checkpoint"
        if what == "ang":
            ttl = f"Joint angles + vertical GRF vs experimental human band (±1 SD, {len(SUBJ)} subjects)"
        elif what == "mom":
            ttl = f"Net joint moments vs experimental human band (±1 SD, {len(SUBJ)} subjects)"
        else:
            ttl = "Joint angular velocities by method"
        fig.suptitle(ttl + "  —  " + ttl_suffix, fontsize=fs_suptitle, color=ink)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        kind = {"ang": "angles", "vel": "velocities", "mom": "moments"}[what]
        p = OUT / f"KINEMATIC_MATCH_{args.regime}_{kind}.png"
        fig.savefig(p, dpi=140, transparent=is_dark)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
