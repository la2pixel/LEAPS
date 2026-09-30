"""Biomechanical-fidelity figures vs real human data (Park et al. 2026 replicas).

F2 so far: per-muscle Pearson r between simulated muscle activation and
experimental EMG envelopes, against a cross-human benchmark (22 Camargo
subjects, pairwise, n=231). Box row + activation-waveform row, like their Fig 2.

    python -m leaps.scripts.biomech_fidelity --figure f2 --body h1622
    python -m leaps.scripts.biomech_fidelity --figure f2 --body h0918

Data: emg_activations_v2.h5 (LEAPS_EMG_H5), current-recipe .sto rollouts under
final_experiments/. Output: results/biomech_fidelity/.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

from leaps.data import N_GAIT_POINTS, time_normalize_stride
from leaps.envs.emg_mapping import MODEL_MAPS
from leaps.paths import SCONE_LIVE_DIR
from leaps.data.metadata import LEAPS_H5_PATH

H5 = LEAPS_H5_PATH
LIVE = SCONE_LIVE_DIR / "lalitha/final_experiments"
OUT = Path(__file__).resolve().parents[3] / "results" / "biomech_fidelity"

# Park et al. Fig 2 muscle order; last one (gluteusmedius) is our extra.
PARK_ORDER = [
    "soleus", "gastrocmed", "tibialisanterior", "vastusmedialis",
    "vastuslateralis", "rectusfemoris", "bicepsfemoris", "semitendinosus",
    "gluteusmedius",
]
# Their Fig 2's literal caption order (SOL GAS TA ST BF RF VM VL) -- for a
# direct-replica plot, --caption-order uses this instead of PARK_ORDER above.
CAPTION_ORDER = [
    "soleus", "gastrocmed", "tibialisanterior", "semitendinosus",
    "bicepsfemoris", "rectusfemoris", "vastusmedialis", "vastuslateralis",
]
# Their Fig 2 color convention: orange = independent, blue = synergistic.
# Only applied to these condition names when --park-colors is passed; every
# other condition (null, loosen-*) keeps its COLORS entry. w00/w01/w05 get a
# light->dark blue gradient (increasing free-residual share), dep-mpo a
# second warm tone distinct from mpo's orange.
PARK_STYLE_COLORS = {
    "mpo": "#e08214", "dep-mpo": "#b35806",
    "real-LAP-w00": "#9ecae1", "real-LAP": "#3182bd", "real-LAP-w05": "#08519c",
}
PARK_ABBR = {
    "soleus": "SOL", "gastrocmed": "GAS", "tibialisanterior": "TA",
    "vastusmedialis": "VM", "vastuslateralis": "VL", "rectusfemoris": "RF",
    "bicepsfemoris": "BF", "semitendinosus": "ST", "gluteusmedius": "GMED",
}

# run dirs per body -> {condition: [run_dir, ...]}. Only dirs that currently
# hold run_checkpoint_*/*.sto; h2190 pending a pre-collapse (18M) eval pass.
RUNS: dict[str, dict[str, list[str]]] = {
    "h0918": {
        "real-LAP": ["emg_lap/h0918/h0918_k6_w01_onlyvelrew_AB06_corrected_seed0",
                     "emg_lap/h0918/h0918_k6_w01_onlyvelrew_AB06_corrected_seed1"],
        "real-LAP-w00": [f"emg_lap/h0918/h0918_k6_w00_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)],
        "real-LAP-w05": [f"emg_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)],
        "mpo": ["mpo/h0918/h0918_onlyvelrew_seed0", "mpo/h0918/h0918_onlyvelrew_seed1"],
        "dep-mpo": ["dep-mpo/h0918/h0918_onlyvelrew_dep_seed0",
                    "dep-mpo/h0918/h0918_onlyvelrew_dep_seed1"],
        "null": ["no_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_null_seed0",
                 "no_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_null_seed1"],
        "untrained": [f"untrained_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_untrainedprior_seed{s}"
                      for s in (0, 1)],
    },
    "h1622": {
        "real-LAP": [f"emg_lap/h1622/h1622_k6_w01_onlyvelrew_AB06_corrected_seed{s}"
                     for s in (0, 1, 2, 3)],
        # seed0 only -- the sole w=0.1 seed that walks rather than bounds.
        # "real-LAP" above stays pooled (other figures depend on it); the
        # biomechanics section reports walking gaits only.
        "real-LAP-w01-seed0": ["emg_lap/h1622/h1622_k6_w01_onlyvelrew_AB06_corrected_seed0"],
        "real-LAP-w00": [f"emg_lap/h1622/h1622_k6_w00_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)],
        "real-LAP-w05": [f"emg_lap/h1622/h1622_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)],
        "mpo": ["mpo/h1622/h1622_onlyvelrew_seed0",
                "mpo/h1622/h1622_net256_onlyvelrew_seed1",
                "mpo/h1622/h1622_onlyvelrew_seed2"],
        "dep-mpo": [f"dep-mpo/h1622/h1622_onlyvelrew_dep_seed{s}" for s in (0, 1, 2)],
        "null": [f"no_lap/h1622/h1622_k6_w05_onlyvelrew_AB06_corrected_null_seed{s}"
                 for s in (0, 1)],
        "untrained": [f"untrained_lap/h1622/h1622_k6_w05_onlyvelrew_AB06_corrected_untrainedprior_seed{s}"
                      for s in (0, 1)],
        "loosen-GM": ["other/emg_lap/h1622/h1622_k6_w01_loosenglut_med05_onlyvelrew_AB06_corrected_seed0",
                      "other/emg_lap/h1622/h1622_k6_w01_loosenglut_med05_onlyvelrew_AB06_corrected_seed1"],
        "loosen-TA": ["other/emg_lap/h1622/h1622_k6_w01_loosentib_ant05_onlyvelrew_AB06_corrected_seed0",
                      "other/emg_lap/h1622/h1622_k6_w01_loosentib_ant05_onlyvelrew_AB06_corrected_seed1"],
    },
    "h2190": {
        "real-LAP": [f"emg_lap/h2190/h2190_k6_w01_onlyvelrew_AB06_corrected_seed{s}"
                     for s in (0, 1, 2, 3)],
        "real-LAP-w00": [f"emg_lap/h2190/h2190_k6_w00_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)],
        "real-LAP-w05": [f"emg_lap/h2190/h2190_k6_w05_onlyvelrew_AB06_corrected_seed{s}" for s in (0, 1)],
        "mpo": [f"mpo/h2190/h2190_onlyvelrew_seed{s}" for s in (0, 1, 2)],
        "dep-mpo": ["dep-mpo/h2190/h2190_onlyvelrew_dep_seed0",
                    "dep-mpo/h2190/h2190_noclip_net256_onlyvelrew_seed1",
                    "dep-mpo/h2190/h2190_onlyvelrew_dep_seed2"],
        "null": [f"no_lap/h2190/h2190_k6_w05_onlyvelrew_AB06_corrected_null_seed{s}"
                 for s in (0, 1)],
        "untrained": [f"untrained_lap/h2190/h2190_k6_w05_onlyvelrew_AB06_corrected_untrainedprior_seed{s}"
                      for s in (0, 1)],
    },
}

# h2190 collapses late (see project_h2190_mechanistic_findings) -- read the
# pre-collapse 18M checkpoint, not the degraded final one.
# --regime full: the 10M full-SCONE-reward runs (not the onlyVelRew ceiling
# runs above) + the bare mpo backbone at full reward, for the clip-vs-noclip /
# prior-vs-backbone gait-quality question. mpo-full is the only full-reward
# backbone rollout that exists (old July recipe, net256, clip=True -- no
# decoder, so recipe era is moot); there is no noclip mpo-full h0918 anywhere.
# run dirs starting "not_used/" resolve against live/lalitha/, not
# final_experiments/.
RUNS_FULL: dict[str, dict[str, list[str]]] = {
    "h0918": {
        "real-LAP-noclip": [
            "emg_lap/h0918/h0918_k6_w01_full_AB06_corrected_seed0",
            "emg_lap/h0918/h0918_k6_w01_full_AB06_corrected_seed1",
            "emg_lap/h0918/h0918_k6_w01_full_AB06_corrected_seed2",
        ],
        "real-LAP-clip": [
            "emg_lap/h0918/h0918_k6_w01_full_clip_AB06_corrected_seed0",
        ],
        "mpo-full": [
            "not_used/early_tests/mpo/h0918/h0918_clip_net256_full_seed0",
        ],
    },
    # h1622/h2190: NO current-recipe (k6, AB06_corrected) full-reward emg_lap
    # run exists -- those configs are still commented out in run_scripts/
    # queue.txt. The only full-reward rollouts are the old July early_tests
    # ones: k11 + generic pooled decoder (pre the 08-16/08-17 decoder fix that
    # mattered most for these two bodies), clip, net256, 20M. So "real-LAP-old"
    # here is NOT the same arm as h0918's real-LAP-{noclip,clip}. mpo-full is
    # likewise old net256. Read this pair as old-recipe-only.
    "h1622": {
        "real-LAP-old": [
            "not_used/early_tests/emg_lap/h1622/h1622_k11_w01_mirror_clip_net256_full_seed0",
        ],
        "mpo-full": [
            "not_used/early_tests/mpo/h1622/h1622_clip_net256_full_seed0",
        ],
    },
    "h2190": {
        "real-LAP-old": [
            "not_used/early_tests/emg_lap/h2190/h2190_k11_w01_mirror_clip_net256_full_seed0",
        ],
        "mpo-full": [
            "not_used/early_tests/mpo/h2190/h2190_clip_net256_full_seed0",
        ],
    },
}

CKPT_PREF: dict[str, int] = {"h2190": 18_000_000}
# h2190 rarely sustains a full episode even at its best checkpoint -- relax the
# fall filter so partial-gait segments are usable (flag n_cyc when reporting).
MIN_ROWS: dict[str, int] = {"h2190": 400}
COLORS = {"real-LAP": "#d62728", "mpo": "#1f77b4", "dep-mpo": "#2ca02c",
          "null": "#7f7f7f", "loosen-GM": "#ff7f0e", "loosen-TA": "#9467bd",
          "real-LAP-noclip": "#d62728", "real-LAP-clip": "#e377c2",
          "real-LAP-old": "#d62728", "mpo-full": "#1f77b4",
          "untrained": "#d55181", "real-LAP-w00": "#9ecae1", "real-LAP-w05": "#08519c",
          "real-LAP-w01-seed0": "#3182bd"}


def read_sto(path: Path) -> pd.DataFrame:
    with open(path) as f:
        lines = f.readlines()
    hdr = next(i for i, l in enumerate(lines) if l.strip() == "endheader")
    return pd.read_csv(path, sep="\t", skiprows=hdr + 1)


def rvc(wave: np.ndarray) -> np.ndarray:
    """Normalize a (101,) or (101,k) waveform to its own peak amplitude."""
    peak = np.nanmax(np.abs(wave), axis=0)
    peak = np.where(peak < 1e-9, 1.0, peak)
    return wave / peak


# ---------------------------------------------------------------- cross-human
def cross_human(speed: float = 1.2, tol: float = 0.15, normalize_per_cycle: bool = False):
    """Per-channel: 22 subject-mean RVC waveforms, the n=231 pairwise-r
    distribution, and the mean/std envelope across subjects.

    normalize_per_cycle=False (default, F2's behavior): average a subject's
    raw strides first, then peak-normalize the resulting mean -- provably
    inert for F2 (Pearson r is scale-invariant), kept as-is so F2's already-
    reported numbers never move.
    normalize_per_cycle=True: peak-normalize each stride to its own peak
    BEFORE averaging -- the standard EMG convention, and the one that
    matters for match_fraction (kinematic_match.py's muscle-activity
    domain), which tests absolute band membership and is NOT scale-
    invariant. Verified 2026-09-14: on h2190/mpo this shifts some muscles'
    match_fraction by only ~0.01-0.07, but one (ext_obl_r) by +0.37 -- a
    blunted/jittery pooled peak under the old order was deflating the whole
    waveform's normalization denominator."""
    with h5py.File(H5, "r") as f:
        subs = [s for s in f.keys() if s.startswith("AB")]
        channels = [c.decode() if isinstance(c, bytes) else str(c)
                    for c in f.attrs["emg_channels"]]
        sub_means = {}
        for s in subs:
            g = f[s]
            modes = g["modes"][:].astype(str)
            cond = g["conditions"][:].astype(str)
            spd = np.array([float(x) if x.replace(".", "", 1).isdigit() else np.nan
                            for x in cond])
            keep = (modes == "treadmill") & (np.abs(spd - speed) <= tol)
            strides = g["strides"][keep]  # (n,101,11)
            if normalize_per_cycle:
                peak = np.nanmax(np.abs(strides), axis=1, keepdims=True)  # (n,1,11)
                peak = np.where(peak < 1e-9, 1.0, peak)
                sub_means[s] = (strides / peak).mean(axis=0)  # (101,11)
            else:
                sub_means[s] = rvc(strides.mean(axis=0))  # (101,11)

    ch_idx = {c: i for i, c in enumerate(channels)}
    pair_r, envelope = {}, {}
    for c in channels:
        i = ch_idx[c]
        mats = np.stack([sub_means[s][:, i] for s in subs])  # (22,101)
        rs = [stats.pearsonr(mats[a], mats[b])[0]
              for a in range(len(subs)) for b in range(a + 1, len(subs))]
        pair_r[c] = np.array(rs)
        envelope[c] = (mats.mean(0), mats.std(0))
    return channels, sub_means, ch_idx, pair_r, envelope


# ------------------------------------------------------------- sim conditions
def segment_cycles(contact: np.ndarray, thr: float, lo: int = 15, hi: int = 220):
    on = contact > thr
    rise = np.where(on[1:] & ~on[:-1])[0] + 1
    return [(rise[i], rise[i + 1]) for i in range(len(rise) - 1)
            if lo <= rise[i + 1] - rise[i] <= hi]


def pool_condition(run_dirs, muscles, min_rows=2500, prefer_ckpt=None,
                    normalize_per_cycle: bool = False):
    """{model_muscle: mean RVC waveform (101,)} pooled over seeds x episodes.
    Only full-length (non-fall) episodes are used. prefer_ckpt: an int (same
    step for every run_dir), or a dict {run_dir: step} for the per-seed t*
    checkpoint (--tstar); if the exact run_checkpoint_<step> dir is missing,
    falls back to the highest available. normalize_per_cycle: see
    cross_human()'s docstring -- same tradeoff, same default (off, preserves
    F2 exactly)."""
    per_musc = {m: [] for m in muscles}
    n_ep = n_cyc = n_skip = 0
    for rd in run_dirs:
        root = LIVE.parent / rd if rd.startswith("not_used/") else LIVE / rd
        cks = sorted(root.glob("*/run_checkpoint_*"))
        if not cks:
            continue
        want = prefer_ckpt.get(rd) if isinstance(prefer_ckpt, dict) else prefer_ckpt
        pref = [p for p in cks if int(p.name.split("_")[-1]) == want] if want else []
        ck = pref[0] if pref else max(cks, key=lambda p: int(p.name.split("_")[-1]))
        for sto in sorted(ck.glob("*.sto")):
            ep = read_sto(sto)
            if len(ep) < min_rows:            # fell before episode end
                n_skip += 1
                continue
            grf_col = "leg1_r.grf_y" if "leg1_r.grf_y" in ep.columns else "leg0_r.grf_y"
            thr = 0.05 * float(np.nanmax(ep[grf_col].to_numpy()))
            cyc = segment_cycles(ep[grf_col].to_numpy(), thr)
            if not cyc:
                continue
            n_ep += 1
            n_cyc += len(cyc)
            for m in muscles:
                col = f"{m}.activation"
                if col not in ep.columns:
                    continue
                sig = ep[col].to_numpy()
                strides = np.stack([time_normalize_stride(sig[a:b, None])[:, 0]
                                    for a, b in cyc])
                per_musc[m].append(strides)
    out = {}
    for m, chunks in per_musc.items():
        if not chunks:
            continue
        allc = np.concatenate(chunks)  # (n_cyc, 101)
        if normalize_per_cycle:
            peak = np.nanmax(np.abs(allc), axis=1, keepdims=True)
            peak = np.where(peak < 1e-9, 1.0, peak)
            out[m] = (allc / peak).mean(axis=0)
        else:
            out[m] = rvc(allc.mean(axis=0))
    return out, n_ep, n_cyc, n_skip


# ------------------------------------------------------------------------ F2
def load_tstar_map():
    """{relative run_dir: t* step} from every select_checkpoint.py CSV under
    results/checkpoint_selection/ (the *.csv, not the *_reeval.csv)."""
    seldir = OUT.parent / "checkpoint_selection"
    m = {}
    for csvp in seldir.glob("*.csv"):
        if csvp.name.endswith("_reeval.csv"):
            continue
        rdr = csv.DictReader(open(csvp))
        if not rdr.fieldnames or "run_dir" not in rdr.fieldnames:
            continue  # not a select_checkpoint output (e.g. RETURN_TABLE.csv)
        for row in rdr:
            rd = row["run_dir"].rstrip("/")
            for pref in (str(LIVE) + "/", str(LIVE.parent) + "/"):
                if rd.startswith(pref):
                    rd = rd[len(pref):]
                    break
            m[rd] = int(row["t_star"])
    return m


def f2(body: str, speed: float, tol: float, regime: str = "onlyvel",
       tstar: bool = False, conditions: list[str] | None = None,
       park_colors: bool = False, caption_order: bool = False,
       dark: bool = False, muscles: list[str] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Same dark/transparent deck convention as kinematics_combined_vs_backbones.py
    # / kinetics_combined_vs_backbones.py -- prior=red, mpo=blue, dep-mpo=gold.
    INK = "#e8e8e8"
    INK_MUTED = "#9aa0a6"
    HUMAN_FILL = "#c9cdd1"
    HUMAN_LINE = "#e8e8e8"
    DECK_COLORS = {"real-LAP": "#e64545", "mpo": "#6fa8dc", "dep-mpo": "#f2c14e"}

    runs = RUNS if regime == "onlyvel" else RUNS_FULL
    if body not in runs:
        raise SystemExit(f"no --regime {regime} runs defined for {body}")
    suffix = ("" if regime == "onlyvel" else "_full") + ("_tstar" if tstar else "")
    if conditions:
        suffix += "_" + "-".join(conditions)
    if caption_order:
        suffix += "_captionorder"
    if muscles:
        suffix += "_" + "-".join(muscles)
    ckpt_pref = load_tstar_map() if tstar else CKPT_PREF.get(body)

    cmap = MODEL_MAPS[body]  # emg_channel -> model muscle (_r)
    order = CAPTION_ORDER if caption_order else PARK_ORDER
    channels = [c for c in order if c in cmap]
    if muscles:
        channels = [c for c in channels if PARK_ABBR[c] in muscles]
    chn, sub_means, ch_idx, pair_r, envelope = cross_human(speed, tol)
    subs = list(sub_means)

    conds = conditions if conditions else list(runs[body])
    unknown = [cd for cd in conds if cd not in runs[body]]
    if unknown:
        raise SystemExit(f"--conditions has unknown condition(s) for {body}: "
                         f"{unknown}; available: {list(runs[body])}")
    colors = dict(COLORS)
    if park_colors:
        colors.update(PARK_STYLE_COLORS)
    if dark:
        colors.update(DECK_COLORS)
    waves = {}
    for cd in conds:
        model_muscles = sorted({cmap[c] for c in channels})
        w, n_ep, n_cyc, n_skip = pool_condition(runs[body][cd], model_muscles,
                                                min_rows=MIN_ROWS.get(body, 2500),
                                                prefer_ckpt=ckpt_pref)
        waves[cd] = w
        print(f"  {body:6s} {cd:10s}: {n_ep} full episodes ({n_skip} falls skipped), "
              f"{n_cyc} gait cycles, {len(w)}/{len(model_muscles)} muscles")

    rows = []
    pct = np.linspace(0, 100, N_GAIT_POINTS)
    ncol = len(channels)
    fig, ax = plt.subplots(2, ncol, figsize=(2.55 * ncol, 6.4),
                           gridspec_kw={"height_ratios": [1.15, 1]})
    if ncol == 1:
        ax = ax.reshape(2, 1)
    if dark:
        fig.patch.set_alpha(0)

    for j, ch in enumerate(channels):
        i = ch_idx[ch]
        mm = cmap[ch]
        # ---- box row: cross-human (n=231) + each condition (n=22) ----
        box_data, box_pos, box_col, xt = [pair_r[ch]], [0], ["#bdbdbd"], ["human\n(n=231)"]
        for k, cd in enumerate(conds, start=1):
            wv = waves[cd].get(mm)
            if wv is None:
                continue
            rr = np.array([stats.pearsonr(wv, sub_means[s][:, i])[0] for s in subs])
            box_data.append(rr)
            box_pos.append(k)
            box_col.append(colors[cd])
            xt.append(cd)
            rows.append(dict(body=body, muscle=PARK_ABBR[ch], model_muscle=mm,
                             condition=cd, r_mean=float(rr.mean()),
                             r_std=float(rr.std()), n=len(rr),
                             human_r_mean=float(pair_r[ch].mean()),
                             in_human_band=bool(
                                 rr.mean() >= np.percentile(pair_r[ch], 5))))
        bp = ax[0, j].boxplot(box_data, positions=box_pos, widths=0.6,
                              patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], box_col):
            patch.set_facecolor(c)
            patch.set_alpha(0.55 if not dark else 0.75)
        for med in bp["medians"]:
            med.set_color("black" if not dark else "#1a1a1a")
        if dark:
            for whisk in bp["whiskers"] + bp["caps"]:
                whisk.set_color(INK_MUTED)
        lo5 = np.percentile(pair_r[ch], 5)
        band_c = "#bdbdbd" if not dark else HUMAN_FILL
        ax[0, j].axhspan(lo5, 1.0, color=band_c, alpha=0.18, zorder=0)
        title_c = "black" if not dark else INK
        ax[0, j].set_title(f"{PARK_ABBR[ch]}  ({mm})", fontsize=12, color=title_c)
        ax[0, j].set_xticks(box_pos)
        ax[0, j].set_xticklabels(xt, rotation=40, ha="right", fontsize=9.5,
                                 color=(None if not dark else INK_MUTED))
        ax[0, j].set_ylim(-0.6, 1.02)
        if j == 0:
            ax[0, j].set_ylabel("Pearson r vs human EMG", color=title_c, fontsize=11)
        if dark:
            ax[0, j].set_facecolor("none")
            for spine in ax[0, j].spines.values():
                spine.set_visible(False)
            ax[0, j].spines["bottom"].set_visible(True)
            ax[0, j].spines["left"].set_visible(True)
            ax[0, j].spines["bottom"].set_color(INK_MUTED)
            ax[0, j].spines["left"].set_color(INK_MUTED)
            ax[0, j].tick_params(axis="y", colors=INK_MUTED, labelsize=10)

        # ---- waveform row ----
        em, es = envelope[ch]
        line_c = "#636363" if not dark else HUMAN_LINE
        ax[1, j].fill_between(pct, em - es, em + es, color=band_c, alpha=(0.55 if not dark else 0.30),
                             label="human ±1SD")
        ax[1, j].plot(pct, em, color=line_c, lw=1)
        for cd in conds:
            wv = waves[cd].get(mm)
            if wv is not None:
                # display-only smoothing (Pearson r above is computed from the
                # unsmoothed wv) -- cyclic wrap since 0%/100% of the gait cycle
                # are the same instant.
                wv_plot = savgol_filter(wv, window_length=21, polyorder=3, mode="wrap") \
                    if len(wv) >= 21 else wv
                ax[1, j].plot(pct, wv_plot, color=colors[cd], lw=1.4, label=cd)
        ax[1, j].set_ylim(-0.05, 1.05)
        ax[1, j].set_xlabel("gait cycle [%]", color=title_c, fontsize=11)
        if j == 0:
            ax[1, j].set_ylabel("activation (peak-normalized)", color=title_c, fontsize=11)
        if dark:
            ax[1, j].set_facecolor("none")
            for spine in ax[1, j].spines.values():
                spine.set_visible(False)
            ax[1, j].spines["bottom"].set_visible(True)
            ax[1, j].spines["left"].set_visible(True)
            ax[1, j].spines["bottom"].set_color(INK_MUTED)
            ax[1, j].spines["left"].set_color(INK_MUTED)
            ax[1, j].tick_params(colors=INK_MUTED, labelsize=10)
    leg = ax[1, -1].legend(fontsize=9.5, loc="upper right",
                           labelcolor=(None if not dark else INK),
                           framealpha=(None if not dark else 0.0))
    fig.suptitle(f"F2 — per-muscle activation vs experimental EMG   "
                 f"[{body}, {regime} reward, ~{speed} m/s]", fontsize=14,
                 color=(None if not dark else INK))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"f2_{body}{suffix}{'_dark' if dark else ''}.png"
    fig.savefig(png, dpi=140, transparent=dark)
    csvp = OUT / f"f2_{body}{suffix}.csv"
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {png}\nwrote {csvp}")
    # console summary
    print(f"\n{'muscle':8s} {'human':>7s} " + " ".join(f"{c:>10s}" for c in conds))
    for ch in channels:
        line = f"{PARK_ABBR[ch]:8s} {pair_r[ch].mean():7.3f} "
        for cd in conds:
            m = [r for r in rows if r["muscle"] == PARK_ABBR[ch] and r["condition"] == cd]
            line += f" {m[0]['r_mean']:10.3f}" if m else f" {'--':>10s}"
        print(line)


def f2_subject(body: str, speed: float, tol: float, ref_subject: str,
                tstar: bool = False, conditions: list[str] | None = None,
                caption_order: bool = False, dark: bool = False,
                muscles: list[str] | None = None):
    """Same per-muscle waveform comparison as f2(), but against ONE named
    Camargo subject's own mean stride (e.g. AB06, the decoder's actual
    source subject) instead of the pooled 22-subject band/boxplot -- "how
    does this compare to the specific human whose EMG the decoder was
    trained on," not "how does this compare to humans in general." No
    boxplot (a single reference has no distribution to box), one Pearson r
    per condition per muscle instead, printed and saved to CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK_MUTED, HUMAN_LINE = "#e8e8e8", "#9aa0a6", "#e8e8e8"
    DECK_COLORS = {"real-LAP": "#e64545", "mpo": "#6fa8dc", "dep-mpo": "#f2c14e"}

    if body not in RUNS:
        raise SystemExit(f"no runs defined for {body}")
    _, sub_means, ch_idx, _, _ = cross_human(speed, tol)
    if ref_subject not in sub_means:
        raise SystemExit(f"{ref_subject} not found; available: {sorted(sub_means)}")
    ref = sub_means[ref_subject]

    cmap = MODEL_MAPS[body]
    order = CAPTION_ORDER if caption_order else PARK_ORDER
    channels = [c for c in order if c in cmap]
    if muscles:
        channels = [c for c in channels if PARK_ABBR[c] in muscles]

    ckpt_pref = load_tstar_map() if tstar else CKPT_PREF.get(body)
    conds = conditions if conditions else list(RUNS[body])
    unknown = [cd for cd in conds if cd not in RUNS[body]]
    if unknown:
        raise SystemExit(f"--conditions has unknown condition(s) for {body}: "
                         f"{unknown}; available: {list(RUNS[body])}")
    colors = dict(COLORS)
    if dark:
        colors.update(DECK_COLORS)

    waves = {}
    for cd in conds:
        model_muscles = sorted({cmap[c] for c in channels})
        w, n_ep, n_cyc, n_skip = pool_condition(RUNS[body][cd], model_muscles,
                                                min_rows=MIN_ROWS.get(body, 2500),
                                                prefer_ckpt=ckpt_pref)
        waves[cd] = w
        print(f"  {body:6s} {cd:10s}: {n_ep} full episodes ({n_skip} falls skipped), "
              f"{n_cyc} gait cycles, {len(w)}/{len(model_muscles)} muscles")

    pct = np.linspace(0, 100, N_GAIT_POINTS)
    rows = []
    ncol = len(channels)
    fig, ax = plt.subplots(1, ncol, figsize=(3.1 * ncol, 3.6), squeeze=False)
    if dark:
        fig.patch.set_alpha(0)
    title_c = INK if dark else "black"

    for j, ch in enumerate(channels):
        i = ch_idx[ch]
        mm = cmap[ch]
        a = ax[0, j]
        refwave = ref[:, i]
        a.plot(pct, refwave, color=(HUMAN_LINE if dark else "#636363"), lw=1.6,
               label=f"{ref_subject} (human)")
        for cd in conds:
            wv = waves[cd].get(mm)
            if wv is None:
                continue
            r = float(stats.pearsonr(wv, refwave)[0])
            rows.append(dict(body=body, muscle=PARK_ABBR[ch], model_muscle=mm,
                             condition=cd, ref_subject=ref_subject, r=round(r, 3)))
            a.plot(pct, wv, color=colors[cd], lw=1.4, label=f"{cd} (r={r:.2f})")
        a.set_title(f"{PARK_ABBR[ch]}  ({mm})", fontsize=12, color=title_c)
        a.set_xlabel("gait cycle [%]", color=title_c)
        if j == 0:
            a.set_ylabel("activation (peak-normalized)", color=title_c)
        a.legend(fontsize=7.5, loc="upper right",
                 labelcolor=(INK if dark else None),
                 framealpha=(0.0 if dark else None))
        if dark:
            a.set_facecolor("none")
            for spine in a.spines.values():
                spine.set_visible(False)
            a.spines["bottom"].set_visible(True)
            a.spines["left"].set_visible(True)
            a.spines["bottom"].set_color(INK_MUTED)
            a.spines["left"].set_color(INK_MUTED)
            a.tick_params(colors=INK_MUTED, labelsize=9)
    fig.suptitle(f"muscle activation vs {ref_subject}'s own EMG   [{body}, ~{speed} m/s]",
                 fontsize=13, color=title_c)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    suffix = ("_tstar" if tstar else "") + ("_" + "-".join(conditions) if conditions else "")
    if muscles:
        suffix += "_" + "-".join(muscles)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"f2_{body}_vs_{ref_subject}{suffix}{'_dark' if dark else ''}.png"
    fig.savefig(png, dpi=140, transparent=dark)
    csvp = OUT / f"f2_{body}_vs_{ref_subject}{suffix}.csv"
    with open(csvp, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {png}\nwrote {csvp}")
    print(f"\n{'muscle':8s} " + " ".join(f"{c:>10s}" for c in conds))
    for ch in channels:
        line = f"{PARK_ABBR[ch]:8s} "
        for cd in conds:
            m = [rr for rr in rows if rr["muscle"] == PARK_ABBR[ch] and rr["condition"] == cd]
            line += f" {m[0]['r']:10.3f}" if m else f" {'--':>10s}"
        print(line)


# ---------------------------------------------------------------- R8 k-sweep
# h0918 only, emg_lap onlyVelRew, w=0.1 fixed, k in {2,4,6,8,11}, n=3 seeds.
# Return is flat ~1000 on h0918 for every k (the R3 null) -- this asks whether
# per-muscle EMG-timing fidelity (F2) or muscle effort move with latent dim.
KSWEEP_KS = (2, 4, 6, 8, 11)
KSWEEP_SEEDS = (0, 1, 2)
KSWEEP_TMPL = "emg_lap/h0918/h0918_k{k}_w01_onlyvelrew_AB06_corrected_seed{s}"


def _episode_effort(sto_path: Path):
    """(mean|a|, rms a) over every *.activation column, dropping the 5% start/
    stop transient. mean|a| matches gait_scorecard's activation_mean and the
    R3 ~0.33 numbers; rms is the sqrt(<a^2>) 'effort' the outline writes."""
    ep = read_sto(sto_path)
    cols = [c for c in ep.columns if c.endswith(".activation")]
    if not cols:
        return None
    a = ep[cols].to_numpy()
    n = len(a)
    a = a[int(0.05 * n):int(0.95 * n)]
    return float(np.mean(np.abs(a))), float(np.sqrt(np.mean(a ** 2)))


def ksweep(speed: float = 1.2, tol: float = 0.15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    body = "h0918"
    cmap = MODEL_MAPS[body]
    channels = [c for c in PARK_ORDER if c in cmap]
    model_muscles = sorted({cmap[c] for c in channels})
    _, sub_means, ch_idx, pair_r, _ = cross_human(speed, tol)
    subs = list(sub_means)

    rows = []
    for k in KSWEEP_KS:
        for s in KSWEEP_SEEDS:
            rd = KSWEEP_TMPL.format(k=k, s=s)
            root = LIVE / rd
            cks = sorted(root.glob("*/run_checkpoint_*"))
            if not cks:
                print(f"  k{k} s{s}: NO run_checkpoint -- skipped")
                continue
            ck = max(cks, key=lambda p: int(p.name.split("_")[-1]))
            stos = sorted(ck.glob("*.sto"))
            # per-muscle F2 r for this single seed
            w, n_ep, n_cyc, n_skip = pool_condition([rd], model_muscles,
                                                    min_rows=MIN_ROWS.get(body, 2500))
            efforts = [e for e in (_episode_effort(p) for p in stos) if e]
            eff_abs = float(np.mean([e[0] for e in efforts])) if efforts else np.nan
            eff_rms = float(np.mean([e[1] for e in efforts])) if efforts else np.nan
            rmus = {}
            for ch in channels:
                wv = w.get(cmap[ch])
                if wv is None:
                    continue
                rr = np.mean([stats.pearsonr(wv, sub_means[sb][:, ch_idx[ch]])[0]
                              for sb in subs])
                rmus[PARK_ABBR[ch]] = float(rr)
                rows.append(dict(k=k, seed=s, muscle=PARK_ABBR[ch], r=float(rr),
                                 effort_abs=eff_abs, effort_rms=eff_rms,
                                 n_ep=n_ep, n_cyc=n_cyc))
            f2m = float(np.mean(list(rmus.values()))) if rmus else np.nan
            print(f"  k{k:2d} s{s}: F2 mean r {f2m:.3f}  effort|a| {eff_abs:.3f}  "
                  f"rms {eff_rms:.3f}  ({n_ep} ep, {n_cyc} cyc)")

    OUT.mkdir(parents=True, exist_ok=True)
    csvp = OUT / "ksweep_h0918.csv"
    with open(csvp, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    # aggregate per k: F2 (mean muscle r, per seed then across seeds) + effort
    df = pd.DataFrame(rows)
    perseed = df.groupby(["k", "seed"]).agg(f2=("r", "mean"),
                                            effort_abs=("effort_abs", "first"),
                                            effort_rms=("effort_rms", "first")).reset_index()
    agg = perseed.groupby("k").agg(f2_mean=("f2", "mean"), f2_sd=("f2", "std"),
                                   eff_mean=("effort_abs", "mean"),
                                   eff_sd=("effort_abs", "std"),
                                   rms_mean=("effort_rms", "mean"),
                                   rms_sd=("effort_rms", "std")).reset_index()
    human_f2 = float(np.mean([pair_r[c].mean() for c in channels]))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ks = agg["k"].to_numpy()
    ax[0].axhline(human_f2, color="#bdbdbd", lw=6, alpha=0.5, label="cross-human")
    for mus in sorted(df["muscle"].unique()):
        g = df[df.muscle == mus].groupby("k")["r"].mean()
        ax[0].plot(g.index, g.values, color="#cccccc", lw=0.9, zorder=1)
    ax[0].errorbar(ks, agg["f2_mean"], yerr=agg["f2_sd"], marker="o", lw=2,
                   color="#d62728", capsize=3, label="F2 mean r (±SD, n=3)")
    ax[0].set_xlabel("latent dim k"); ax[0].set_ylabel("Pearson r vs human EMG")
    ax[0].set_title("F2 per-muscle EMG-timing fidelity vs k")
    ax[0].set_xticks(ks); ax[0].legend(fontsize=8)

    ax[1].errorbar(ks, agg["eff_mean"], yerr=agg["eff_sd"], marker="o", lw=2,
                   color="#1f77b4", capsize=3, label="mean |activation|")
    ax[1].errorbar(ks, agg["rms_mean"], yerr=agg["rms_sd"], marker="s", lw=2,
                   color="#2ca02c", capsize=3, label="rms activation")
    ax[1].set_xlabel("latent dim k"); ax[1].set_ylabel("activation")
    ax[1].set_title("Muscle effort vs k"); ax[1].set_xticks(ks)
    ax[1].legend(fontsize=8)
    fig.suptitle("R8 k-sweep — h0918 emg_lap onlyVelRew, w=0.1  "
                 "(episode return flat ~1000 for all k)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = OUT / "ksweep_h0918.png"
    fig.savefig(png, dpi=140)
    print(f"\nwrote {png}\nwrote {csvp}")
    print(agg.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figure", default="f2", choices=["f2", "ksweep"])
    p.add_argument("--body", default="h1622", choices=["h0918", "h1622", "h2190"])
    p.add_argument("--regime", default="onlyvel", choices=["onlyvel", "full"],
                   help="onlyvel = velocity-ceiling runs; full = 10M full-SCONE-reward + mpo backbone")
    p.add_argument("--speed", type=float, default=None,
                   help="human ref speed (default 1.2 onlyvel / 1.3 full)")
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--tstar", action="store_true",
                   help="use per-seed t* checkpoints from results/checkpoint_selection/")
    p.add_argument("--conditions", default=None,
                   help="comma-separated subset of conditions to plot, e.g. "
                        "mpo,real-LAP (default: every condition defined for --body)")
    p.add_argument("--park-colors", action="store_true",
                   help="recolor mpo orange / real-LAP blue, matching Park et al. "
                        "Fig 2's independent/synergistic convention")
    p.add_argument("--caption-order", action="store_true",
                   help="use Park et al. Fig 2's literal caption muscle order "
                        "(SOL GAS TA ST BF RF VM VL) instead of the reconstruction-elbow order")
    p.add_argument("--dark", action="store_true",
                   help="dark/transparent deck styling matching the kinematics/kinetics "
                        "combined figures (prior=red, mpo=blue, dep-mpo=gold)")
    p.add_argument("--muscles", default=None,
                   help="comma-separated subset of PARK_ABBR codes to plot, e.g. "
                        "ST,BF,VM,VL (default: every muscle mapped for --body)")
    p.add_argument("--ref-subject", default=None,
                   help="Camargo subject id (e.g. AB06) -- switches to f2_subject(): "
                        "compare against that ONE subject's own EMG instead of the "
                        "22-subject pooled band/boxplot")
    a = p.parse_args()
    speed = a.speed if a.speed is not None else (1.2 if a.regime == "onlyvel" else 1.3)
    tol = a.tol if a.tol is not None else (0.15 if a.regime == "onlyvel" else 0.25)
    conditions = a.conditions.split(",") if a.conditions else None
    muscles = a.muscles.split(",") if a.muscles else None
    if a.figure == "ksweep":
        ksweep(speed, tol)
    elif a.ref_subject:
        f2_subject(a.body, speed, tol, a.ref_subject, a.tstar, conditions,
                    a.caption_order, a.dark, muscles)
    else:
        f2(a.body, speed, tol, a.regime, a.tstar, conditions,
           a.park_colors, a.caption_order, a.dark, muscles)


if __name__ == "__main__":
    main()
