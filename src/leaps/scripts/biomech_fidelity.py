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
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from leaps.data.processing import N_GAIT_POINTS, time_normalize_stride
from leaps.envs.emg_mapping import MODEL_MAPS

H5 = os.environ.get("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")
LIVE = Path("/media/calc_2/scone_results/live/lalitha/final_experiments")
OUT = Path(__file__).resolve().parents[3] / "results" / "biomech_fidelity"

# Park et al. Fig 2 muscle order; last one (gluteusmedius) is our extra.
PARK_ORDER = [
    "soleus", "gastrocmed", "tibialisanterior", "vastusmedialis",
    "vastuslateralis", "rectusfemoris", "bicepsfemoris", "semitendinosus",
    "gluteusmedius",
]
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
        "mpo": ["mpo/h0918/h0918_onlyvelrew_seed0", "mpo/h0918/h0918_onlyvelrew_seed1"],
        "dep-mpo": ["dep-mpo/h0918/h0918_onlyvelrew_dep_seed0",
                    "dep-mpo/h0918/h0918_onlyvelrew_dep_seed1"],
        "null": ["no_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_null_seed0",
                 "no_lap/h0918/h0918_k6_w05_onlyvelrew_AB06_corrected_null_seed1"],
    },
    "h1622": {
        "real-LAP": ["emg_lap/h1622/h1622_k6_w01_onlyvelrew_AB06_corrected_seed0",
                     "emg_lap/h1622/h1622_k6_w01_onlyvelrew_AB06_corrected_seed1"],
        "mpo": ["mpo/h1622/h1622_onlyvelrew_seed0", "mpo/h1622/h1622_net256_onlyvelrew_seed1"],
        "dep-mpo": ["dep-mpo/h1622/h1622_onlyvelrew_dep_seed0",
                    "dep-mpo/h1622/h1622_onlyvelrew_dep_seed1"],
        "null": ["no_lap/h1622/h1622_k6_w05_onlyvelrew_AB06_corrected_null_seed0",
                 "no_lap/h1622/h1622_k6_w05_onlyvelrew_AB06_corrected_null_seed1"],
        "loosen-GM": ["other/emg_lap/h1622/h1622_k6_w01_loosenglut_med05_onlyvelrew_AB06_corrected_seed0",
                      "other/emg_lap/h1622/h1622_k6_w01_loosenglut_med05_onlyvelrew_AB06_corrected_seed1"],
        "loosen-TA": ["other/emg_lap/h1622/h1622_k6_w01_loosentib_ant05_onlyvelrew_AB06_corrected_seed0",
                      "other/emg_lap/h1622/h1622_k6_w01_loosentib_ant05_onlyvelrew_AB06_corrected_seed1"],
    },
}
COLORS = {"real-LAP": "#d62728", "mpo": "#1f77b4", "dep-mpo": "#2ca02c",
          "null": "#7f7f7f", "loosen-GM": "#ff7f0e", "loosen-TA": "#9467bd"}


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
def cross_human(speed: float = 1.2, tol: float = 0.15):
    """Per-channel: 22 subject-mean RVC waveforms, the n=231 pairwise-r
    distribution, and the mean/std envelope across subjects."""
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


def pool_condition(run_dirs, muscles):
    """{model_muscle: mean RVC waveform (101,)} pooled over seeds x episodes."""
    per_musc = {m: [] for m in muscles}
    n_ep = n_cyc = 0
    for rd in run_dirs:
        root = LIVE / rd
        cks = sorted(root.glob("*/run_checkpoint_*"))
        if not cks:
            continue
        ck = max(cks, key=lambda p: int(p.name.split("_")[-1]))
        for sto in sorted(ck.glob("*.sto")):
            ep = read_sto(sto)
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
        out[m] = rvc(np.concatenate(chunks).mean(axis=0))
    return out, n_ep, n_cyc


# ------------------------------------------------------------------------ F2
def f2(body: str, speed: float, tol: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = MODEL_MAPS[body]  # emg_channel -> model muscle (_r)
    channels = [c for c in PARK_ORDER if c in cmap]
    chn, sub_means, ch_idx, pair_r, envelope = cross_human(speed, tol)
    subs = list(sub_means)

    conds = list(RUNS[body])
    waves = {}
    for cd in conds:
        model_muscles = sorted({cmap[c] for c in channels})
        w, n_ep, n_cyc = pool_condition(RUNS[body][cd], model_muscles)
        waves[cd] = w
        print(f"  {body:6s} {cd:10s}: {n_ep} episodes, {n_cyc} gait cycles, "
              f"{len(w)}/{len(model_muscles)} muscles")

    rows = []
    pct = np.linspace(0, 100, N_GAIT_POINTS)
    ncol = len(channels)
    fig, ax = plt.subplots(2, ncol, figsize=(2.55 * ncol, 6.4),
                           gridspec_kw={"height_ratios": [1.15, 1]})
    if ncol == 1:
        ax = ax.reshape(2, 1)

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
            box_col.append(COLORS[cd])
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
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        lo5 = np.percentile(pair_r[ch], 5)
        ax[0, j].axhspan(lo5, 1.0, color="#bdbdbd", alpha=0.18, zorder=0)
        ax[0, j].set_title(f"{PARK_ABBR[ch]}  ({mm})", fontsize=9)
        ax[0, j].set_xticks(box_pos)
        ax[0, j].set_xticklabels(xt, rotation=40, ha="right", fontsize=6.5)
        ax[0, j].set_ylim(-0.6, 1.02)
        if j == 0:
            ax[0, j].set_ylabel("Pearson r vs human EMG")

        # ---- waveform row ----
        em, es = envelope[ch]
        ax[1, j].fill_between(pct, em - es, em + es, color="#bdbdbd", alpha=0.55,
                             label="human ±1SD")
        ax[1, j].plot(pct, em, color="#636363", lw=1)
        for cd in conds:
            wv = waves[cd].get(mm)
            if wv is not None:
                ax[1, j].plot(pct, wv, color=COLORS[cd], lw=1.4, label=cd)
        ax[1, j].set_ylim(-0.05, 1.05)
        ax[1, j].set_xlabel("gait cycle [%]")
        if j == 0:
            ax[1, j].set_ylabel("activation (RVC-norm)")
    ax[1, -1].legend(fontsize=6.5, loc="upper right")
    fig.suptitle(f"F2 — per-muscle activation vs experimental EMG   "
                 f"[{body}, ~{speed} m/s]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"f2_{body}.png"
    fig.savefig(png, dpi=140)
    csvp = OUT / f"f2_{body}.csv"
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figure", default="f2", choices=["f2"])
    p.add_argument("--body", default="h1622", choices=["h0918", "h1622", "h2190"])
    p.add_argument("--speed", type=float, default=1.2)
    p.add_argument("--tol", type=float, default=0.15)
    a = p.parse_args()
    f2(a.body, a.speed, a.tol)


if __name__ == "__main__":
    main()
