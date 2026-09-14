"""Does a temporal decoder reconstruct lower-limb EMG better at low K?

Adds a temporal variant to the per-frame comparison in
ae_vs_linear_synergies.py:
  - phase-AE : feed-forward AE + sin/cos gait-phase conditioning
              (NOT RL-deployable -- no online gait-phase signal in the sim --
               but shows the ceiling temporal context buys)

Analysis only. Reconstruction fidelity vs the frozen per-frame AE and the
linear baselines (read from r2_vs_k.csv).

    python -m leaps.scripts.temporal_ae_check
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/nadinebadie/lalitha/LEAPS/src")
from leaps.envs.emg_mapping import EMG_CHANNELS
from leaps.synergy_common import prepare_training_arrays, train_ae, r2

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parents[3] / "results" / "synergy_ae_vs_linear"
K_VALUES = [1, 2, 3, 4, 5, 6]
K_WAVE = 3

MUSCLE_LABEL = {
    "gastrocmed": "Gastroc", "tibialisanterior": "Tib. Ant.", "soleus": "Soleus",
    "vastusmedialis": "Vast. Med.", "vastuslateralis": "Vast. Lat.",
    "rectusfemoris": "Rect. Fem.", "bicepsfemoris": "Biceps Fem.",
    "semitendinosus": "Semitend.", "gracilis": "Gracilis",
    "gluteusmedius": "Glut. Med.", "rightexternaloblique": "Ext. Obl.",
}
LABELS = [MUSCLE_LABEL[c] for c in EMG_CHANNELS]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "#777777",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
})
PAL = {"real": "#3A3A3A", "AE (per-frame)": "#5FA8A0", "PCA": "#8AA9D6",
       "NMF": "#E7A08C", "phase-AE": "#E8C36B"}
MARK = {"AE (per-frame)": "D", "PCA": "o", "NMF": "^", "phase-AE": "P"}


def main():
    d = prepare_training_arrays()
    Xva = d["X_val"]
    Xtr_sub, ph_sub = d["X_train_sub"], d["phase_train_sub"]
    ph_va = d["phase_val"]
    n_va = Xva.shape[0] // 101
    Xva_seq = Xva.reshape(n_va, 101, 11)
    print(f"data: frames train {Xtr_sub.shape}, val {Xva.shape}")

    base = pd.read_csv(OUT / "r2_vs_k.csv", index_col=0)  # PCA/ICA/NMF/AE

    rows = []
    wave = {}  # method -> (101,11) mean reconstructed envelope at K_WAVE
    real_env = Xva_seq.mean(0)
    for k in K_VALUES:
        # frozen per-frame AE (retrain for a fair same-recipe compare)
        ae = train_ae(Xtr_sub, dim_latent=k, epochs=50, batch_size=512, seed=0)
        ae.eval()
        with torch.no_grad():
            xh_ff, _ = ae(torch.from_numpy(Xva.astype(np.float32)))
        r_ff = r2(Xva, xh_ff.numpy())

        # phase-conditioned FF-AE
        pae = train_ae(Xtr_sub, dim_latent=k, phase_train=ph_sub, epochs=50, batch_size=512, seed=0)
        pae.eval()
        with torch.no_grad():
            xh_p, _ = pae(torch.from_numpy(Xva.astype(np.float32)),
                          phase=torch.from_numpy(ph_va.astype(np.float32)))
        r_p = r2(Xva, xh_p.numpy())

        rows.append(dict(K=k, ff_ae=r_ff, phase_ae=r_p,
                         PCA=base.loc[k, "PCA"], NMF=base.loc[k, "NMF"], AE_cached=base.loc[k, "AE"]))
        print(f"K={k}: per-frame AE={r_ff:.3f}  phase-AE={r_p:.3f}  "
              f"(cached AE={base.loc[k,'AE']:.3f}, PCA={base.loc[k,'PCA']:.3f}, NMF={base.loc[k,'NMF']:.3f})")

        if k == K_WAVE:
            wave["AE (per-frame)"] = xh_ff.numpy().reshape(n_va, 101, 11).mean(0)
            wave["phase-AE"] = xh_p.numpy().reshape(n_va, 101, 11).mean(0)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "temporal_ae_r2_vs_k.csv", index=False)
    print("\n" + df.to_string(index=False))

    # ---- fig 1: R^2 vs K ----
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    series = {"AE (per-frame)": df.ff_ae, "phase-AE": df.phase_ae,
              "PCA": df.PCA, "NMF": df.NMF}
    for name, y in series.items():
        ax.plot(df.K, y, marker=MARK[name], ms=5, lw=1.8, color=PAL[name], label=name,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlabel("number of synergies  K")
    ax.set_ylabel("held-out  $R^2$")
    ax.set_title("Temporal vs per-frame decoders — lower-limb EMG reconstruction")
    ax.set_xticks(K_VALUES)
    ax.legend(ncol=2, loc="lower right", columnspacing=1.0, handletextpad=0.4)
    fig.savefig(OUT / "fig_temporal_r2_vs_k.png")
    plt.close(fig)

    # ---- fig 2: waveform reconstruction at K_WAVE ----
    pct = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.2), sharex=True)
    axf = axes.flatten()
    for i, ch in enumerate(EMG_CHANNELS):
        ax = axf[i]
        ax.plot(pct, real_env[:, i], color=PAL["real"], lw=2.2, label="real EMG", zorder=5)
        for name in ["AE (per-frame)", "phase-AE"]:
            ax.plot(pct, wave[name][:, i], color=PAL[name], lw=1.5, label=name)
        ax.set_title(LABELS[i], fontsize=8)
        ax.set_ylim(-0.02, 1.02); ax.set_yticks([0, 0.5, 1.0])
        if i % 4 == 0:
            ax.set_ylabel("activation (RVC)")
        if i >= 7:
            ax.set_xlabel("gait cycle [%]")
    axf[11].axis("off")
    h, l = axf[0].get_legend_handles_labels()
    axf[11].legend(h, l, loc="center", fontsize=9)
    fig.suptitle(f"Waveform reconstruction at K = {K_WAVE}  —  temporal vs per-frame decoder",
                 fontsize=11, y=1.02)
    fig.savefig(OUT / "fig_temporal_waveforms_k3.png")
    plt.close(fig)
    print(f"\nwrote fig_temporal_r2_vs_k.png, fig_temporal_waveforms_k3.png, temporal_ae_r2_vs_k.csv\ndone.")


if __name__ == "__main__":
    main()
