"""AE vs linear synergy extraction (PCA/ICA/NMF) on lower-limb EMG.

Spüler et al. 2016 (ICANN) replication for the lower limb. Three questions:
  A. reconstruction fit vs K  (does the AE fit better at low-mid K?)
  B. agonist-antagonist capture vs K  (K=2..6; K=2-only was the notebook's TODO)
  C. waveform reconstruction  (the paper's point: PCA gets the *weights* right
     but the reconstructed muscle *waveforms* are worse than the AE's)

    python -m leaps.scripts.ae_vs_linear_synergies

Outputs -> LEAPS/results/synergy_ae_vs_linear/  (publication-styled PNGs @ 300dpi)
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
from sklearn.decomposition import PCA, NMF, FastICA

sys.path.insert(0, "/home/nadinebadie/lalitha/LEAPS/src")
from leaps.envs.emg_mapping import EMG_CHANNELS
from leaps.synergy_common import prepare_training_arrays, train_ae, r2

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parents[3] / "results" / "synergy_ae_vs_linear"
OUT.mkdir(parents=True, exist_ok=True)

AE_EPOCHS = 50
K_TABLE = list(range(1, 12))
K_PAIRS = [2, 3, 4, 5, 6]
K_WAVE = [3, 6]

MUSCLE_LABEL = {
    "gastrocmed": "Gastroc", "tibialisanterior": "Tib. Ant.", "soleus": "Soleus",
    "vastusmedialis": "Vast. Med.", "vastuslateralis": "Vast. Lat.",
    "rectusfemoris": "Rect. Fem.", "bicepsfemoris": "Biceps Fem.",
    "semitendinosus": "Semitend.", "gracilis": "Gracilis",
    "gluteusmedius": "Glut. Med.", "rightexternaloblique": "Ext. Obl.",
}
LABELS = [MUSCLE_LABEL[c] for c in EMG_CHANNELS]

# ---------------------------------------------------------------- publication style
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "#777777",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
})
PAL = {"real": "#3A3A3A", "AE": "#5FA8A0", "PCA": "#8AA9D6", "ICA": "#C89BC4", "NMF": "#E7A08C"}
MARK = {"AE": "D", "PCA": "o", "ICA": "s", "NMF": "^"}
GRP = {"a": "#7FB2D5", "b": "#E7A08C", "other": "#DCDCDC"}
METHODS = ["PCA", "ICA", "NMF", "AE"]

# --- antagonist pairs available in the Camargo channel set ---
IX = {c: EMG_CHANNELS.index(c) for c in EMG_CHANNELS}
PAIRS = [
    ("ankle", [IX["tibialisanterior"]], [IX["gastrocmed"], IX["soleus"]]),
    ("knee", [IX["vastusmedialis"], IX["vastuslateralis"], IX["rectusfemoris"]],
     [IX["bicepsfemoris"], IX["semitendinosus"]]),
    ("hip", [IX["gluteusmedius"]], [IX["gracilis"]]),
]


# ------------------------------------------------------------------ fitting
def fit_models(X_train, k, seed=0):
    """Return (weights, recon_fns): per-method (k,11) weight matrix and a
    reconstruct(X)->X_hat closure (encode to k dims, decode back)."""
    pca = PCA(n_components=k).fit(X_train)
    nmf = NMF(n_components=k, init="nndsvda", max_iter=500, random_state=42).fit(X_train)
    ica = FastICA(n_components=k, random_state=42, max_iter=500).fit(X_train)
    ae = train_ae(X_train, dim_latent=k, epochs=AE_EPOCHS, batch_size=512, seed=seed)
    ae.eval()
    z0 = torch.zeros(1, k, requires_grad=True)
    jac = torch.autograd.functional.jacobian(lambda zz: ae.decoder(zz).squeeze(0), z0)
    ae_w = jac.squeeze(1).T.detach().numpy()

    def ae_recon(X):
        with torch.no_grad():
            xh, _ = ae(torch.from_numpy(X.astype(np.float32)))
        return xh.numpy()

    weights = {"PCA": pca.components_, "ICA": ica.mixing_.T, "NMF": nmf.components_, "AE": ae_w}
    recon = {
        "PCA": lambda X: ((X - pca.mean_) @ pca.components_.T) @ pca.components_ + pca.mean_,
        "ICA": lambda X: ica.inverse_transform(ica.transform(X)),
        "NMF": lambda X: nmf.inverse_transform(nmf.transform(X)),
        "AE": ae_recon,
    }
    return weights, recon


def opp_sign(W, ga, gb):
    a, b = W[:, ga].mean(1), W[:, gb].mean(1)
    prod = a * b
    bd = int(np.argmin(prod))
    return bool(prod[bd] < 0), bd, float(a[bd]), float(b[bd]), float(-prod[bd])


# ============================ A: R^2 vs K ============================
def part_a(Xtr, Xva):
    csv = OUT / "r2_vs_k.csv"
    if csv.exists():
        tbl = pd.read_csv(csv, index_col=0)
    else:
        rows = []
        for k in K_TABLE:
            w, rec = fit_models(Xtr, k)
            for m in METHODS:
                rows.append((m, k, r2(Xva, rec[m](Xva))))
            print(f"[A] K={k:2d} " + " ".join(f"{m}={v:.3f}" for m, _, v in rows[-4:]))
        tbl = (pd.DataFrame(rows, columns=["method", "K", "R2"])
               .pivot(index="K", columns="method", values="R2")[METHODS])
        tbl.to_csv(csv)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for m in METHODS:
        ax.plot(tbl.index, tbl[m], marker=MARK[m], ms=5, lw=1.8, color=PAL[m],
                label=m, markeredgecolor="white", markeredgewidth=0.6)
    ax.axvline(6, color="#B0B0B0", ls=(0, (4, 3)), lw=1)
    ax.annotate("k = 6\n(production)", xy=(6, tbl.min().min()), xytext=(6.25, tbl.min().min() + 0.02),
                fontsize=7.5, color="#888888", va="bottom")
    ax.set_xlabel("number of synergies  K")
    ax.set_ylabel("held-out  $R^2$")
    ax.set_title("Reconstruction fit vs K — lower-limb EMG (11 channels)", pad=10)
    ax.set_xticks(K_TABLE)
    ax.legend(ncol=4, loc="lower right", columnspacing=1.0, handletextpad=0.4)
    fig.savefig(OUT / "fig_r2_vs_k.png")
    plt.close(fig)
    print("[A] fig_r2_vs_k.png")
    return tbl


# ============ B: agonist-antagonist capture at matched LOW K ============
# The advantage has to show at low K (that is where compression happens).
# "best of K dims" gives high-K methods more lottery tickets, so the headline
# comparison is at a fixed low K (K_MAIN) and joint with reconstruction fit.
K_MAIN = 3


def _pairs_captured(W):
    """How many of the 3 antagonist pairs does ANY dim of W represent
    reciprocally (opposite mean sign on the two muscle groups)?"""
    n = 0
    detail = {}
    for pn, ga, gb in PAIRS:
        a, b = W[:, ga].mean(1), W[:, gb].mean(1)
        cap = bool((a * b < 0).any())
        detail[pn] = cap
        n += cap
    return n, detail


def part_b(Xtr, Xva):
    n_val = Xva.shape[0] // 101

    # capture vs K (kept only to show NMF flatlines and AE reaches it early;
    # NOT the headline -- see caveat in the figure).
    rows, W_by_k = [], {}
    for k in K_PAIRS:
        W_by_k[k], rec = fit_models(Xtr, k)
        for m in METHODS:
            nc, det = _pairs_captured(W_by_k[k][m])
            rows.append(dict(K=k, method=m, pairs_captured=nc, **{p: det[p] for p in det}))
        print(f"[B] K={k} fit")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "agonist_antagonist_by_k.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for m in METHODS:
        s = df[df.method == m].sort_values("K")
        ax.plot(s.K, s.pairs_captured, marker=MARK[m], ms=6, lw=1.8, color=PAL[m], label=m,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xticks(K_PAIRS); ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("K"); ax.set_ylabel("antagonist pairs represented\n(of 3)")
    ax.set_title("Antagonist-pair representation vs K", pad=8)
    ax.legend(ncol=4, loc="lower right", columnspacing=1.0, handletextpad=0.4)
    ax.text(0.02, 0.02, "note: more K = more dimensions to search;\nthe fair test is at matched low K (see scatter)",
            transform=ax.transAxes, fontsize=6.8, color="#999999", va="bottom")
    fig.savefig(OUT / "fig_antagonist_vs_k.png")
    plt.close(fig)

    # ---- HEADLINE: at fixed low K, reconstruction fit vs antagonism ----
    Wk, reck = fit_models(Xtr, K_MAIN)
    pts = {}
    for m in METHODS:
        agg = r2(Xva, reck[m](Xva))
        nc, _ = _pairs_captured(Wk[m])
        pts[m] = (agg, nc)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    lbl_off = {"PCA": (9, 8), "ICA": (9, -16), "NMF": (9, 6), "AE": (9, 6)}
    for m in METHODS:
        x, y = pts[m]
        ax.scatter([x], [y], s=170, color=PAL[m], edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(m, (x, y), textcoords="offset points", xytext=lbl_off[m], fontsize=9,
                    color=PAL[m], fontweight="bold")
    ax.set_xlabel(f"held-out reconstruction  $R^2$   (K = {K_MAIN})")
    ax.set_ylabel("antagonist pairs represented  (of 3)")
    ax.set_yticks([0, 1, 2, 3]); ax.set_ylim(-0.3, 3.3)
    ax.set_title(f"At matched low K = {K_MAIN}:  fit  vs  agonist–antagonist structure", pad=10)
    ax.annotate("better fit  →", xy=(0.98, -0.13), xycoords="axes fraction",
                ha="right", fontsize=7.5, color="#999999")
    fig.savefig(OUT / "fig_fit_vs_antagonism_k3.png")
    plt.close(fig)

    # ---- illustrative weights, per pair, at matched K=K_MAIN ----
    # For each pair, show the one synergy dim (of K_MAIN -- all methods have the
    # same number) that most clearly separates the two muscle groups by sign.
    PAIR_LBL = {"ankle": "ankle\n(TA vs plantarflex.)", "knee": "knee\n(quads vs hamstr.)",
                "hip": "hip\n(glut med vs gracilis)"}
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.2), sharex=True)
    for r, (pn, ga, gb) in enumerate(PAIRS):
        for c, m in enumerate(METHODS):
            ax = axes[r][c]
            _, bd, _, _, _ = opp_sign(Wk[m], ga, gb)
            w = Wk[m][bd]
            cols = [GRP["a"] if i in ga else GRP["b"] if i in gb else GRP["other"]
                    for i in range(len(w))]
            ax.bar(range(len(w)), w, color=cols, edgecolor="#00000022", linewidth=0.5)
            ax.axhline(0, color="#666666", lw=0.9)
            ax.set_xticks(range(len(w))); ax.set_xticklabels(LABELS, rotation=90, fontsize=6.5)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(m, pad=8)
            if c == 0:
                ax.set_ylabel(PAIR_LBL[pn], fontsize=8)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=GRP["a"], label="agonist group"),
                        Patch(fc=GRP["b"], label="antagonist group"),
                        Patch(fc=GRP["other"], label="other muscles")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"Synergy weights at matched K = {K_MAIN}  —  clearest dim per pair "
                 f"(reciprocal inhibition = blue and red on opposite sides of 0)",
                 fontsize=11, y=1.06)
    fig.savefig(OUT / f"fig_weights_k{K_MAIN}_per_pair.png")
    plt.close(fig)
    print("[B] figs written")
    return df


# ===================== C: waveform reconstruction =====================
def part_c(Xtr, Xva):
    n_val = Xva.shape[0] // 101
    real_env = Xva.reshape(n_val, 101, 11).mean(0)         # (101,11)
    pct = np.linspace(0, 100, 101)
    per_musc_rows = []

    for k in K_WAVE:
        _, rec = fit_models(Xtr, k)
        recon_env, chan_r2 = {}, {}
        for m in METHODS:
            xh = rec[m](Xva)
            recon_env[m] = xh.reshape(n_val, 101, 11).mean(0)
            chan_r2[m] = [r2(Xva[:, i], xh[:, i]) for i in range(11)]
            for i, ch in enumerate(EMG_CHANNELS):
                per_musc_rows.append(dict(K=k, method=m, muscle=ch, channel_R2=round(chan_r2[m][i], 4)))

        # 3 x 4 grid of muscles (11 used, last cell = legend)
        fig, axes = plt.subplots(3, 4, figsize=(12, 7.2), sharex=True)
        axf = axes.flatten()
        for i, ch in enumerate(EMG_CHANNELS):
            ax = axf[i]
            ax.plot(pct, real_env[:, i], color=PAL["real"], lw=2.2, label="real EMG", zorder=5)
            for m in METHODS:
                ax.plot(pct, recon_env[m][:, i], color=PAL[m], lw=1.5, label=m, alpha=0.95)
            ax.set_title(f"{LABELS[i]}   (AE $R^2$={chan_r2['AE'][i]:.2f}, "
                         f"best lin {max(chan_r2['PCA'][i], chan_r2['NMF'][i]):.2f})", fontsize=8)
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0, 0.5, 1.0])
            if i % 4 == 0:
                ax.set_ylabel("activation (RVC)")
            if i >= 7:
                ax.set_xlabel("gait cycle [%]")
        axf[11].axis("off")
        h, l = axf[0].get_legend_handles_labels()
        axf[11].legend(h, l, loc="center", fontsize=9)
        agg = {m: r2(Xva, rec[m](Xva)) for m in METHODS}
        fig.suptitle(f"Muscle-activation waveform reconstruction  —  K = {k}   "
                     f"(aggregate $R^2$:  " + "  ".join(f"{m} {agg[m]:.3f}" for m in METHODS) + ")",
                     fontsize=11, y=1.02)
        fig.savefig(OUT / f"fig_waveform_reconstruction_k{k}.png")
        plt.close(fig)
        print(f"[C] fig_waveform_reconstruction_k{k}.png")

    pm = pd.DataFrame(per_musc_rows)
    pm.to_csv(OUT / "per_muscle_reconstruction_r2.csv", index=False)

    # --- summary heatmap: method x muscle channel R^2, one row-block per K ---
    fig, axes = plt.subplots(len(K_WAVE), 1, figsize=(9.5, 2.2 * len(K_WAVE) + 1))
    if len(K_WAVE) == 1:
        axes = [axes]
    for ax, k in zip(axes, K_WAVE):
        M = pm[pm.K == k].pivot(index="method", columns="muscle", values="channel_R2").reindex(METHODS)
        M = M[EMG_CHANNELS]
        im = ax.imshow(M.values, cmap="BuGn", vmin=0.3, vmax=1.0, aspect="auto")
        ax.set_xticks(range(11)); ax.set_xticklabels(LABELS, rotation=90, fontsize=7)
        ax.set_yticks(range(4)); ax.set_yticklabels(METHODS, fontsize=8)
        ax.set_title(f"per-channel held-out $R^2$   (K = {k})", fontsize=9, pad=6)
        ax.grid(False)
        for yy in range(4):
            for xx in range(11):
                v = M.values[yy, xx]
                ax.text(xx, yy, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.72 else "#333333")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="$R^2$")
    fig.suptitle("Per-muscle waveform reconstruction fidelity", fontsize=11, y=1.0)
    fig.savefig(OUT / "fig_per_muscle_r2_heatmap.png")
    plt.close(fig)
    print("[C] fig_per_muscle_r2_heatmap.png")
    return pm


def main():
    d = prepare_training_arrays()
    Xtr, Xva = d["X_train_sub"], d["X_val"]
    print(f"data: train {Xtr.shape}, val {Xva.shape}")
    part_a(Xtr, Xva)
    part_b(Xtr, Xva)
    part_c(Xtr, Xva)
    print("\ndone.")


if __name__ == "__main__":
    main()
