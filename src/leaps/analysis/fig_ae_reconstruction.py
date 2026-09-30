"""Thesis figure: production AE decoder (AB06, k=6) reconstructing held-out AB06 strides, all 11 channels.
Rebuilds the AE with the exact train_single_subject_decoder.py recipe (seed 0) and asserts its decoder
equals the saved one, since only the decoder is persisted."""
import numpy as np
import torch
from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides
from leaps.synergy_common import train_ae
from leaps.paths import FIGURES_DIR, PRIORS_DIR

DEC = str(PRIORS_DIR / "decoder_k6_AB06_corrected")
OUT_DIR = str(FIGURES_DIR)

strides, meta = load_strides(LEAPS_H5_PATH, subjects=["AB06"], modes=ALL_MODES, with_metadata=True)
rng = np.random.default_rng(0)
idx = rng.permutation(len(strides))
n_val = max(1, int(len(strides) * 0.2))
val_i, train_i = idx[:n_val], idx[n_val:]
s_val, s_train = strides[val_i], strides[train_i]

div = np.percentile(s_train, 99, axis=1).mean(axis=0) + 1e-8
to_unit = lambda s: np.clip(s / div, 0.0, 1.0).astype(np.float32)
X_train, X_val = to_unit(s_train).reshape(-1, 11), to_unit(s_val).reshape(-1, 11)
sub = np.random.default_rng(0).choice(len(X_train), size=min(50_000, len(X_train)), replace=False)

model = train_ae(X_train[sub], dim_latent=6, epochs=50, batch_size=512, seed=0, threshold=0.6, scale=0.9, depth=1)
model.eval()

saved = torch.load(f"{DEC}/decoder.pt")
same = all(torch.equal(saved[k], v) for k, v in model.decoder.state_dict().items())
assert same, "retrained decoder differs from saved decoder.pt"
print("divisors match norm.npz:", np.allclose(np.load(f"{DEC}/norm.npz")["p99"], div, atol=1e-5))

with torch.no_grad():
    rec, _ = model(torch.from_numpy(X_val))
rec = rec.numpy()
r2 = 1 - ((X_val - rec) ** 2).sum() / ((X_val - X_val.mean(0)) ** 2).sum()
print("held-out R2:", round(float(r2), 4))

real, rec = X_val.reshape(-1, 101, 11), rec.reshape(-1, 101, 11)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = ["GAS", "TA", "SOL", "VM", "VL", "RF", "BF", "ST", "GRA", "GMED", "EO"]

INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e6e5e1"
REAL, AE = "#52514e", "#2a78d6"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.labelcolor": INK, "pdf.fonttype": 42,
})

x = np.linspace(0, 100, 101)
fig, axes = plt.subplots(3, 4, figsize=(6.3, 4.1), sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    if i == 11:
        ax.axis("off")
        continue
    r, h = real[:, :, i], rec[:, :, i]
    r2 = 1 - ((r - h) ** 2).sum() / ((r - r.mean()) ** 2).sum()
    m, s = r.mean(0), r.std(0)
    ax.fill_between(x, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=REAL, alpha=0.15, lw=0)
    ax.plot(x, m, color=REAL, lw=1.6, label="EMG (held out)")
    ax.plot(x, h.mean(0), color=AE, lw=1.6, ls=(0, (4, 1.5)), label="AE reconstruction")
    ax.text(0.04, 0.95, NAMES[i], transform=ax.transAxes, va="top", fontsize=8, fontweight="bold", color=INK)
    ax.text(0.96, 0.95, f"$R^2$ = {r2:.2f}", transform=ax.transAxes, va="top", ha="right", fontsize=7, color=MUTED)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

for ax in axes[-1]:
    ax.set_xlabel("Gait cycle (%)")
axes[-2, -1].tick_params(labelbottom=True)
axes[-2, -1].set_xlabel("Gait cycle (%)")
for ax in axes[:, 0]:
    ax.set_ylabel("Activation (norm.)")

h, l = axes[0, 0].get_legend_handles_labels()
axes[-1, -1].legend(h, l, loc="center", frameon=False, fontsize=8, handlelength=2.6)

fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.5)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT_DIR}/ae_reconstruction_k6.{ext}", dpi=300)
