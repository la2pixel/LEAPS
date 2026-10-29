"""Train the EMG autoencoder on one Camargo subject and save its frozen decoder.

Writes priors/decoder_k{k}_{subject}{suffix}/:
    decoder.pt, encoder.pt   state dicts (the decoder is the prior used in RL)
    norm.npz                 P99 divisors, latent range, whitening params
    results.yaml             fit-set and held-out reconstruction error
    ae_training.pdf          training and held-out loss per epoch
    pca.pdf                  cumulative explained variance of the training frames

Subject choice: AB06 is the closest match to the SCONE models by (height,
mass) z-score distance against the Camargo population.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA

from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH, SUBJECTS
from leaps.data.stride_dataset import load_strides
from leaps.paths import PRIORS_DIR
from leaps.models.autoencoder import train_ae

EPOCHS = 50

parser = argparse.ArgumentParser(description="Train a single-subject synergy decoder.")
parser.add_argument("--subject", default="AB06", help="Camargo subject ID (default: AB06)")
parser.add_argument("--k", type=int, default=6, help="latent dim (default: 6)")
parser.add_argument("--val-fraction", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--strides-per-mode", type=int, default=None,
    help="If set, subsample to N strides per mode (e.g. 1 for a Hausdorfer-scale "
         "minimal-data probe) instead of using every stride the subject has.",
)
parser.add_argument("--threshold", type=float, default=0.6, help="latent_norm_penalty dead zone")
parser.add_argument("--scale", type=float, default=0.9, help="latent_norm_penalty ramp scale")
parser.add_argument("--depth", type=int, default=1, help="hidden layers per side (SynergyAE)")
parser.add_argument(
    "--suffix", default="", help="appended to the output dir name, e.g. '_corrected' "
    "to avoid overwriting an existing decoder_k{k}_{subject} in place",
)
args = parser.parse_args()

if args.subject not in SUBJECTS:
    raise ValueError(f"{args.subject} not in Camargo SUBJECTS: {sorted(SUBJECTS)}")

# --- data: one subject, all 4 modes ---
if args.strides_per_mode is not None:
    strides, meta = load_strides(
        LEAPS_H5_PATH, subjects=[args.subject], modes=ALL_MODES, with_metadata=True,
    )
    rng_mode = np.random.default_rng(args.seed)
    keep_idx = []
    for mode in ALL_MODES:
        mode_idx = np.where(meta["modes"] == mode)[0]
        n_take = min(args.strides_per_mode, len(mode_idx))
        if n_take < args.strides_per_mode:
            print(f"warning: only {len(mode_idx)} strides available for mode {mode}, using all of them")
        keep_idx.extend(rng_mode.choice(mode_idx, size=n_take, replace=False).tolist())
    strides = strides[keep_idx]
else:
    strides = load_strides(LEAPS_H5_PATH, subjects=[args.subject], modes=ALL_MODES)
print(f"subject {args.subject}: {len(strides)} strides ({len(strides) * strides.shape[1]} frames)")

# random stride-level train/val split
rng = np.random.default_rng(args.seed)
idx = rng.permutation(len(strides))
n_val = max(1, int(len(strides) * args.val_fraction))
strides_val, strides_train = strides[idx[:n_val]], strides[idx[n_val:]]

# --- decoder-input scaling: per-channel mean of the per-stride P99, train split only ---
stride_p99s = np.percentile(strides_train, 99, axis=1)
global_divisors = stride_p99s.mean(axis=0) + 1e-8


def to_unit(strides_3d: np.ndarray) -> np.ndarray:
    return np.clip(strides_3d / global_divisors, 0.0, 1.0).astype(np.float32)


X_train = to_unit(strides_train).reshape(-1, 11)
X_val = to_unit(strides_val).reshape(-1, 11)

n_sub = min(50_000, len(X_train))
rng_sub = np.random.default_rng(0)
sub_idx = rng_sub.choice(len(X_train), size=n_sub, replace=False)
X_train_sub = X_train[sub_idx]
print(f"training on {n_sub} frames (val: {len(X_val)} frames)")

decoder_dir = PRIORS_DIR / (
    f"decoder_k{args.k}_{args.subject}"
    + (f"_spm{args.strides_per_mode}_seed{args.seed}" if args.strides_per_mode is not None else "")
    + args.suffix
)
decoder_dir.mkdir(parents=True, exist_ok=True)

# --- PCA of the training frames: how many linear components the 11 channels need ---
pca_cumvar = np.cumsum(PCA().fit(X_train_sub).explained_variance_ratio_)
plt.figure(figsize=(4, 3))
plt.plot(range(1, 12), pca_cumvar, marker="o", ms=3)
plt.axvline(args.k, color="gray", ls="--", lw=0.8)
plt.xlabel("Principal components")
plt.ylabel("Cumulative explained variance")
plt.tight_layout()
plt.savefig(decoder_dir / "pca.pdf")
plt.close()

# --- train + freeze ---
history = []
model = train_ae(
    X_train_sub, dim_latent=args.k, epochs=EPOCHS, batch_size=512, seed=0,
    threshold=args.threshold, scale=args.scale, depth=args.depth,
    X_val=X_val, history=history,
)
model.eval()

for p in model.decoder.parameters():
    p.requires_grad = False

torch.save(model.decoder.state_dict(), decoder_dir / "decoder.pt")
torch.save(model.encoder.state_dict(), decoder_dir / "encoder.pt")

plt.figure(figsize=(4, 3))
plt.plot(range(1, EPOCHS + 1), [h[0] for h in history], label="training (incl. latent penalty)")
plt.plot(range(1, EPOCHS + 1), [h[1] for h in history], label="held-out reconstruction")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(decoder_dir / "ae_training.pdf")
plt.close()

with torch.no_grad():
    train_hat, z_train = model(torch.from_numpy(X_train_sub))
    val_hat, _ = model(torch.from_numpy(X_val))

train_hat, val_hat = train_hat.numpy(), val_hat.numpy()
ss_res = ((X_val - val_hat) ** 2).sum()
ss_tot = ((X_val - X_val.mean(axis=0)) ** 2).sum()
held_out_r2 = 1 - ss_res / ss_tot
print(f"held-out R2 (this subject's own val split): {held_out_r2:.4f}")

# fit-set vs held-out error, as in the reference LAP repo's results.yaml
train_mse = float(((X_train_sub - train_hat) ** 2).mean())
train_mae = float(np.abs(X_train_sub - train_hat).mean())
val_mse = float(((X_val - val_hat) ** 2).mean())
val_mae = float(np.abs(X_val - val_hat).mean())
print(f"fit-set MSE/MAE: {train_mse:.6e} / {train_mae:.6e}")
print(f"held-out MSE/MAE: {val_mse:.6e} / {val_mae:.6e}  (ratio vs fit-set: {val_mse/train_mse:.2f}x)")

z_train = z_train.numpy()

# whitening params of the training latents: z = whiten_A @ z_u + whiten_mu
# would map [-1, 1] onto a rotated +/-3 sigma box (not used by the wrapper)
whiten_mu = z_train.mean(axis=0)
Sigma = np.cov(z_train, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(Sigma)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
whiten_A = 3.0 * (eigvecs @ np.diag(np.sqrt(eigvals)))
volume_ratio = np.sqrt(np.linalg.det(Sigma / np.sqrt(np.outer(np.diag(Sigma), np.diag(Sigma)))))
print(f"whitening: volume_ratio (raw box vs real data) = {volume_ratio:.3f}")

np.savez(
    decoder_dir / "norm.npz",
    p01=np.zeros(11, dtype=np.float32), p99=global_divisors.astype(np.float32),
    z_lo=z_train.min(axis=0).astype(np.float32), z_hi=z_train.max(axis=0).astype(np.float32),
    whiten_A=whiten_A.astype(np.float32), whiten_mu=whiten_mu.astype(np.float32),
)

results = {
    "subject": args.subject,
    "latent_dim": args.k,
    "depth": args.depth,
    "threshold": args.threshold,
    "scale": args.scale,
    "seed": args.seed,
    "num_strides_total": int(len(strides)),
    "num_strides_train": int(len(strides_train)),
    "num_strides_val": int(len(strides_val)),
    "num_frames_fit": int(len(X_train_sub)),
    "num_frames_val": int(len(X_val)),
    "fit_mse": train_mse,
    "fit_mae": train_mae,
    "held_out_mse": val_mse,
    "held_out_mae": val_mae,
    "held_out_r2": float(held_out_r2),
    "held_out_vs_fit_mse_ratio": float(val_mse / train_mse),
    "whiten_volume_ratio": float(volume_ratio),
    "pca_cumulative_variance": [round(float(v), 4) for v in pca_cumvar],
}
with open(decoder_dir / "results.yaml", "w") as f:
    yaml.dump(results, f, sort_keys=False)

print(f"saved decoder + norm + results to {decoder_dir}")
