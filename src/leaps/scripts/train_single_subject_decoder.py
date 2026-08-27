"""Train a synergy-AE decoder on one Camargo subject instead of all 22 pooled.

Tests whether cross-subject pooling (get_synergies.py's approach) is what
makes the frozen decoder uninformative for RL, by isolating that one
variable: same architecture, same K, same modes, same normalization
convention -- just one subject's data instead of 22 pooled together.

Subject choice: closest morphology match to the target SCONE model by
(height, mass) z-score distance against the Camargo population stats.
"""

import argparse
import os

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import torch
import yaml

from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH, SUBJECTS
from leaps.data.stride_dataset import load_strides
from leaps.synergy_common import CACHE_DIR, train_ae

parser = argparse.ArgumentParser(description="Train a single-subject synergy decoder.")
parser.add_argument("--subject", default="AB06", help="Camargo subject ID (default: AB06, closest match to h0918)")
parser.add_argument("--k", type=int, default=11, help="latent dim (default: 11, matching the hardconstraint11 test)")
parser.add_argument("--val-fraction", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--strides-per-mode", type=int, default=None,
    help="If set, subsample to N strides per mode (e.g. 1 for a Hausdorfer-scale "
         "minimal-data probe) instead of using every stride the subject has.",
)
parser.add_argument("--threshold", type=float, default=0.5, help="latent_norm_penalty dead zone")
parser.add_argument("--scale", type=float, default=0.6, help="latent_norm_penalty ramp scale")
parser.add_argument("--depth", type=int, default=1, help="hidden layers per side (SynergyAE)")
parser.add_argument(
    "--suffix", default="", help="appended to the output dir name, e.g. '_corrected' "
    "to avoid overwriting an existing decoder_k{k}_{subject} in place",
)
args = parser.parse_args()

if args.subject not in SUBJECTS:
    raise ValueError(f"{args.subject} not in Camargo SUBJECTS: {sorted(SUBJECTS)}")

# --- data: one subject, all 4 modes, same pooling-across-modes as get_synergies.py ---
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

# Random stride-level train/val split -- subject-level split (get_synergies.py's
# default) is meaningless with n=1 subject, so this uses the same fallback path
# stride_train_val_split() already has for when there's only one subject's worth
# of metadata, just done explicitly here since we never pass subject metadata in.
rng = np.random.default_rng(args.seed)
idx = rng.permutation(len(strides))
n_val = max(1, int(len(strides) * args.val_fraction))
strides_val, strides_train = strides[idx[:n_val]], strides[idx[n_val:]]

# --- per-subject P99 normalization, fit on this subject's train split only ---
# Same convention as get_synergies.py (per-stride P99, averaged across the
# training pool) but computed from this one subject instead of the pooled 22.
stride_p99s = np.percentile(strides_train, 99, axis=1)
global_divisors = stride_p99s.mean(axis=0) + 1e-8


def to_unit(strides_3d: np.ndarray) -> np.ndarray:
    return np.clip(strides_3d / global_divisors, 0.0, 1.0).astype(np.float32)


X_train = to_unit(strides_train).reshape(-1, 11)
X_val = to_unit(strides_val).reshape(-1, 11)

# Match get_synergies.py's subsample size where possible (50k), but don't
# oversample past what one subject actually has.
n_sub = min(50_000, len(X_train))
rng_sub = np.random.default_rng(0)
sub_idx = rng_sub.choice(len(X_train), size=n_sub, replace=False)
X_train_sub = X_train[sub_idx]
print(f"training on {n_sub} frames (val: {len(X_val)} frames)")

# --- train + freeze, same hyperparameters as get_synergies.py's frozen-decoder path ---
model = train_ae(
    X_train_sub, dim_latent=args.k, epochs=50, batch_size=512, seed=0,
    threshold=args.threshold, scale=args.scale, depth=args.depth,
)
model.eval()

for p in model.decoder.parameters():
    p.requires_grad = False

suffix = f"_spm{args.strides_per_mode}_seed{args.seed}" if args.strides_per_mode is not None else ""
decoder_dir = CACHE_DIR / f"decoder_k{args.k}_{args.subject}{suffix}{args.suffix}"
decoder_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.decoder.state_dict(), decoder_dir / "decoder.pt")

with torch.no_grad():
    train_hat, z_train = model(torch.from_numpy(X_train_sub))
    val_hat, _ = model(torch.from_numpy(X_val))

train_hat, val_hat = train_hat.numpy(), val_hat.numpy()
ss_res = ((X_val - val_hat) ** 2).sum()
ss_tot = ((X_val - X_val.mean(axis=0)) ** 2).sum()
held_out_r2 = 1 - ss_res / ss_tot
print(f"held-out R2 (this subject's own val split): {held_out_r2:.4f}")

# Same reporting convention as the reference LAP repo's own
# expert_demonstrations/*/results.yaml (fit-set vs held-out MSE/MAE) --
# 2026-08-19, see project_leaps_overview.md memory for why this comparison
# matters (their single-cycle recipe shows a ~99,000x fit/held-out MSE
# gap; ours, trained on many strides across all 4 modes instead of one
# cycle, does not -- NOT a like-for-like difficulty comparison though,
# their held-out set is the full episode including non-cyclic dynamics
# never in their training crop, ours is a same-distribution random split).
train_mse = float(((X_train_sub - train_hat) ** 2).mean())
train_mae = float(np.abs(X_train_sub - train_hat).mean())
val_mse = float(((X_val - val_hat) ** 2).mean())
val_mae = float(np.abs(X_val - val_hat).mean())
print(f"fit-set MSE/MAE: {train_mse:.6e} / {train_mae:.6e}")
print(f"held-out MSE/MAE: {val_mse:.6e} / {val_mae:.6e}  (ratio vs fit-set: {val_mse/train_mse:.2f}x)")

z_train = z_train.numpy()

# Whitening params (see notebooks/autoencoder_loss.ipynb): the raw latent
# dims are correlated (real synergies aren't independent), so an
# axis-aligned [z_lo, z_hi] box wastes volume on combinations that never
# occur in real data. whiten_A/whiten_mu let latent_env.py map the policy's
# [-1,1] tanh output directly onto a rotated +/-3sigma box that matches the
# real data's covariance instead: z = whiten_A @ z_u + whiten_mu.
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
}
with open(decoder_dir / "results.yaml", "w") as f:
    yaml.dump(results, f, sort_keys=False)

print(f"saved decoder + norm + results to {decoder_dir}")
