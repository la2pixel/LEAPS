"""Synergy extraction (PCA/ICA/NMF/AE) on all 4 modes.

Normalization: each channel is divided by one global divisor, built by
averaging every stride's own P99 (per stride, per channel) across the
training pool. This keeps genuine amplitude differences between conditions
(e.g. stair vs treadmill) while diluting single-frame artifacts, since a
lone spike frame barely moves its own stride's P99.
"""

import argparse


import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA, NMF, FastICA

from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides, stride_train_val_split
from leaps.synergy_common import CACHE_DIR, r2, train_ae

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description="Fit synergy decompositions and freeze an AE decoder for RL.")
parser.add_argument("--k-check", type=int, default=6, help="latent dim of the frozen decoder to save for RL")
parser.add_argument("--skip-sweep", action="store_true", help="skip the K=1..11 PCA/NMF/ICA/AE comparison sweep")
parser.add_argument("--skip-freeze", action="store_true", help="diagnostic-only run -- don't overwrite decoder_k<K>/ on disk")
args = parser.parse_args()

# Data: all 4 modes pooled, subject-level held-out split (4 ppl)
strides, meta = load_strides(LEAPS_H5_PATH, modes=ALL_MODES, with_metadata=True)
strides_train, strides_val, meta_train, meta_val = stride_train_val_split(strides, meta, val_fraction=0.2, seed=42)

# Per-stride-P99 normalization, fit on train only
stride_p99s = np.percentile(strides_train, 99, axis=1)     # (n_train_strides, 11)
global_divisors = stride_p99s.mean(axis=0) + 1e-8          # (11,)


def to_unit(strides_3d: np.ndarray) -> np.ndarray:
    return np.clip(strides_3d / global_divisors, 0.0, 1.0).astype(np.float32)


X_train = to_unit(strides_train).reshape(-1, 11)
X_val = to_unit(strides_val).reshape(-1, 11)

# Gait-phase encoding, free from the stride layout: reshape(-1, 11) above
# flattens (n_strides, 101, 11) stride-major/frame-minor, so row i's phase
# is exactly (i % 101)/100 -- verified directly against the source h5 (each
# stride is time-normalized to 101 points, 0%-100% of the gait cycle).
# sin/cos, not a raw fraction, avoids a fake discontinuity at 100%->0%.
def _phase_sincos(n_strides: int) -> np.ndarray:
    frac = np.tile(np.linspace(0.0, 1.0, 101, dtype=np.float32), n_strides)
    return np.stack([np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)], axis=1)


phase_train = _phase_sincos(strides_train.shape[0])
phase_val = _phase_sincos(strides_val.shape[0])

# same training subsample used for every method, for a fair comparison
rng_sub = np.random.default_rng(0)
sub_idx = rng_sub.choice(len(X_train), size=50_000, replace=False)
X_train_sub = X_train[sub_idx]
phase_train_sub = phase_train[sub_idx]

# ICA needs its own subsample of the (already subsampled) training set
rng_ica = np.random.default_rng(42)
ica_sub_idx = rng_ica.choice(len(X_train_sub), size=min(20_000, len(X_train_sub)), replace=False)
X_train_ica_sub = X_train_sub[ica_sub_idx]

# ---------------------------------------------------------------------------
# Sweep K=1..11 for all four methods, same training subsample, same held-out set
# ---------------------------------------------------------------------------
if not args.skip_sweep:
    K_VALUES = list(range(1, 12))
    rows = []

    for K in K_VALUES:
        # PCA
        pca = PCA(n_components=K)
        pca.fit(X_train_sub)
        W = pca.components_
        X_hat = ((X_val - pca.mean_) @ W.T) @ W + pca.mean_
        rows.append({"K": K, "method": "PCA", "held_out_R2": r2(X_val, X_hat)})

        # NMF (data already in [0,1], no clipping needed)
        nmf = NMF(n_components=K, init="nndsvda", max_iter=500, random_state=42)
        nmf.fit(X_train_sub)
        X_hat = nmf.inverse_transform(nmf.transform(X_val))
        rows.append({"K": K, "method": "NMF", "held_out_R2": r2(X_val, X_hat)})

        # ICA
        ica = FastICA(n_components=K, random_state=42, max_iter=500)
        ica.fit(X_train_ica_sub)
        X_hat = ica.inverse_transform(ica.transform(X_val))
        rows.append({"K": K, "method": "ICA", "held_out_R2": r2(X_val, X_hat)})

        # AE, finalized architecture + penalty (see synergy_common.SynergyAE)
        model = train_ae(X_train_sub, dim_latent=K, epochs=50, batch_size=512, seed=0)
        model.eval()
        with torch.no_grad():
            val_hat, _ = model(torch.from_numpy(X_val))
        rows.append({"K": K, "method": "AE", "held_out_R2": r2(X_val, val_hat.numpy())})

        # Same AE, conditioned on gait phase -- tests whether frame-level
        # pooling (no cycle context) is what's capping reconstruction
        # quality. Diagnostic only: the frozen decoder saved below for RL
        # is unaffected, still trained phase-blind, until this comparison
        # justifies wiring a phase source into the RL rollout itself.
        model_phase = train_ae(X_train_sub, dim_latent=K, phase_train=phase_train_sub, epochs=50, batch_size=512, seed=0)
        model_phase.eval()
        with torch.no_grad():
            val_hat_phase, _ = model_phase(torch.from_numpy(X_val), torch.from_numpy(phase_val))
        rows.append({"K": K, "method": "AE_phase", "held_out_R2": r2(X_val, val_hat_phase.numpy())})

        print(f"K={K} done: " + ", ".join(f"{r['method']}={r['held_out_R2']:.3f}" for r in rows[-5:]))

    comparison_table = pd.DataFrame(rows).pivot(index="K", columns="method", values="held_out_R2")
    comparison_table = comparison_table[["PCA", "ICA", "NMF", "AE", "AE_phase"]]
    print()
    print(comparison_table)

# ---------------------------------------------------------------------------
# Per-mode breakdown at a fixed K, to check whether one mode is dominating
# ---------------------------------------------------------------------------
K_CHECK = args.k_check
# PCA/NMF/ICA can't have more components than the 11 EMG channels -- only
# meaningful as a comparison baseline for K_CHECK <= 11. The AE has no such
# limit (it's just an overcomplete autoencoder above K=11) and is the only
# one actually frozen for RL below, so it's unaffected.
classical_methods_valid = K_CHECK <= 11
if classical_methods_valid:
    pca_c = PCA(n_components=K_CHECK); pca_c.fit(X_train_sub)
    nmf_c = NMF(n_components=K_CHECK, init="nndsvda", max_iter=500, random_state=42); nmf_c.fit(X_train_sub)
    ica_c = FastICA(n_components=K_CHECK, random_state=42, max_iter=500); ica_c.fit(X_train_ica_sub)
ae_c = train_ae(X_train_sub, dim_latent=K_CHECK, epochs=50, batch_size=512, seed=0)
ae_c.eval()

# ---------------------------------------------------------------------------
# Freeze the K_CHECK decoder for RL (leaps.envs.LatentActionPriorWrapper)
# ---------------------------------------------------------------------------
for p in ae_c.decoder.parameters():
    p.requires_grad = False

if args.skip_freeze:
    print(f"\n--skip-freeze: not touching decoder_k{K_CHECK}/ on disk (diagnostic-only run).")
else:
    decoder_dir = CACHE_DIR / f"decoder_k{K_CHECK}"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ae_c.decoder.state_dict(), decoder_dir / "decoder.pt")

    # z_lo/z_hi: the actual per-dim latent range the decoder was trained on
    # (encoder has no output activation, only the soft latent_norm_penalty --
    # so this is roughly [-0.55, 0.55], not [0, 1]). LatentActionPriorWrapper
    # rescales its policy's [0,1] z output into this range before decoding.
    with torch.no_grad():
        _, z_train = ae_c(torch.from_numpy(X_train_sub))
    z_train = z_train.numpy()

    np.savez(
        decoder_dir / "norm.npz",
        p01=np.zeros(11, dtype=np.float32), p99=global_divisors.astype(np.float32),
        z_lo=z_train.min(axis=0).astype(np.float32), z_hi=z_train.max(axis=0).astype(np.float32),
    )
    print(f"\nFrozen decoder (K={K_CHECK}) saved to {decoder_dir}")

mode_rows = []
val_modes = meta_val["modes"]
for mode in ALL_MODES:
    mode_mask = val_modes == mode
    if not mode_mask.any():
        continue
    X_val_mode = to_unit(strides_val[mode_mask]).reshape(-1, 11)

    if classical_methods_valid:
        W = pca_c.components_
        X_hat = ((X_val_mode - pca_c.mean_) @ W.T) @ W + pca_c.mean_
        mode_rows.append({"mode": mode, "method": "PCA", "R2": r2(X_val_mode, X_hat)})

        X_hat = nmf_c.inverse_transform(nmf_c.transform(X_val_mode))
        mode_rows.append({"mode": mode, "method": "NMF", "R2": r2(X_val_mode, X_hat)})

        X_hat = ica_c.inverse_transform(ica_c.transform(X_val_mode))
        mode_rows.append({"mode": mode, "method": "ICA", "R2": r2(X_val_mode, X_hat)})

    with torch.no_grad():
        X_hat, _ = ae_c(torch.from_numpy(X_val_mode))
    mode_rows.append({"mode": mode, "method": "AE", "R2": r2(X_val_mode, X_hat.numpy())})

mode_table = pd.DataFrame(mode_rows).pivot(index="mode", columns="method", values="R2")
mode_table = mode_table[["PCA", "ICA", "NMF", "AE"] if classical_methods_valid else ["AE"]]
print(f"\nper-mode held-out R2 at K={K_CHECK}:")
print(mode_table)
