"""Ablate decoder depth and latent-penalty tightness against the reference
LAP repo's nonLinearAE (github.com/OliEfr/latent-action-priors), on the same
AB06 single-subject data/split as decoder_k6_AB06.

Three variants x 3 seeds each:
  baseline      -- current production settings (depth=1, threshold=0.5, scale=0.6)
  deeper        -- depth=2, matching their nonLinearAE hidden-layer count
  loose_penalty -- threshold=0.8, scale=1.2, matching their penalty constants
"""

import os

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import torch

from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides
from leaps.synergy_common import train_ae

SUBJECT = "AB06"
K = 6
VAL_FRACTION = 0.2

strides = load_strides(LEAPS_H5_PATH, subjects=[SUBJECT], modes=ALL_MODES)
print(f"subject {SUBJECT}: {len(strides)} strides")

VARIANTS = {
    "baseline":      dict(depth=1, threshold=0.5, scale=0.6),
    "deeper":        dict(depth=2, threshold=0.5, scale=0.6),
    "loose_penalty": dict(depth=1, threshold=0.8, scale=1.2),
}
SEEDS = [0, 1, 2]

results = []
for variant_name, hp in VARIANTS.items():
    for split_seed in SEEDS:
        rng = np.random.default_rng(split_seed)
        idx = rng.permutation(len(strides))
        n_val = max(1, int(len(strides) * VAL_FRACTION))
        strides_val, strides_train = strides[idx[:n_val]], strides[idx[n_val:]]

        stride_p99s = np.percentile(strides_train, 99, axis=1)
        global_divisors = stride_p99s.mean(axis=0) + 1e-8

        def to_unit(strides_3d):
            return np.clip(strides_3d / global_divisors, 0.0, 1.0).astype(np.float32)

        X_train = to_unit(strides_train).reshape(-1, 11)
        X_val = to_unit(strides_val).reshape(-1, 11)

        n_sub = min(50_000, len(X_train))
        rng_sub = np.random.default_rng(0)
        sub_idx = rng_sub.choice(len(X_train), size=n_sub, replace=False)
        X_train_sub = X_train[sub_idx]

        model = train_ae(
            X_train_sub, dim_latent=K, epochs=50, batch_size=512, seed=split_seed,
            depth=hp["depth"], threshold=hp["threshold"], scale=hp["scale"],
        )
        model.eval()

        with torch.no_grad():
            val_hat, z_val = model(torch.from_numpy(X_val))
            train_hat, z_train = model(torch.from_numpy(X_train_sub))

        val_hat, z_val = val_hat.numpy(), z_val.numpy()
        train_hat = train_hat.numpy()

        ss_res_val = ((X_val - val_hat) ** 2).sum()
        ss_tot_val = ((X_val - X_val.mean(axis=0)) ** 2).sum()
        r2_val = 1 - ss_res_val / ss_tot_val

        ss_res_train = ((X_train_sub - train_hat) ** 2).sum()
        ss_tot_train = ((X_train_sub - X_train_sub.mean(axis=0)) ** 2).sum()
        r2_train = 1 - ss_res_train / ss_tot_train

        results.append(dict(
            variant=variant_name, seed=split_seed,
            r2_val=r2_val, r2_train=r2_train,
            z_lo=z_val.min(axis=0), z_hi=z_val.max(axis=0),
        ))
        print(f"{variant_name:14s} seed={split_seed}  val_R2={r2_val:.4f}  train_R2={r2_train:.4f}  "
              f"z_range=[{z_val.min():.3f}, {z_val.max():.3f}]")

print("\n=== summary (mean +/- std over seeds) ===")
for variant_name in VARIANTS:
    vals = [r["r2_val"] for r in results if r["variant"] == variant_name]
    print(f"{variant_name:14s} val_R2 = {np.mean(vals):.4f} +/- {np.std(vals):.4f}  (n={len(vals)})")
