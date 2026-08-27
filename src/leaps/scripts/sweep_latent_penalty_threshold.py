"""Find our own latent-penalty threshold via a sweep on AB06 data, instead of
adopting the reference repo's 0.8/1.2 by authority. Fixes the ramp margin
(scale - threshold) at 0.3 -- wide enough for a smooth non-vanishing
gradient (both 0.1 and 0.4 margins already worked in the prior probe), and
sweeps threshold alone so this is a one-parameter search, not a two-
parameter grid. 3 seeds per point.
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
MARGIN = 0.3
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [0, 1, 2]

strides = load_strides(LEAPS_H5_PATH, subjects=[SUBJECT], modes=ALL_MODES)
print(f"subject {SUBJECT}: {len(strides)} strides")

results = []
for threshold in THRESHOLDS:
    scale = threshold + MARGIN
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
            depth=1, threshold=threshold, scale=scale,
        )
        model.eval()

        with torch.no_grad():
            val_hat, z_val = model(torch.from_numpy(X_val))
        val_hat, z_val = val_hat.numpy(), z_val.numpy()

        ss_res = ((X_val - val_hat) ** 2).sum()
        ss_tot = ((X_val - X_val.mean(axis=0)) ** 2).sum()
        r2_val = 1 - ss_res / ss_tot

        results.append(dict(threshold=threshold, scale=scale, seed=split_seed,
                             r2_val=r2_val, z_absmax=np.abs(z_val).max()))
        print(f"threshold={threshold:.1f} scale={scale:.1f} seed={split_seed}  "
              f"val_R2={r2_val:.4f}  z_absmax={np.abs(z_val).max():.3f}")

print("\n=== summary (mean +/- std over seeds) ===")
for threshold in THRESHOLDS:
    vals = [r["r2_val"] for r in results if r["threshold"] == threshold]
    print(f"threshold={threshold:.1f}  val_R2 = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
