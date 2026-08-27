"""Redo the k=1..11 reconstruction-elbow sweep (originally done under the
old loss recipe -- see project_k_dimension_decision memory) under the
corrected recipe from today's threshold sweep: depth=1, threshold=0.6,
scale=0.9. Output activation is untouched (still Sigmoid) -- only the
latent-penalty constants changed, so this checks whether the elbow location
itself is robust to that fix, not whether k=6 was ever wrong.

AB06 alone, same fixed held-out split used in the pooling sweep, 3 seeds/k.
"""

import os

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import torch

from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides
from leaps.synergy_common import train_ae

SUBJECT = "AB06"
DEPTH = 1
THRESHOLD = 0.6
SCALE = 0.9
VAL_FRACTION = 0.2
SEEDS = [0, 1, 2]
N_SUB = 50_000
K_GRID = list(range(1, 12))

strides = load_strides(LEAPS_H5_PATH, subjects=[SUBJECT], modes=ALL_MODES)
rng = np.random.default_rng(0)
idx = rng.permutation(len(strides))
n_val = max(1, int(len(strides) * VAL_FRACTION))
holdout = strides[idx[:n_val]]
train_pool = strides[idx[n_val:]]

stride_p99s = np.percentile(train_pool, 99, axis=1)
global_divisors = stride_p99s.mean(axis=0) + 1e-8


def to_unit(s):
    return np.clip(s / global_divisors, 0.0, 1.0).astype(np.float32)


X_train_full = to_unit(train_pool).reshape(-1, 11)
X_holdout = to_unit(holdout).reshape(-1, 11)

results = []
for k in K_GRID:
    for seed in SEEDS:
        rng_sub = np.random.default_rng(seed)
        n_sub = min(N_SUB, len(X_train_full))
        sub_idx = rng_sub.choice(len(X_train_full), size=n_sub, replace=False)
        X_train_sub = X_train_full[sub_idx]

        model = train_ae(
            X_train_sub, dim_latent=k, epochs=50, batch_size=512, seed=seed,
            depth=DEPTH, threshold=THRESHOLD, scale=SCALE,
        )
        model.eval()
        with torch.no_grad():
            holdout_hat, _ = model(torch.from_numpy(X_holdout))
        holdout_hat = holdout_hat.numpy()

        ss_res = ((X_holdout - holdout_hat) ** 2).sum()
        ss_tot = ((X_holdout - X_holdout.mean(axis=0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        results.append(dict(k=k, seed=seed, r2=r2))
        print(f"k={k:2d}  seed={seed}  AB06-holdout R2={r2:.4f}")

print("\n=== summary (mean +/- std over seeds) ===")
for k in K_GRID:
    vals = [r["r2"] for r in results if r["k"] == k]
    print(f"k={k:2d}  R2 = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

import json
with open("/home/nadinebadie/lalitha/LEAPS/results/backfill/k_sweep_corrected_recipe.json", "w") as f:
    json.dump(results, f, indent=2)
