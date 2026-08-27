"""Full 2D grid instead of the earlier fixed-margin slice, to actually settle
two open questions rather than assume them:
  (1) does depth=2 help once the latent penalty is loosened? (earlier test
      only compared depth at the OLD tight penalty, 0.5/0.6)
  (2) was margin=0.3 actually the right choice, or just "wide enough"?
depth in {1,2} x threshold in {0.5..1.0} x margin in {0.1..0.5} x 3 seeds
= 180 fits, AB06 alone, same fixed held-out split as every other sweep today.
"""

import os

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import pandas as pd
import torch

from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides
from leaps.synergy_common import train_ae

SUBJECT = "AB06"
K = 6
VAL_FRACTION = 0.2
SEEDS = [0, 1, 2]
N_SUB = 50_000
DEPTHS = [1, 2]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MARGINS = [0.1, 0.2, 0.3, 0.4, 0.5]

strides = load_strides(LEAPS_H5_PATH, subjects=[SUBJECT], modes=ALL_MODES)
rng = np.random.default_rng(0)
idx = rng.permutation(len(strides))
n_val = max(1, int(len(strides) * VAL_FRACTION))
holdout = strides[idx[:n_val]]
train_pool = strides[idx[n_val:]]

p99 = np.percentile(train_pool, 99, axis=1).mean(axis=0) + 1e-8


def to_unit(s):
    return np.clip(s / p99, 0.0, 1.0).astype(np.float32)


X_train_full = to_unit(train_pool).reshape(-1, 11)
X_holdout = to_unit(holdout).reshape(-1, 11)

results = []
total = len(DEPTHS) * len(THRESHOLDS) * len(MARGINS) * len(SEEDS)
count = 0
for depth in DEPTHS:
    for threshold in THRESHOLDS:
        for margin in MARGINS:
            scale = threshold + margin
            for seed in SEEDS:
                count += 1
                rng_sub = np.random.default_rng(seed)
                n_sub = min(N_SUB, len(X_train_full))
                sub_idx = rng_sub.choice(len(X_train_full), size=n_sub, replace=False)
                X_train_sub = X_train_full[sub_idx]

                model = train_ae(
                    X_train_sub, dim_latent=K, epochs=50, batch_size=512, seed=seed,
                    depth=depth, threshold=threshold, scale=scale,
                )
                model.eval()
                with torch.no_grad():
                    holdout_hat, _ = model(torch.from_numpy(X_holdout))
                holdout_hat = holdout_hat.numpy()

                ss_res = ((X_holdout - holdout_hat) ** 2).sum()
                ss_tot = ((X_holdout - X_holdout.mean(axis=0)) ** 2).sum()
                r2 = 1 - ss_res / ss_tot

                results.append(dict(depth=depth, threshold=threshold, margin=margin, scale=scale, seed=seed, r2=r2))
                print(f"[{count}/{total}] depth={depth} threshold={threshold:.1f} margin={margin:.1f} "
                      f"scale={scale:.1f} seed={seed}  R2={r2:.4f}", flush=True)

df = pd.DataFrame(results)
df.to_csv("/home/nadinebadie/lalitha/LEAPS/results/backfill/depth_threshold_margin_grid.csv", index=False)

summary = df.groupby(["depth", "threshold", "margin"])["r2"].agg(["mean", "std"]).sort_values("mean", ascending=False)
print("\n=== top 15 combos by mean R2 ===")
print(summary.head(15))

print("\n=== depth=1 vs depth=2, marginalized over threshold/margin ===")
print(df.groupby("depth")["r2"].agg(["mean", "std"]))

print("\n=== best margin per threshold (depth=1) ===")
d1 = df[df["depth"] == 1]
best_margin_per_threshold = d1.groupby(["threshold", "margin"])["r2"].mean().reset_index()
print(best_margin_per_threshold.loc[best_margin_per_threshold.groupby("threshold")["r2"].idxmax()])

best_row = summary.reset_index().iloc[0]
print(f"\n=== overall best: depth={int(best_row['depth'])} threshold={best_row['threshold']:.1f} "
      f"margin={best_row['margin']:.1f}  R2={best_row['mean']:.4f} +/- {best_row['std']:.4f} ===")
