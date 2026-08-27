"""Repeat today's two AB06-only conclusions on a second subject (AB08, the
next-closest morphology match to h0918 after AB06) to check they aren't
AB06-specific artifacts -- mirrors the rigor bar the notebook's existing
stride-count section already applied (it checked AB06/AB08/AB19 for that
question; today's new sweeps didn't, until now):
  (1) pooling comparison: AB08 alone vs small-pooled vs all-pooled vs leave-AB08-out
  (2) k=1..11 elbow, locked recipe (depth=1, threshold=0.6, scale=0.9)
"""

import os

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import torch

from leaps.data.metadata import ALL_MODES, ALL_SUBJECT_IDS, LEAPS_H5_PATH, SUBJECTS
from leaps.data.stride_dataset import load_strides
from leaps.synergy_common import train_ae

TARGET = "AB08"
K = 6
DEPTH = 1
THRESHOLD = 0.6
SCALE = 0.9
VAL_FRACTION = 0.2
SEEDS = [0, 1, 2]
N_SUB = 50_000

target_strides = load_strides(LEAPS_H5_PATH, subjects=[TARGET], modes=ALL_MODES)
rng = np.random.default_rng(0)
idx = rng.permutation(len(target_strides))
n_val = max(1, int(len(target_strides) * VAL_FRACTION))
holdout = target_strides[idx[:n_val]]
target_train_pool = target_strides[idx[n_val:]]
print(f"{TARGET}: {len(target_strides)} strides total -- {len(holdout)} held out, {len(target_train_pool)} for training")


def to_unit(s, divisors):
    return np.clip(s / divisors, 0.0, 1.0).astype(np.float32)


def fit_eval(train_pool, seed, k=K):
    p99 = np.percentile(train_pool, 99, axis=1).mean(axis=0) + 1e-8
    X_train_full = to_unit(train_pool, p99).reshape(-1, 11)
    X_holdout = to_unit(holdout, p99).reshape(-1, 11)
    rng_sub = np.random.default_rng(seed)
    n_sub = min(N_SUB, len(X_train_full))
    sub_idx = rng_sub.choice(len(X_train_full), size=n_sub, replace=False)
    model = train_ae(X_train_full[sub_idx], dim_latent=k, epochs=50, batch_size=512, seed=seed,
                      depth=DEPTH, threshold=THRESHOLD, scale=SCALE)
    model.eval()
    with torch.no_grad():
        holdout_hat, _ = model(torch.from_numpy(X_holdout))
    holdout_hat = holdout_hat.numpy()
    ss_res = ((X_holdout - holdout_hat) ** 2).sum()
    ss_tot = ((X_holdout - X_holdout.mean(axis=0)) ** 2).sum()
    return 1 - ss_res / ss_tot


# ---- (1) pooling comparison ----
heights = np.array([SUBJECTS[s].height for s in ALL_SUBJECT_IDS])
masses = np.array([SUBJECTS[s].mass for s in ALL_SUBJECT_IDS])
h_mean, h_std, m_mean, m_std = heights.mean(), heights.std(), masses.mean(), masses.std()
target_z = np.array([(SUBJECTS[TARGET].height - h_mean) / h_std, (SUBJECTS[TARGET].mass - m_mean) / m_std])
dists = sorted(
    (np.linalg.norm(np.array([(SUBJECTS[s].height - h_mean) / h_std, (SUBJECTS[s].mass - m_mean) / m_std]) - target_z), s)
    for s in ALL_SUBJECT_IDS if s != TARGET
)
closest4 = [s for _, s in dists[:4]]
print(f"closest-morphology to {TARGET}: {closest4}")

OTHER_SUBJECTS = [s for s in ALL_SUBJECT_IDS if s != TARGET]
CONDITIONS = {
    f"{TARGET}_alone":        [TARGET],
    "small_pooled_5":          [TARGET] + closest4,
    "all_pooled_22":            ALL_SUBJECT_IDS,
    f"leave_{TARGET}_out_21":  OTHER_SUBJECTS,
}

pooling_results = []
for cond_name, subj_list in CONDITIONS.items():
    other_subjects = [s for s in subj_list if s != TARGET]
    other_strides = (
        load_strides(LEAPS_H5_PATH, subjects=other_subjects, modes=ALL_MODES)
        if other_subjects else np.empty((0,) + target_strides.shape[1:], dtype=target_strides.dtype)
    )
    if TARGET in subj_list:
        train_pool = np.concatenate([target_train_pool, other_strides], axis=0) if len(other_strides) else target_train_pool
    else:
        train_pool = other_strides

    for seed in SEEDS:
        r2 = fit_eval(train_pool, seed)
        pooling_results.append((cond_name, seed, r2, len(train_pool)))
        print(f"[pooling] {cond_name:20s} seed={seed}  R2={r2:.4f}  (train_pool={len(train_pool)} strides)")

print(f"\n=== pooling summary ({TARGET}) ===")
for cond_name in CONDITIONS:
    vals = [r[2] for r in pooling_results if r[0] == cond_name]
    print(f"{cond_name:20s} R2 = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# ---- (2) k elbow ----
k_results = []
for k in range(1, 12):
    for seed in SEEDS:
        r2 = fit_eval(target_train_pool, seed, k=k)
        k_results.append((k, seed, r2))
        print(f"[k-sweep] k={k:2d}  seed={seed}  R2={r2:.4f}")

print(f"\n=== k-elbow summary ({TARGET}) ===")
for k in range(1, 12):
    vals = [r[2] for r in k_results if r[0] == k]
    print(f"k={k:2d}  R2 = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

import json
with open("/home/nadinebadie/lalitha/LEAPS/results/backfill/cross_subject_ab08_check.json", "w") as f:
    json.dump(dict(pooling=pooling_results, k_sweep=k_results), f, indent=2)
