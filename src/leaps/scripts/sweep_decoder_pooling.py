"""Stage 1 of the pooling comparison (AB06 alone / small pooled / all pooled /
leave-AB06-out). Locked AE recipe from the threshold sweep: depth=1,
threshold=0.6, scale=0.9 -- pooling strategy is the only free variable here.

All conditions are evaluated on the SAME fixed AB06 held-out strides (never
trained on by any condition), so R2 differences are attributable to what
data went into training, not to different eval sets. Each condition also
gets its own P99 normalization computed from its own training pool, matching
the existing production convention (get_synergies.py / train_single_subject_
decoder.py) -- this mirrors what would actually ship in that decoder's own
norm.npz at deployment, not an artificial shared scaling.

Every condition subsamples its training pool to the same 50k-frame cap, so
"more subjects" and "more total frames" don't get conflated -- AB06 alone
already exceeds 50k frames on its own, so every condition trains on the same
frame budget by construction.
"""

import os

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import torch

from leaps.data.metadata import ALL_MODES, ALL_SUBJECT_IDS, LEAPS_H5_PATH, SUBJECTS
from leaps.data.stride_dataset import load_strides
from leaps.synergy_common import train_ae

TARGET = "AB06"
K = 6
DEPTH = 1
THRESHOLD = 0.6
SCALE = 0.9
VAL_FRACTION = 0.2
SEEDS = [0, 1, 2]
N_SUB = 50_000

ab06_strides = load_strides(LEAPS_H5_PATH, subjects=[TARGET], modes=ALL_MODES)
rng = np.random.default_rng(0)
idx = rng.permutation(len(ab06_strides))
n_val = max(1, int(len(ab06_strides) * VAL_FRACTION))
ab06_holdout = ab06_strides[idx[:n_val]]
ab06_train_pool = ab06_strides[idx[n_val:]]
print(f"{TARGET}: {len(ab06_strides)} strides total -- {len(ab06_holdout)} held out (fixed across all "
      f"conditions below), {len(ab06_train_pool)} available for training")

# closest-morphology subjects to AB06, by (height, mass) z-distance -- same
# logic train_single_subject_decoder.py used to justify picking AB06 itself
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
    f"{TARGET}_alone":          [TARGET],
    "small_pooled_5":            [TARGET] + closest4,
    "all_pooled_22":              ALL_SUBJECT_IDS,
    f"leave_{TARGET}_out_21":    OTHER_SUBJECTS,
}


def to_unit(strides_3d, divisors):
    return np.clip(strides_3d / divisors, 0.0, 1.0).astype(np.float32)


results = []
for cond_name, subj_list in CONDITIONS.items():
    other_subjects = [s for s in subj_list if s != TARGET]
    other_strides = (
        load_strides(LEAPS_H5_PATH, subjects=other_subjects, modes=ALL_MODES)
        if other_subjects else np.empty((0,) + ab06_strides.shape[1:], dtype=ab06_strides.dtype)
    )
    if TARGET in subj_list:
        train_pool = np.concatenate([ab06_train_pool, other_strides], axis=0) if len(other_strides) else ab06_train_pool
    else:
        train_pool = other_strides

    stride_p99s = np.percentile(train_pool, 99, axis=1)
    global_divisors = stride_p99s.mean(axis=0) + 1e-8

    X_train_full = to_unit(train_pool, global_divisors).reshape(-1, 11)
    X_holdout = to_unit(ab06_holdout, global_divisors).reshape(-1, 11)

    for seed in SEEDS:
        rng_sub = np.random.default_rng(seed)
        n_sub = min(N_SUB, len(X_train_full))
        sub_idx = rng_sub.choice(len(X_train_full), size=n_sub, replace=False)
        X_train_sub = X_train_full[sub_idx]

        model = train_ae(
            X_train_sub, dim_latent=K, epochs=50, batch_size=512, seed=seed,
            depth=DEPTH, threshold=THRESHOLD, scale=SCALE,
        )
        model.eval()
        with torch.no_grad():
            holdout_hat, _ = model(torch.from_numpy(X_holdout))
        holdout_hat = holdout_hat.numpy()

        ss_res = ((X_holdout - holdout_hat) ** 2).sum()
        ss_tot = ((X_holdout - X_holdout.mean(axis=0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        results.append(dict(cond=cond_name, seed=seed, r2=r2, n_train_pool=len(train_pool), n_sub=n_sub))
        print(f"{cond_name:20s} seed={seed}  AB06-holdout R2={r2:.4f}  "
              f"(train_pool={len(train_pool)} strides -> subsampled to {n_sub} frames)")

print(f"\n=== summary (mean +/- std over seeds), all evaluated on the SAME fixed {TARGET} held-out strides ===")
for cond_name in CONDITIONS:
    vals = [r["r2"] for r in results if r["cond"] == cond_name]
    print(f"{cond_name:20s} R2 = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
