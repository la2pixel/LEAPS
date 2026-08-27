"""Train the subject-fingerprint-conditioned decoder -- step 2 of the
personalization framework (step 1: extract_subject_fingerprints.py).

One shared decoder, trained across a pool of subjects, each frame tagged
with its own subject's fingerprint -- instead of one decoder per subject
(what AB06/AB20 did). Uses SynergyAE's new subject_aware=True path
(synergy_common.py) with the fingerprints already extracted and sanity-
checked in step 1.

AB06 and AB20 are deliberately excluded from the training pool -- they're
the zero-shot test subjects: both already have a known, expensive
single-subject ceiling (fully retrained decoder + dedicated RL policy,
R2=0.98/0.9701, RUNS_INDEX.md) to compare this cheaper conditioned
approach against.

**Revision (2026-08-03): the first attempt's zero-shot check failed
outright** -- both held-out subjects reconstructed *better* with the
wrong (pooled-average) fingerprint than with their own. Root cause: the
decoder's hidden layer is only 22 units wide (2*dim_latent, this
architecture's own convention) but the raw fingerprint is 55-dim -- with
only 20 distinct fingerprints ever seen in training, a network that
narrow had every incentive to memorize those 20 points rather than learn
a mapping that generalizes to a nearby-but-unseen one. Two fixes applied
together: (1) PCA-reduce the fingerprint to k=10 (91.5% of variance
across the 20 training subjects, a real computed cutoff, not a guess),
fit ONLY on the training pool so the held-out subjects' projections are
genuinely out-of-sample; (2) inject Gaussian noise into the fingerprint
during training so the network can't just memorize the exact training
points.
"""

from __future__ import annotations

import os

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

from pathlib import Path

import numpy as np
import torch

from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH, SUBJECTS
from leaps.data.stride_dataset import load_strides
from leaps.synergy_common import CACHE_DIR, train_ae

FP_PATH = Path(__file__).resolve().parents[3] / "results" / "subject_fingerprints" / "fingerprints.npz"
OUT_DIR = CACHE_DIR.parent / "subject_conditioned" / "decoder_k11_pooled20"

HELD_OUT_SUBJECTS = {"AB06", "AB20"}
K = 11
VAL_FRACTION = 0.2
SEED = 0
# Per-subject frame cap after the train/val split, matching the established
# train_single_subject_decoder.py precedent (which subsamples ONE subject's
# ~180-220k frames down to 50k for speed). With 20 subjects instead of 1,
# capping per-subject keeps every subject represented in the pool rather
# than subsampling the pooled total (which would silently drown out
# whichever subjects happen to have fewer strides). First attempt trained
# on all ~3.85M pooled frames uncapped and didn't finish inside 20 minutes
# on this machine's single-threaded CPU training path
# (torch.set_num_threads(1) in synergy_common.py) -- this cap is the fix.
TRAIN_FRAMES_PER_SUBJECT = 15_000
VAL_FRAMES_PER_SUBJECT = 3_000
PCA_K = 10  # 91.5% explained variance across the 20 training-pool fingerprints, computed directly
NOISE_FRACTION = 0.25  # subject_noise_std as a fraction of each PCA dim's own training-pool std


def fit_pca(X: np.ndarray, k: int):
    """Fit PCA on X (n_samples, n_features), fit-only -- no sklearn dependency
    needed for something this small. Returns (mean, components[k, n_features])."""
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mean.astype(np.float32), Vt[:k].astype(np.float32)


def apply_pca(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (X - mean) @ components.T


def build_pool(subjects: list[str], fingerprints: dict[str, np.ndarray], rng: np.random.Generator):
    """Per-subject stride-level 80/20 split, pooled across subjects, each
    frame tagged with its own subject's fingerprint (broadcast across all
    101 gait-cycle points and all strides for that subject). Frames are
    subsampled per subject after the split -- see TRAIN_FRAMES_PER_SUBJECT."""
    X_train_parts, X_val_parts = [], []
    subj_train_parts, subj_val_parts = [], []

    for subj in subjects:
        strides = load_strides(LEAPS_H5_PATH, subjects=[subj], modes=ALL_MODES)  # (n_strides, 101, 11)
        n_strides = strides.shape[0]
        idx = rng.permutation(n_strides)
        n_val = max(1, int(n_strides * VAL_FRACTION))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        train_frames = strides[train_idx].reshape(-1, strides.shape[-1])
        val_frames = strides[val_idx].reshape(-1, strides.shape[-1])

        if train_frames.shape[0] > TRAIN_FRAMES_PER_SUBJECT:
            sub_idx = rng.choice(train_frames.shape[0], size=TRAIN_FRAMES_PER_SUBJECT, replace=False)
            train_frames = train_frames[sub_idx]
        if val_frames.shape[0] > VAL_FRAMES_PER_SUBJECT:
            sub_idx = rng.choice(val_frames.shape[0], size=VAL_FRAMES_PER_SUBJECT, replace=False)
            val_frames = val_frames[sub_idx]

        fp = fingerprints[subj].astype(np.float32)
        X_train_parts.append(train_frames)
        X_val_parts.append(val_frames)
        subj_train_parts.append(np.tile(fp, (train_frames.shape[0], 1)))
        subj_val_parts.append(np.tile(fp, (val_frames.shape[0], 1)))

        print(f"  {subj}: {n_strides} strides -> {train_frames.shape[0]:,} train frames, {val_frames.shape[0]:,} val frames", flush=True)

    X_train = np.concatenate(X_train_parts, axis=0)
    X_val = np.concatenate(X_val_parts, axis=0)
    subj_train = np.concatenate(subj_train_parts, axis=0)
    subj_val = np.concatenate(subj_val_parts, axis=0)
    return X_train, X_val, subj_train, subj_val


def r2_score(x, x_hat):
    ss_res = ((x - x_hat) ** 2).sum()
    ss_tot = ((x - x.mean(axis=0)) ** 2).sum()
    return 1 - ss_res / ss_tot


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp_data = np.load(FP_PATH)
    all_subjects = [s for s in fp_data["subjects"]]
    fingerprints = {s: fp_data[f"fp_{s}"] for s in all_subjects}

    training_pool = sorted(set(all_subjects) - HELD_OUT_SUBJECTS)
    print(f"Training pool: {len(training_pool)} subjects (held out: {sorted(HELD_OUT_SUBJECTS)})")

    # PCA fit ONLY on the training pool's fingerprints -- held-out subjects'
    # projections must be genuinely out-of-sample, not leak into the basis.
    fp_matrix = np.stack([fingerprints[s] for s in training_pool])
    pca_mean, pca_components = fit_pca(fp_matrix, PCA_K)
    fp_reduced = {s: apply_pca(fingerprints[s][None], pca_mean, pca_components)[0].astype(np.float32) for s in all_subjects}
    reduced_std = apply_pca(fp_matrix, pca_mean, pca_components).std(axis=0)
    noise_std = float(np.mean(reduced_std) * NOISE_FRACTION)
    print(f"PCA: {PCA_K} dims, per-dim std (training pool) = {np.round(reduced_std, 3)}, "
          f"subject_noise_std = {noise_std:.4f}")

    rng = np.random.default_rng(SEED)
    X_train_raw, X_val_raw, subj_train, subj_val = build_pool(training_pool, fp_reduced, rng)
    print(f"\nPooled: {X_train_raw.shape[0]:,} train frames, {X_val_raw.shape[0]:,} val frames")

    # per-stride P99 normalization, same convention as get_synergies.py /
    # train_single_subject_decoder.py -- fit on the pooled training frames.
    p99 = np.percentile(X_train_raw, 99, axis=0) + 1e-8
    X_train = np.clip(X_train_raw / p99, 0.0, 1.0).astype(np.float32)
    X_val = np.clip(X_val_raw / p99, 0.0, 1.0).astype(np.float32)

    print(f"\nTraining subject-conditioned decoder, k={K}, subject_dim={subj_train.shape[1]}, "
          f"subject_noise_std={noise_std:.4f}...")
    model = train_ae(
        X_train, dim_latent=K, subject_train=subj_train, subject_noise_std=noise_std,
        epochs=50, batch_size=512, seed=SEED,
    )
    model.eval()
    for p in model.decoder.parameters():
        p.requires_grad = False

    with torch.no_grad():
        val_hat, z_train_full = model(torch.from_numpy(X_train), subject=torch.from_numpy(subj_train))
        val_hat_val, _ = model(torch.from_numpy(X_val), subject=torch.from_numpy(subj_val))
    held_out_r2 = r2_score(X_val, val_hat_val.numpy())
    print(f"\nPooled held-out R2 (20 training subjects, held-out frames): {held_out_r2:.4f}")

    torch.save(model.decoder.state_dict(), OUT_DIR / "decoder.pt")
    np.savez(
        OUT_DIR / "norm.npz",
        p01=np.zeros(11, dtype=np.float32), p99=p99.astype(np.float32),
        z_lo=z_train_full.numpy().min(axis=0).astype(np.float32),
        z_hi=z_train_full.numpy().max(axis=0).astype(np.float32),
    )
    np.savez(OUT_DIR / "subject_pca.npz", mean=pca_mean, components=pca_components, k=PCA_K)
    print(f"Saved conditioned decoder + PCA transform to {OUT_DIR}")

    # --- zero-shot check on the held-out subjects, before any RL compute ---
    print("\n=== Zero-shot check: held-out subjects, fingerprint never seen during training ===")
    pooled_avg_fp_reduced = np.mean([fp_reduced[s] for s in training_pool], axis=0).astype(np.float32)

    for subj in sorted(HELD_OUT_SUBJECTS):
        strides = load_strides(LEAPS_H5_PATH, subjects=[subj], modes=ALL_MODES)
        frames = strides.reshape(-1, strides.shape[-1])
        X = np.clip(frames / p99, 0.0, 1.0).astype(np.float32)
        fp = fp_reduced[subj]
        fp_tiled = np.tile(fp, (X.shape[0], 1))
        avg_tiled = np.tile(pooled_avg_fp_reduced, (X.shape[0], 1))

        with torch.no_grad():
            hat_own_fp, _ = model(torch.from_numpy(X), subject=torch.from_numpy(fp_tiled))
            hat_avg_fp, _ = model(torch.from_numpy(X), subject=torch.from_numpy(avg_tiled))

        r2_own = r2_score(X, hat_own_fp.numpy())
        r2_avg = r2_score(X, hat_avg_fp.numpy())
        print(f"  {subj}: R2 with own fingerprint = {r2_own:.4f}  |  R2 with pooled-average fingerprint = {r2_avg:.4f}"
              f"  |  delta = {r2_own - r2_avg:+.4f}")


if __name__ == "__main__":
    main()
