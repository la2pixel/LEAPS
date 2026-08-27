"""Extract a per-subject EMG coordination "fingerprint" for every Camargo subject.

Step 1 of the subject-conditioned personalization framework (see RUNS_INDEX.md
/ memory session 2026-08-03): instead of training a whole separate decoder +
RL policy per person (what AB06/AB20 did), the plan is one shared decoder +
one shared RL policy, conditioned on a cheap per-subject signal computed
directly from that subject's own raw EMG -- no training required for the
fingerprint itself.

Fingerprint = the upper triangle of that subject's own 11x11 cross-channel
EMG correlation matrix (55 values) -- the exact same coordination-structure
metric this project's own co-activation-correlation pipeline already uses
and has already validated as tracking something real (see the muscle
co-activation section of RUNS_INDEX.md). Not a new representation invented
for this -- reused deliberately.

Pearson correlation is invariant to each channel's own per-subject scale
(mean/variance), so -- unlike decoder training -- no P99 normalization step
is needed here: correlation already standardizes each channel internally.
"""

from __future__ import annotations

import os

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

from pathlib import Path

import numpy as np

from leaps.data.metadata import ALL_MODES, EMG_CHANNELS, LEAPS_H5_PATH, SUBJECTS
from leaps.data.stride_dataset import load_strides

OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "subject_fingerprints"


def upper_tri_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def fingerprint_for_subject(subject: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (fingerprint_vector[55], full_corr_matrix[11,11], n_frames_used)."""
    strides = load_strides(LEAPS_H5_PATH, subjects=[subject], modes=ALL_MODES)
    frames = strides.reshape(-1, strides.shape[-1])
    corr = np.corrcoef(frames, rowvar=False)
    iu, ju = upper_tri_indices(corr.shape[0])
    fingerprint = corr[iu, ju].astype(np.float32)
    return fingerprint, corr.astype(np.float32), frames.shape[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = sorted(SUBJECTS.keys())
    print(f"Extracting fingerprints for {len(subjects)} subjects from {LEAPS_H5_PATH}")

    fingerprints: dict[str, np.ndarray] = {}
    corr_matrices: dict[str, np.ndarray] = {}
    n_frames: dict[str, int] = {}

    for subj in subjects:
        fp, corr, n = fingerprint_for_subject(subj)
        fingerprints[subj] = fp
        corr_matrices[subj] = corr
        n_frames[subj] = n
        print(f"  {subj}: {n:,} frames -> fingerprint dim {fp.shape[0]}")

    iu, ju = upper_tri_indices(len(EMG_CHANNELS))
    pair_names = [f"{EMG_CHANNELS[i]}~{EMG_CHANNELS[j]}" for i, j in zip(iu, ju)]

    np.savez(
        OUT_DIR / "fingerprints.npz",
        subjects=np.array(subjects),
        pair_names=np.array(pair_names),
        channels=np.array(EMG_CHANNELS),
        **{f"fp_{s}": fingerprints[s] for s in subjects},
        **{f"corr_{s}": corr_matrices[s] for s in subjects},
        **{f"n_{s}": n_frames[s] for s in subjects},
    )
    print(f"\nSaved to {OUT_DIR / 'fingerprints.npz'}")

    # --- sanity check before committing any RL compute: does this actually
    # differentiate subjects, or does everyone look the same? ---
    print("\n=== Sanity check: pairwise cosine similarity of fingerprints ===")
    mat = np.stack([fingerprints[s] for s in subjects])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    cos_sim = (mat @ mat.T) / (norms @ norms.T + 1e-8)

    for i, s in enumerate(subjects):
        row = cos_sim[i].copy()
        row[i] = -np.inf  # exclude self
        j_max = int(np.argmax(row))
        j_min = int(np.argmin(cos_sim[i]))
        print(
            f"  {s}: most similar -> {subjects[j_max]} ({row[j_max]:.3f}), "
            f"least similar -> {subjects[j_min]} ({cos_sim[i][j_min]:.3f})"
        )

    if "AB06" in subjects and "AB20" in subjects:
        i, j = subjects.index("AB06"), subjects.index("AB20")
        print(f"\nAB06 <-> AB20 cosine similarity: {cos_sim[i, j]:.3f}")
        print(f"AB06 <-> pooled-average similarity: {np.mean(np.delete(cos_sim[i], i)):.3f}")

    off_diag = cos_sim[~np.eye(len(subjects), dtype=bool)]
    print(f"\nAll-pairs off-diagonal similarity: mean={off_diag.mean():.3f}, std={off_diag.std():.3f}, "
          f"min={off_diag.min():.3f}, max={off_diag.max():.3f}")


if __name__ == "__main__":
    main()
