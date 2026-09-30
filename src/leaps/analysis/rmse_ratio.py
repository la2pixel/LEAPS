"""RMSE ratio (Park et al. 2026, arXiv:2603.10474 convention) for joint kinematics.

Park et al.'s exact definition (verified via direct fetch of their methods
text, 2026-09-15): "the RMSE between simulated and experimental patterns,
normalized by the mean RMSE obtained from all cross-human comparisons within
the experimental dataset." Ratio ~1.0 => sim error is about as large as
normal human-to-human variability.

Different metric from match_fraction (Schumacher et al.'s "experimental
match" -- band-membership fraction, already computed in kinematic_match.py).
This one needs per-subject mean curves (not just pooled mean+-SD), so it
re-runs the human-data loading loop keeping subjects separate -- does not
touch or invalidate the cached human_gait_band.npz used elsewhere.

    python -m leaps.analysis.rmse_ratio [--h1622-seed0-only]

Sim side reuses kinematic_match.py's pool_sim() unchanged.
"""
from __future__ import annotations

import argparse
import glob
from itertools import combinations
from pathlib import Path

import numpy as np

from leaps.data import load_mat_table, find_stride_intervals, time_normalize_stride
from leaps.data.metadata import SUBJECTS
from leaps.scripts.biomech_fidelity import load_tstar_map, RUNS
from leaps.scripts.kinematic_match import (
    JOINTS, GON, SUBJ, METHODS_VARIANTS, pool_sim,
)
from leaps.data.metadata import CAMARGO_DATA_ROOT

SPEED, TOL = 1.35, 0.25


def per_subject_curves(speed: float, tol: float) -> dict[str, dict[str, np.ndarray]]:
    """subject -> joint -> mean angle curve (101,), kept separate (not pooled)."""
    out = {}
    for s in SUBJ:
        base = glob.glob(f"{CAMARGO_DATA_ROOT}/{s}/*/treadmill")[0]
        ang = {j: [] for j in JOINTS}
        for gon_f in sorted(glob.glob(f"{base}/gon/*.mat")):
            stem = Path(gon_f).name
            try:
                g = load_mat_table(gon_f)
                cnd = load_mat_table(f"{base}/conditions/{stem}")
                gc = load_mat_table(f"{base}/gcRight/{stem}")
            except Exception:
                continue
            iv = find_stride_intervals(gc["Header"], gc["HeelStrike"])
            for t0, t1 in iv:
                sp = cnd["Speed"][(cnd["Header"] >= t0) & (cnd["Header"] < t1)]
                if len(sp) == 0 or abs(float(np.mean(sp)) - speed) > tol:
                    continue
                m = (g["Header"] >= t0) & (g["Header"] < t1)
                if m.sum() < 20:
                    continue
                for j in JOINTS:
                    raw = g[GON[j]][m]
                    ang[j].append(time_normalize_stride(raw[:, None])[:, 0])
        if all(ang[j] for j in JOINTS):
            out[s] = {j: np.stack(ang[j]).mean(0) for j in JOINTS}
    return out


def cross_human_rmse(subj_curves: dict) -> dict[str, float]:
    """Per joint: mean RMSE over all subject pairs -- the human-to-human baseline."""
    subs = list(subj_curves)
    out = {}
    for j in JOINTS:
        vals = [np.sqrt(np.mean((subj_curves[a][j] - subj_curves[b][j]) ** 2))
                for a, b in combinations(subs, 2)]
        out[j] = float(np.mean(vals))
    return out


def human_mean(subj_curves: dict) -> dict[str, np.ndarray]:
    return {j: np.mean([subj_curves[s][j] for s in subj_curves], axis=0) for j in JOINTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1622-seed0-only", action="store_true",
                     help="use only seed0 (the walking checkpoint) for h1622's real-LAP conditions, "
                          "instead of pooling all seeds")
    args = ap.parse_args()

    print("loading per-subject human curves (Camargo, 6 subjects)...")
    subj_curves = per_subject_curves(SPEED, TOL)
    print(f"  usable subjects: {list(subj_curves)}")
    baseline = cross_human_rmse(subj_curves)
    hmean = human_mean(subj_curves)
    print(f"  cross-human RMSE baseline (deg): {baseline}\n")

    tmap = load_tstar_map()
    bodies = ["h0918", "h1622", "h2190"]
    print(f"{'body':<7}{'method':<24}" + "".join(f"{j+'_ratio':>12}" for j in JOINTS))
    for body in bodies:
        for cond, ml in METHODS_VARIANTS.items():
            if cond not in RUNS[body]:
                continue
            dirs = RUNS[body][cond]
            if args.h1622_seed0_only and body == "h1622" and cond.startswith("real-LAP"):
                dirs = [d for d in dirs if d.endswith("seed0")]
                if not dirs:
                    continue
            sd = pool_sim(dirs, tmap)
            if sd is None:
                print(f"{body:<7}{ml:<24}  no usable cycles")
                continue
            row = f"{body:<7}{ml:<24}"
            for j in JOINTS:
                # Same DC-offset alignment as match_fraction (kinematic_match.py's
                # ALIGN_OFFSET) -- the Camargo gon signal has a per-subject/per-
                # sim relative-goniometer zero, so raw RMSE would be dominated by
                # that arbitrary offset rather than shape/timing error. Park et
                # al.'s own text doesn't specify this explicitly; applying it here
                # for consistency with the rest of this project's kinematics
                # reporting, not because it's confirmed to be their exact choice.
                sim_mean = sd["ang"][j]
                sim_aligned = sim_mean - (sim_mean - hmean[j]).mean()
                rmse = float(np.sqrt(np.mean((sim_aligned - hmean[j]) ** 2)))
                ratio = rmse / baseline[j]
                row += f"{ratio:>12.2f}"
            print(row)


if __name__ == "__main__":
    main()
