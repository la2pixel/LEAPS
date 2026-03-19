"""Per-subject data quality analysis.

Checks:
  1. Stride counts per subject and mode
  2. Normalization bounds quality (min_vals/max_vals per subject)
  3. Per-subject reconstruction error using trained StrideFlatAE

Usage:
    python3 scripts/analyze_subjects.py \
        --data /fast/lsivakumar/data/processed/emg_activations.h5 \
        --checkpoint experiments/stride_sweep_v2/checkpoints/StrideFlatAE_d6.pkl \
        --output results/subject_analysis/
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


EMG_CHANNELS = [
    "gastrocmed", "tibialisanterior", "soleus",
    "vastusmedialis", "vastuslateralis", "rectusfemoris",
    "bicepsfemoris", "semitendinosus", "gracilis",
    "gluteusmedius", "rightexternaloblique",
]


def load_per_subject(path: str):
    """Load strides, modes, and normalization bounds per subject from HDF5."""
    subjects = {}
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            if not key.startswith("AB"):
                continue
            grp = f[key]
            subjects[key] = {
                "strides": grp["strides"][:],           # (n, 101, 11)
                "modes":   grp["modes"][:].astype(str), # (n,)
                "min_vals": grp["min_vals"][:],          # (11,)
                "max_vals": grp["max_vals"][:],          # (11,)
            }
    return subjects


def stride_stats(strides: np.ndarray):
    """Return mean activation, std, and per-channel range of a stride array."""
    flat = strides.reshape(-1, strides.shape[-1])  # (n*101, 11)
    return {
        "mean": flat.mean(axis=0),
        "std":  flat.std(axis=0),
        "min":  flat.min(axis=0),
        "max":  flat.max(axis=0),
    }


def compute_per_subject_r2(subjects: dict, checkpoint: str | None, latent_dim: int = 6):
    """Load StrideFlatAE and compute R² per subject. Returns None if no checkpoint."""
    if checkpoint is None or not Path(checkpoint).exists():
        return None

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from leaps.training.stride_trainer import build_stride_model

    model = build_stride_model("StrideFlatAE", latent_dim)
    model.load(checkpoint)

    r2_per_subject = {}
    for subj, data in subjects.items():
        strides = data["strides"].astype(np.float32)  # (n, 101, 11)
        recon = model.predict(strides)                 # (n, 101, 11)

        ss_res = ((strides - recon) ** 2).sum()
        ss_tot = ((strides - strides.mean()) ** 2).sum()
        r2_per_subject[subj] = float(1 - ss_res / ss_tot)

    return r2_per_subject


def print_table(subjects: dict, r2: dict | None):
    modes_all = ["treadmill", "levelground", "ramp", "stair"]

    header = f"{'Subject':<10} {'Total':>7}"
    for m in modes_all:
        header += f" {m[:6]:>8}"
    header += f" {'norm_range_mean':>16}"
    if r2:
        header += f" {'R2':>7}"
    print(header)
    print("-" * len(header))

    total_strides = 0
    for subj in sorted(subjects.keys()):
        d = subjects[subj]
        n_total = len(d["strides"])
        total_strides += n_total

        mode_counts = {m: int((d["modes"] == m).sum()) for m in modes_all}
        norm_range = float((d["max_vals"] - d["min_vals"]).mean())

        row = f"{subj:<10} {n_total:>7}"
        for m in modes_all:
            row += f" {mode_counts[m]:>8}"
        row += f" {norm_range:>16.4f}"
        if r2:
            row += f" {r2.get(subj, float('nan')):>7.4f}"
        print(row)

    print("-" * len(header))
    print(f"{'TOTAL':<10} {total_strides:>7}")


def plot_norm_bounds(subjects: dict, output_dir: Path):
    """Heatmap of per-subject normalization ranges (max-min per channel)."""
    subj_ids = sorted(subjects.keys())
    ranges = np.array([
        subjects[s]["max_vals"] - subjects[s]["min_vals"]
        for s in subj_ids
    ])  # (n_subjects, 11)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(ranges.T, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(subj_ids)))
    ax.set_xticklabels(subj_ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(11))
    ax.set_yticklabels(EMG_CHANNELS, fontsize=8)
    ax.set_title("Normalization range (max-min) per subject per channel\n"
                 "Narrow range = few reference strides = unreliable normalization")
    fig.colorbar(im, ax=ax, label="max - min")
    plt.tight_layout()
    path = output_dir / "norm_ranges.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_stride_counts(subjects: dict, output_dir: Path):
    """Stacked bar of stride counts per subject by mode."""
    subj_ids = sorted(subjects.keys())
    modes = ["treadmill", "levelground", "ramp", "stair"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    counts = {m: [] for m in modes}
    for s in subj_ids:
        for m in modes:
            counts[m].append(int((subjects[s]["modes"] == m).sum()))

    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(len(subj_ids))
    for m, c in zip(modes, colors):
        vals = np.array(counts[m])
        ax.bar(subj_ids, vals, bottom=bottom, label=m, color=c, alpha=0.85)
        bottom += vals

    ax.set_xlabel("Subject")
    ax.set_ylabel("Stride count")
    ax.set_title("Stride counts per subject by mode")
    ax.legend()
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    path = output_dir / "stride_counts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_r2_per_subject(r2: dict, output_dir: Path):
    subj_ids = sorted(r2.keys())
    vals = [r2[s] for s in subj_ids]
    mean_r2 = np.mean(vals)

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["#e53935" if v < mean_r2 - 0.05 else "#43a047" for v in vals]
    ax.bar(subj_ids, vals, color=colors)
    ax.axhline(mean_r2, color="black", linestyle="--", linewidth=1, label=f"mean={mean_r2:.3f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Subject")
    ax.set_ylabel("R²")
    ax.set_title("Per-subject reconstruction R² (StrideFlatAE)\nRed = >5% below mean")
    ax.legend()
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    path = output_dir / "r2_per_subject.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="StrideFlatAE checkpoint for R² computation")
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--output", default="results/subject_analysis")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data}")
    subjects = load_per_subject(args.data)
    print(f"Found {len(subjects)} subjects: {sorted(subjects.keys())}\n")

    r2 = compute_per_subject_r2(subjects, args.checkpoint, args.latent_dim)
    if r2 is None:
        print("No checkpoint provided — skipping R² computation\n")

    print_table(subjects, r2)

    plot_stride_counts(subjects, output_dir)
    plot_norm_bounds(subjects, output_dir)
    if r2:
        plot_r2_per_subject(r2, output_dir)

    # Flag outliers
    print("\n=== OUTLIER FLAGS ===")
    treadmill_counts = {s: int((subjects[s]["modes"] == "treadmill").sum())
                        for s in subjects}
    mean_tm = np.mean(list(treadmill_counts.values()))
    for s, n in sorted(treadmill_counts.items()):
        if n < mean_tm * 0.5:
            print(f"  LOW TREADMILL STRIDES: {s} has {n} "
                  f"(mean={mean_tm:.0f}) — normalization may be unreliable")

    if r2:
        mean_r2 = np.mean(list(r2.values()))
        for s, v in sorted(r2.items()):
            if v < mean_r2 - 0.05:
                print(f"  LOW R²: {s} = {v:.4f} (mean={mean_r2:.4f})")

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
