"""Visualize HausdorferAE reconstruction quality on held-out val strides.

Produces three plots:
1. Gait profiles — original vs reconstructed activation curves (18 muscles × N strides)
2. Per-muscle R² bar chart — reveals which muscles reconstruct poorly
3. Latent histograms — z distribution per dim, with ±0.8 Lnorm dead-zone marked

Run locally (Windows) after copying checkpoint from cluster, or on cluster with Agg backend.

Usage:
    python -m leaps.scripts.visualize_snapshot_ae \\
        --checkpoint experiments/hausdorfer_ae/hausdorfer_ae.pt \\
        --data $LEAPS_EMG_H5 \\
        --output-dir experiments/hausdorfer_ae/plots
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from leaps.data.stride_dataset import load_strides, stride_train_val_split
from leaps.envs.emg_mapping import MODEL_ACTUATORS
from leaps.models.legacy_models import HausdorferAE
from leaps.scripts.train_snapshot import strides_to_snapshots


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to hausdorfer_ae.pt")
    p.add_argument("--data", required=True, help="Path to emg_activations_v2.h5")
    p.add_argument("--output-dir", default=None,
                   help="Where to save plots (default: checkpoint dir / plots)")
    p.add_argument("--n-strides", type=int, default=6,
                   help="Number of sample strides to show in gait profile plot")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ── helpers ───────────────────────────────────────────────────────────────────


def _r2_per_muscle(orig: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """R² per muscle. orig/recon: (N, 18)."""
    ss_res = ((orig - recon) ** 2).sum(axis=0)
    ss_tot = ((orig - orig.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-8)


# ── Plot 1: gait profiles ─────────────────────────────────────────────────────


def plot_gait_profiles(
    strides_orig: np.ndarray,   # (N, 101, 18) original muscle activations
    strides_recon: np.ndarray,  # (N, 101, 18) reconstructed
    sample_idx: np.ndarray,     # which strides to show
    output_dir: Path,
) -> None:
    """Rows = 18 muscles, cols = sample strides. Blue = original, red dashed = reconstructed."""
    n_muscles = 18
    n_samples = len(sample_idx)
    gait_pct = np.linspace(0, 100, 101)

    fig, axes = plt.subplots(
        n_muscles, n_samples,
        figsize=(2.8 * n_samples, 1.6 * n_muscles),
        sharex=True, sharey=True,
    )
    if n_samples == 1:
        axes = axes[:, np.newaxis]

    for col, si in enumerate(sample_idx):
        for row in range(n_muscles):
            ax = axes[row, col]
            ax.plot(gait_pct, strides_orig[si, :, row], "b-", linewidth=1.0, alpha=0.85)
            ax.plot(gait_pct, strides_recon[si, :, row], "r--", linewidth=1.0, alpha=0.85)
            ax.set_ylim(-0.05, 1.15)
            ax.tick_params(labelsize=5)
            if col == 0:
                ax.set_ylabel(MODEL_ACTUATORS[row], fontsize=6.5, rotation=0,
                               labelpad=60, ha="right", va="center")
            if row == 0:
                ax.set_title(f"stride {si}", fontsize=8)
            if row == n_muscles - 1:
                ax.set_xlabel("% gait", fontsize=7)

    # legend in top-right corner of last subplot
    axes[0, -1].plot([], [], "b-", label="original")
    axes[0, -1].plot([], [], "r--", label="reconstructed")
    axes[0, -1].legend(fontsize=7, loc="upper right")

    fig.suptitle("HausdorferAE — Gait Profile Reconstructions", fontsize=12, y=1.001)
    fig.tight_layout()
    path = output_dir / "gait_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 2: per-muscle R² ─────────────────────────────────────────────────────


def plot_per_muscle_r2(
    snapshots_orig: np.ndarray,   # (N_snapshots, 18)
    snapshots_recon: np.ndarray,
    output_dir: Path,
) -> None:
    r2 = _r2_per_muscle(snapshots_orig, snapshots_recon)
    order = np.argsort(r2)  # worst → best

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#d32f2f" if v < 0.8 else "#388e3c" for v in r2[order]]
    bars = ax.barh(np.arange(18), r2[order], color=colors, alpha=0.85)
    ax.set_yticks(np.arange(18))
    ax.set_yticklabels([MODEL_ACTUATORS[i] for i in order], fontsize=8)
    ax.axvline(0.8, color="gray", linestyle=":", linewidth=1.2, label="R²=0.80")
    ax.axvline(0.9, color="steelblue", linestyle=":", linewidth=1.2, label="R²=0.90")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("R²", fontsize=10)
    ax.set_title("HausdorferAE — Per-Muscle Reconstruction Quality", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    # annotate values
    for i, (bar, v) in enumerate(zip(bars, r2[order])):
        ax.text(min(v + 0.01, 1.01), bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    path = output_dir / "per_muscle_r2.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 3: latent histograms ─────────────────────────────────────────────────


def plot_latent_histograms(
    z_val: np.ndarray,   # (N_snapshots, latent_dim)
    output_dir: Path,
) -> None:
    """Histogram of z values per latent dim. Shows whether Lnorm keeps z inside ±0.8."""
    latent_dim = z_val.shape[1]
    ncols = 3
    nrows = (latent_dim + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for i in range(latent_dim):
        ax = axes[i]
        z_dim = z_val[:, i]
        ax.hist(z_dim, bins=80, color="steelblue", alpha=0.75, density=True)
        ax.axvline(-0.8, color="orange", linewidth=1.2, linestyle="--", label="±0.8 dead zone")
        ax.axvline(0.8, color="orange", linewidth=1.2, linestyle="--")
        ax.axvline(-1.2, color="red", linewidth=1.0, linestyle=":", label="±1.2 hard wall")
        ax.axvline(1.2, color="red", linewidth=1.0, linestyle=":")
        pct_inside = float(np.mean(np.abs(z_dim) < 0.8) * 100)
        ax.set_title(f"z[{i}]  μ={z_dim.mean():.2f}  σ={z_dim.std():.2f}  {pct_inside:.0f}% inside",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6)

    for i in range(latent_dim, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("HausdorferAE — Latent Space Distribution (val set)", fontsize=11)
    fig.tight_layout()
    path = output_dir / "latent_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv=None):
    args = parse_args(argv)
    ckpt_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # ── load model ───────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    model = HausdorferAE.from_checkpoint(ckpt_path)
    print(f"  {model}")

    # ── load val strides ─────────────────────────────────────────────────────
    print(f"Loading data: {args.data}")
    strides, metadata = load_strides(args.data, with_metadata=True)
    _, s_val, _, _ = stride_train_val_split(
        strides, metadata, val_fraction=args.val_fraction, seed=args.seed,
    )
    print(f"  Val strides: {s_val.shape}")

    # ── map EMG → 18-muscle bilateral snapshots ───────────────────────────────
    snapshots_val = strides_to_snapshots(s_val)        # (N*101, 18)
    strides_18 = snapshots_val.reshape(-1, 101, 18)    # (N, 101, 18)

    # ── reconstruct ──────────────────────────────────────────────────────────
    print("Reconstructing val set...")
    snapshots_recon = model.reconstruct(snapshots_val)  # (N*101, 18)
    strides_recon = snapshots_recon.reshape(-1, 101, 18)

    z_val = model.transform(snapshots_val)              # (N*101, latent_dim)

    # ── global metrics ───────────────────────────────────────────────────────
    mse = float(np.mean((snapshots_val - snapshots_recon) ** 2))
    ss_res = np.sum((snapshots_val - snapshots_recon) ** 2)
    ss_tot = np.sum((snapshots_val - snapshots_val.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    pct_inside = float(np.mean(np.abs(z_val) < 0.8) * 100)
    print(f"  Val R²:         {r2:.4f}")
    print(f"  Val MSE:        {mse:.6f}")
    print(f"  pct inside ±0.8: {pct_inside:.1f}%")
    print(f"  Latent range:   [{z_val.min():.3f}, {z_val.max():.3f}]")

    # ── plots ────────────────────────────────────────────────────────────────
    n = min(args.n_strides, strides_18.shape[0])
    sample_idx = np.random.choice(strides_18.shape[0], size=n, replace=False)
    sample_idx.sort()

    print("Plotting gait profiles...")
    plot_gait_profiles(strides_18, strides_recon, sample_idx, output_dir)

    print("Plotting per-muscle R²...")
    plot_per_muscle_r2(snapshots_val, snapshots_recon, output_dir)

    print("Plotting latent histograms...")
    plot_latent_histograms(z_val, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
