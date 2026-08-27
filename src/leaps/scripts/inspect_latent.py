"""Inspect the learned latent space of a trained stride model.

Checks that are actually meaningful for a deterministic AE (FlatAE):
  1. Latent distribution — what z values does the encoder actually produce?
  2. Reconstructions on real val strides — ground truth quality check
  3. Linear interpolation between two real z points — is the manifold smooth?
  4. z=0 decode — what's at the latent centroid?
  5. Latent space coloured by speed/mode — does z organise by condition?

NOT done here: sampling from N(0,1) — that's only meaningful for a VAE with
a Gaussian prior. FlatAE has no prior; its latent std is ~0.32, not 1.0.
Sampling from N(0,1) would produce mostly out-of-distribution z values.

Usage:
    python -m leaps.scripts.inspect_latent \\
        --data $LEAPS_EMG_H5 \\
        --checkpoint experiments/stride_flatvae/checkpoints/StrideFlatAE_d8.pt \\
        --model StrideFlatAE --latent-dim 8 \\
        --output-dir experiments/stride_flatvae/latent_inspection
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from leaps.data.metadata import EMG_CHANNEL_LABELS, EMG_CHANNELS
from leaps.data.stride_dataset import load_strides, stride_train_val_split
from leaps.training.stride_trainer import build_stride_model, STRIDE_SKLEARN_MODELS


# ── Args ──────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="StrideFlatAE")
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--output-dir", default="experiments/latent_inspection")
    p.add_argument("--n-recon-samples", type=int, default=6,
                   help="Number of val strides to show in reconstruction plot.")
    p.add_argument("--n-interp-steps", type=int, default=8,
                   help="Number of steps in latent interpolation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    return p.parse_args(argv)


# ── Helpers ───────────────────────────────────────────────────────────────────


MUSCLE_LABELS = [EMG_CHANNEL_LABELS.get(ch, ch) for ch in EMG_CHANNELS]
GAIT_PCT = np.linspace(0, 100, 101)


def _muscle_axes_grid(n_muscles, n_cols, figsize, sharex=True):
    n_rows = (n_muscles + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex)
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat[n_muscles:]:
        ax.set_visible(False)
    return fig, axes_flat[:n_muscles]


# ── Plot 1: Latent distribution ───────────────────────────────────────────────


def plot_latent_distribution(z: np.ndarray, output_dir: Path) -> None:
    """Histogram of each latent dimension + pairplot of first 4 dims.

    Shows: what range does the encoder actually use? Are dims gaussian-ish?
    Is each dim active (std > 0.1) or collapsed?
    """
    d = z.shape[1]

    # --- Per-dim histograms ---
    ncols = min(d, 4)
    nrows = (d + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes_flat = np.array(axes).flatten()

    for i in range(d):
        ax = axes_flat[i]
        ax.hist(z[:, i], bins=50, color="#009688", alpha=0.8, edgecolor="none")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"z[{i}]  μ={z[:,i].mean():.2f}  σ={z[:,i].std():.3f}", fontsize=9)
        ax.set_xlabel("Latent value", fontsize=8)
        if z[:, i].std() < 0.1:
            ax.set_facecolor("#FFEBEE")  # red tint = collapsed dim
            ax.set_title(ax.get_title() + "  ⚠ COLLAPSED", fontsize=9, color="red")
    for ax in axes_flat[d:]:
        ax.set_visible(False)

    fig.suptitle(f"Latent dimension distributions  (n={len(z)} val strides)", fontsize=12)
    fig.tight_layout()
    path = output_dir / "latent_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Pairplot of first min(d,4) dims ---
    d_plot = min(d, 4)
    fig, axes = plt.subplots(d_plot, d_plot, figsize=(3 * d_plot, 3 * d_plot))
    for i in range(d_plot):
        for j in range(d_plot):
            ax = axes[i][j]
            if i == j:
                ax.hist(z[:, i], bins=40, color="#009688", alpha=0.7)
                ax.set_title(f"z[{i}]", fontsize=8)
            else:
                ax.scatter(z[:, j], z[:, i], s=2, alpha=0.3, color="#455A64")
            if i < d_plot - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
    fig.suptitle("Latent pairplot (first 4 dims)", fontsize=11)
    fig.tight_layout()
    path = output_dir / "latent_pairplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 2: Reconstructions on real val strides ───────────────────────────────


def plot_reconstructions(
    x_val: np.ndarray, model, indices: np.ndarray, output_dir: Path
) -> None:
    """Original (blue) vs reconstructed (red) for N val strides, all 11 muscles."""
    n = len(indices)
    n_muscles = x_val.shape[-1]

    fig, axes = plt.subplots(
        n_muscles, n, figsize=(3 * n, 2 * n_muscles), sharex=True
    )
    if n == 1:
        axes = axes[:, np.newaxis]

    x_samples = x_val[indices]
    x_hat = model.reconstruct(x_samples)
    # per-sample R²
    r2_samples = []
    for k in range(n):
        ss_res = ((x_samples[k] - x_hat[k]) ** 2).sum()
        ss_tot = ((x_samples[k] - x_samples[k].mean()) ** 2).sum()
        r2_samples.append(1.0 - ss_res / (ss_tot + 1e-8))

    for col in range(n):
        for row in range(n_muscles):
            ax = axes[row, col]
            ax.plot(GAIT_PCT, x_samples[col, :, row], "b-", linewidth=1.2, alpha=0.85)
            ax.plot(GAIT_PCT, x_hat[col, :, row], "r--", linewidth=1.2, alpha=0.85)
            ax.set_ylim(-0.05, 1.15)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(MUSCLE_LABELS[row], fontsize=7, rotation=0, ha="right", labelpad=55)
            if row == n_muscles - 1:
                ax.set_xlabel("% Gait", fontsize=8)
        axes[0, col].set_title(f"Stride {indices[col]}\nR²={r2_samples[col]:.3f}", fontsize=8)

    # Legend
    axes[0, -1].plot([], [], "b-", label="Original")
    axes[0, -1].plot([], [], "r--", label="Recon")
    axes[0, -1].legend(fontsize=7, loc="upper right")

    fig.suptitle("Val Stride Reconstructions", fontsize=12)
    fig.tight_layout()
    path = output_dir / "reconstructions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 3: Latent interpolation between two real strides ─────────────────────


def plot_interpolation(
    x_val: np.ndarray, z_val: np.ndarray, model,
    n_steps: int, output_dir: Path, rng: np.random.Generator,
) -> None:
    """Decode a linear path z_a → z_b in latent space.

    Both z_a and z_b come from REAL encoded strides (not sampled).
    Pick two strides that differ visibly — one slow treadmill-like, one fast-looking.
    If no metadata, just pick two random val strides far apart in latent space.

    This is the meaningful test: does the decoder interpolate smoothly,
    or does it jump / collapse in the middle?
    """
    # Pick the two val strides most distant in latent space
    dists = np.linalg.norm(z_val[:, None] - z_val[None, :], axis=-1)  # (N, N)
    idx_a, idx_b = np.unravel_index(np.argmax(dists), dists.shape)

    alphas = np.linspace(0, 1, n_steps)
    z_interp = np.array([(1 - a) * z_val[idx_a] + a * z_val[idx_b] for a in alphas])
    x_interp = model.decode(z_interp).reshape(n_steps, 101, -1)

    n_muscles = x_val.shape[-1]
    fig, axes = plt.subplots(
        n_muscles, n_steps + 2, figsize=(2.5 * (n_steps + 2), 2 * n_muscles), sharex=True
    )
    if n_muscles == 1:
        axes = axes[np.newaxis, :]

    # Endpoints: original strides
    for row in range(n_muscles):
        axes[row, 0].plot(GAIT_PCT, x_val[idx_a, :, row], "b-", linewidth=1.3)
        axes[row, -1].plot(GAIT_PCT, x_val[idx_b, :, row], "b-", linewidth=1.3)
        axes[row, 0].set_ylim(-0.05, 1.15)
        axes[row, -1].set_ylim(-0.05, 1.15)
        if row == n_muscles - 1:
            axes[row, 0].set_xlabel("% Gait", fontsize=7)
            axes[row, -1].set_xlabel("% Gait", fontsize=7)
        axes[row, 0].set_ylabel(MUSCLE_LABELS[row], fontsize=7, rotation=0, ha="right", labelpad=55)
        axes[row, 0].tick_params(labelsize=5)
        axes[row, -1].tick_params(labelsize=5)

    axes[0, 0].set_title(f"Original A\n(stride {idx_a})", fontsize=8, color="blue")
    axes[0, -1].set_title(f"Original B\n(stride {idx_b})", fontsize=8, color="blue")

    # Interpolated steps
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_steps))
    for step in range(n_steps):
        col = step + 1
        for row in range(n_muscles):
            axes[row, col].plot(GAIT_PCT, x_interp[step, :, row],
                                color=colors[step], linewidth=1.2)
            axes[row, col].set_ylim(-0.05, 1.15)
            axes[row, col].tick_params(labelsize=5)
        axes[0, col].set_title(f"α={alphas[step]:.2f}", fontsize=8)

    fig.suptitle(
        f"Latent Interpolation: stride {idx_a} → {idx_b}\n"
        f"Latent dist = {dists[idx_a, idx_b]:.3f}  |  "
        f"Look for smooth transitions — jumps indicate poor manifold structure",
        fontsize=10,
    )
    fig.tight_layout()
    path = output_dir / "latent_interpolation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 4: z=0 decode ────────────────────────────────────────────────────────


def plot_zero_decode(model, n_channels: int, output_dir: Path) -> None:
    """Decode z=0. This is the latent centroid (Lnorm pushes z near 0).

    Should look like an average gait cycle. If it's garbage, the latent
    space isn't organised around the data mean.
    """
    z_zero = np.zeros((1, model.latent_dim), dtype=np.float32)
    x_zero = model.decode(z_zero).reshape(101, n_channels)
    x_zero = np.clip(x_zero, 0, 1)

    ncols = 4
    fig, axes = _muscle_axes_grid(n_channels, ncols, figsize=(4 * ncols, 3 * ((n_channels + ncols - 1) // ncols)))

    for i, ax in enumerate(axes):
        ax.plot(GAIT_PCT, x_zero[:, i], color="#009688", linewidth=1.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(MUSCLE_LABELS[i], fontsize=9)
        ax.tick_params(labelsize=7)
        if i >= n_channels - ncols:
            ax.set_xlabel("% Gait Cycle", fontsize=8)

    fig.suptitle("Decoded z=0 — latent centroid\n"
                 "Should look like an average gait cycle. "
                 "Garbage here = latent space not centred on data.", fontsize=11)
    fig.tight_layout()
    path = output_dir / "z0_decode.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 5: Latent coloured by condition ──────────────────────────────────────


def plot_latent_by_condition(
    z: np.ndarray, metadata: dict, output_dir: Path
) -> None:
    """PCA projection of z coloured by speed and mode.

    If the latent space is organised, treadmill strides at similar speeds
    should cluster together, and different modes should be separable.
    This tells you whether z encodes biomechanically meaningful structure.
    """
    from sklearn.decomposition import PCA as skPCA

    pca = skPCA(n_components=2)
    z2 = pca.fit_transform(z)
    var = pca.explained_variance_ratio_

    modes = np.array([m.decode() if isinstance(m, bytes) else m for m in metadata["modes"]])
    speeds = metadata.get("speeds", np.full(len(z), np.nan))

    mode_colors = {"treadmill": "#2196F3", "levelground": "#4CAF50",
                   "ramp": "#FF9800", "stair": "#9C27B0"}
    unique_modes = sorted(set(modes))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colour by mode
    ax = axes[0]
    for mode in unique_modes:
        mask = modes == mode
        ax.scatter(z2[mask, 0], z2[mask, 1], s=4, alpha=0.4,
                   color=mode_colors.get(mode, "gray"), label=mode)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=11)
    ax.set_title("Latent space — coloured by mode", fontsize=11)
    ax.legend(fontsize=9, markerscale=4)
    ax.grid(True, alpha=0.2)

    # Right: colour by speed (treadmill only)
    ax = axes[1]
    tread_mask = (modes == "treadmill") & ~np.isnan(speeds.astype(float))
    if tread_mask.sum() > 0:
        spd = speeds[tread_mask].astype(float)
        sc = ax.scatter(z2[tread_mask, 0], z2[tread_mask, 1],
                        s=5, alpha=0.5, c=spd, cmap="plasma",
                        vmin=spd.min(), vmax=spd.max())
        plt.colorbar(sc, ax=ax, label="Speed (m/s)")
        ax.set_title("Latent space — treadmill, coloured by speed\n"
                     "Gradient → speed is encoded. Random → speed is lost.", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No treadmill speed data available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Speed (no data)", fontsize=11)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=11)
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Latent space PCA (2D projection of z)\n"
        "Structure here = model learned meaningful variation. "
        "Blob = model compressed but didn't organise.",
        fontsize=11,
    )
    fig.tight_layout()
    path = output_dir / "latent_by_condition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_stride_model(args.model, args.latent_dim)
    ext = ".pkl" if args.model in STRIDE_SKLEARN_MODELS else ".pt"
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load(ckpt)
    print(f"Loaded {args.model} d={args.latent_dim} from {ckpt}")

    # Load val data (same split logic as training)
    strides, metadata = load_strides(args.data, with_metadata=True)
    _, x_val, _, meta_val = stride_train_val_split(
        strides, metadata, val_fraction=args.val_fraction, seed=args.seed,
    )
    print(f"Val set: {x_val.shape}  ({len(np.unique(meta_val['subjects']))} subjects)")

    n_muscles = x_val.shape[-1]

    # Encode all val strides
    print("Encoding val strides...")
    z_val = model.transform(x_val)
    print(f"  z_val: {z_val.shape}  "
          f"std_mean={z_val.std(axis=0).mean():.3f}  "
          f"std_min={z_val.std(axis=0).min():.3f}  "
          f"range=[{z_val.min():.3f}, {z_val.max():.3f}]")

    # Global R²
    x_hat = model.reconstruct(x_val)
    ss_res = ((x_val - x_hat) ** 2).sum()
    ss_tot = ((x_val - x_val.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    print(f"  Val R²: {r2:.4f}")

    # Pick sample indices for reconstruction plot
    sample_idx = rng.choice(len(x_val), size=min(args.n_recon_samples, len(x_val)), replace=False)
    sample_idx.sort()

    print("\nPlotting latent distributions...")
    plot_latent_distribution(z_val, output_dir)

    print("Plotting reconstructions on real val strides...")
    plot_reconstructions(x_val, model, sample_idx, output_dir)

    print("Plotting latent interpolation...")
    plot_interpolation(x_val, z_val, model, args.n_interp_steps, output_dir, rng)

    print("Decoding z=0...")
    plot_zero_decode(model, n_muscles, output_dir)

    print("Plotting latent space coloured by condition...")
    plot_latent_by_condition(z_val, meta_val, output_dir)

    print(f"\nAll plots saved to {output_dir}/")
    print("\nWhat to look for:")
    print("  latent_distributions.png  — red-tinted dims are collapsed (bad)")
    print("  reconstructions.png       — per-stride R² in title. Low = decoder can't fit this stride")
    print("  latent_interpolation.png  — smooth = well-organised manifold. Jumps = not smooth")
    print("  z0_decode.png             — should look like an average gait cycle")
    print("  latent_by_condition.png   — speed gradient = model learned speed. Blob = it didn't")


if __name__ == "__main__":
    main()
