"""Plot reconstruction quality across models and latent dimensions.

Reads training results (results.json) and checkpoints to produce:

1. R² vs latent_dim curve — all models on one plot, the key comparison.
2. Per-model stride reconstructions — original vs reconstructed for sample strides.
3. Mean gait profile overlay — average EMG envelope, original vs each model.
4. Per-muscle R² bar chart — which muscles are well/poorly reconstructed.

Usage:
    # After running train_strides.py with multiple latent dims:
    python -m leaps.scripts.plot_reconstructions \
        --data $LEAPS_EMG_H5 \
        --experiment-dir experiments/stride_sweep

    # Single latent dim, specific models:
    python -m leaps.scripts.plot_reconstructions \
        --data $LEAPS_EMG_H5 \
        --experiment-dir experiments/stride_sweep \
        --models StridePCA StrideNMF StrideCNMF StrideFlatAE \
        --latent-dims 9
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np

from leaps.data.metadata import EMG_CHANNEL_LABELS, EMG_CHANNELS
from leaps.data.stride_dataset import load_strides, stride_train_val_split
from leaps.models.metrics import per_muscle_r2
from leaps.training.stride_trainer import (
    STRIDE_PYTORCH_MODELS,
    STRIDE_SKLEARN_MODELS,
    build_stride_model,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot EMG reconstruction quality.")
    parser.add_argument("--data", required=True, help="Path to HDF5 data file.")
    parser.add_argument(
        "--experiment-dir", default="experiments/stride_sweep",
        help="Directory containing checkpoints/ and results.json.",
    )
    parser.add_argument("--output-dir", default=None, help="Plot output dir (default: experiment-dir/plots).")
    parser.add_argument("--models", nargs="+", default=None, help="Models to plot (default: auto-detect).")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=None, help="Latent dims (default: from results.json).")
    parser.add_argument("--n-samples", type=int, default=4, help="Number of sample strides for reconstruction plot.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Plot 1: R² vs latent dimension — the most important plot
# ---------------------------------------------------------------------------

def plot_r2_vs_latent_dim(results: list[dict], output_dir: Path) -> None:
    """Line plot of validation R² vs latent dimension for each model.

    This is the key figure: shows how reconstruction quality scales
    with latent dimension, and whether AE outperforms linear baselines.
    The 80% VAF (= 0.80 R²) threshold line is shown for reference —
    this is the standard criterion for sufficient muscle synergies.
    """
    # Group results by model
    models = {}
    for r in results:
        name = r["model"]
        if name not in models:
            models[name] = {"dims": [], "r2": [], "mse": []}
        models[name]["dims"].append(r["latent_dim"])
        # results.json stores val_r2 / val_mse directly (not nested under eval_metrics)
        r2 = r.get("val_r2", r.get("eval_metrics", {}).get("r2", float("nan")))
        mse = r.get("val_mse", r.get("eval_metrics", {}).get("mse", float("nan")))
        models[name]["r2"].append(r2)
        models[name]["mse"].append(mse)

    # Style: distinct markers for each model type
    style_map = {
        "StridePCA":     ("o", "#2196F3", "PCA (linear)"),
        "StrideNMF":     ("s", "#FF9800", "NMF (synergies)"),
        "StrideCNMF":    ("D", "#9C27B0", "CNMF"),
        "StrideAE":      ("^", "#4CAF50", "AE Conv1D"),
        "StrideVAE":     ("v", "#F44336", "VAE Conv1D"),
        "StrideWAE":     ("<", "#795548", "WAE Conv1D"),
        "StrideFlatAE":  ("P", "#009688", "AE MLP"),
        "StrideFlatVAE": ("*", "#E91E63", "VAE MLP"),
        "StrideMAE":     ("h", "#607D8B", "MAE Conv1D"),
    }

    fig, (ax_r2, ax_mse) = plt.subplots(1, 2, figsize=(12, 5))

    for name, data in sorted(models.items()):
        marker, color, label = style_map.get(name, ("x", "gray", name))
        order = np.argsort(data["dims"])
        dims = np.array(data["dims"])[order]
        r2 = np.array(data["r2"])[order]
        mse_vals = np.array(data["mse"])[order]

        ax_r2.plot(dims, r2, marker=marker, color=color, label=label, linewidth=2, markersize=7)
        ax_mse.plot(dims, mse_vals, marker=marker, color=color, label=label, linewidth=2, markersize=7)

    # VAF = 80% threshold
    ax_r2.axhline(0.80, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="VAF=80%")

    ax_r2.set_xlabel("Latent Dimension", fontsize=11)
    ax_r2.set_ylabel("R² (Validation)", fontsize=11)
    ax_r2.set_title("Reconstruction Quality vs Latent Dimension", fontsize=12)
    ax_r2.legend(fontsize=9)
    ax_r2.grid(True, alpha=0.3)
    ax_r2.set_ylim(0, 1.05)

    ax_mse.set_xlabel("Latent Dimension", fontsize=11)
    ax_mse.set_ylabel("MSE (Validation)", fontsize=11)
    ax_mse.set_title("Reconstruction Error vs Latent Dimension", fontsize=12)
    ax_mse.legend(fontsize=9)
    ax_mse.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = output_dir / "r2_vs_latent_dim.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: Per-model stride reconstructions
# ---------------------------------------------------------------------------

def plot_stride_reconstructions(
    x_val: np.ndarray,
    models: dict[str, object],
    sample_indices: np.ndarray,
    output_dir: Path,
) -> None:
    """Grid of original vs reconstructed strides for each model.

    One figure per model: rows = 11 muscles, columns = sample strides.
    Blue = original EMG, red dashed = reconstruction.
    """
    n_muscles = x_val.shape[-1]
    n_samples = len(sample_indices)
    gait_pct = np.linspace(0, 100, x_val.shape[1])
    muscle_labels = [EMG_CHANNEL_LABELS.get(ch, ch) for ch in EMG_CHANNELS[:n_muscles]]

    for model_name, model in models.items():
        if model is None:
            continue

        fig, axes = plt.subplots(
            n_muscles, n_samples,
            figsize=(3.5 * n_samples, 2 * n_muscles),
            sharex=True,
        )
        if n_samples == 1:
            axes = axes[:, np.newaxis]

        x_samples = x_val[sample_indices]
        x_hat = model.reconstruct(x_samples)

        for col in range(n_samples):
            for row in range(n_muscles):
                ax = axes[row, col]
                ax.plot(gait_pct, x_samples[col, :, row], "b-", linewidth=1.2, alpha=0.8)
                ax.plot(gait_pct, x_hat[col, :, row], "r--", linewidth=1.2, alpha=0.8)
                ax.set_ylim(-0.05, 1.15)
                if col == 0:
                    ax.set_ylabel(muscle_labels[row], fontsize=7, rotation=0, ha="right")
                if row == 0:
                    ax.set_title(f"Stride {sample_indices[col]}", fontsize=9)
                if row == n_muscles - 1:
                    ax.set_xlabel("% Gait Cycle", fontsize=8)
                if row == 0 and col == n_samples - 1:
                    ax.plot([], [], "b-", label="Original")
                    ax.plot([], [], "r--", label="Recon")
                    ax.legend(fontsize=7, loc="upper right")
                ax.tick_params(labelsize=6)

        fig.suptitle(f"{model_name} — Stride Reconstructions", fontsize=12)
        fig.tight_layout()
        save_path = output_dir / f"recon_{model_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Plot 3: Mean gait profile comparison
# ---------------------------------------------------------------------------

def plot_mean_profile_comparison(
    x_val: np.ndarray,
    models: dict[str, object],
    output_dir: Path,
) -> None:
    """Mean EMG envelope per muscle — original (black) vs all models (colored).

    Shows whether each model captures the average gait cycle shape.
    """
    n_muscles = x_val.shape[-1]
    gait_pct = np.linspace(0, 100, x_val.shape[1])
    muscle_labels = [EMG_CHANNEL_LABELS.get(ch, ch) for ch in EMG_CHANNELS[:n_muscles]]
    mean_orig = x_val.mean(axis=0)  # (101, 11)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
    ncols = 4
    nrows = (n_muscles + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
    axes = np.atleast_2d(axes)

    for i in range(n_muscles):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        ax.plot(gait_pct, mean_orig[:, i], "k-", linewidth=2.0, label="Original")

        for j, (name, model) in enumerate(models.items()):
            if model is None:
                continue
            mean_recon = model.reconstruct(x_val).mean(axis=0)
            ax.plot(gait_pct, mean_recon[:, i], color=colors[j], linestyle="--",
                    linewidth=1.3, alpha=0.85, label=name)

        ax.set_ylim(-0.05, 1.05)
        ax.set_title(muscle_labels[i], fontsize=9)
        ax.tick_params(labelsize=7)
        if row == nrows - 1:
            ax.set_xlabel("% Gait Cycle", fontsize=8)

    for i in range(n_muscles, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(models) + 1, 6), fontsize=8)
    fig.suptitle("Mean Gait Profile — All Models", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    save_path = output_dir / "mean_profile_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Plot 4: Per-muscle R² bar chart
# ---------------------------------------------------------------------------

def plot_per_muscle_r2(
    x_val: np.ndarray,
    models: dict[str, object],
    output_dir: Path,
) -> None:
    """Grouped bar chart of R² per muscle for each model.

    Reveals which muscles are poorly reconstructed — important for
    checking if the model captures all synergy groups.
    """
    n_muscles = x_val.shape[-1]
    muscle_labels = [EMG_CHANNEL_LABELS.get(ch, ch) for ch in EMG_CHANNELS[:n_muscles]]

    valid_models = {n: m for n, m in models.items() if m is not None}
    n_models = len(valid_models)
    if n_models == 0:
        return

    x = np.arange(n_muscles)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, (name, model) in enumerate(valid_models.items()):
        x_hat = model.reconstruct(x_val)
        x_flat = x_val.reshape(-1, n_muscles)
        x_hat_flat = x_hat.reshape(-1, n_muscles)
        r2_vals = per_muscle_r2(x_flat, x_hat_flat)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, r2_vals, width, label=name, color=colors[i], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(muscle_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("Per-Muscle Reconstruction Quality", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.80, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save_path = output_dir / "per_muscle_r2.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_name: str, latent_dim: int, ckpt_dir: Path):
    """Load a trained stride model from checkpoint."""
    model = build_stride_model(model_name, latent_dim)
    ext = ".pkl" if model_name in STRIDE_SKLEARN_MODELS else ".pt"
    ckpt_path = ckpt_dir / f"{model.name}_d{latent_dim}{ext}"
    if not ckpt_path.exists():
        print(f"  WARNING: checkpoint not found: {ckpt_path}")
        return None
    model.load(ckpt_path)
    return model


def _detect_available(ckpt_dir: Path) -> list[tuple[str, int]]:
    """Detect (model_name, latent_dim) pairs from checkpoint filenames."""
    # Build class_name → registry_name mapping
    all_reg_names = sorted(STRIDE_SKLEARN_MODELS | STRIDE_PYTORCH_MODELS)
    class_to_reg = {}
    for reg_name in all_reg_names:
        m = build_stride_model(reg_name, 2)  # dummy dim to get class name
        class_to_reg[m.name] = reg_name

    found = []
    for p in sorted(ckpt_dir.iterdir()):
        stem = p.stem  # e.g. "StridePCAModel_d9" or "StrideFlatAE_d9"
        for class_name, reg_name in class_to_reg.items():
            if stem.startswith(class_name + "_d"):
                try:
                    dim = int(stem[len(class_name) + 2:])
                    found.append((reg_name, dim))
                except ValueError:
                    pass
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    exp_dir = Path(args.experiment_dir)
    ckpt_dir = exp_dir / "checkpoints"
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # --- Load results.json for R² vs latent_dim plot ---
    results_path = exp_dir / "results.json"
    results = []
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        # Filter to requested models/dims if specified
        if args.models:
            results = [r for r in results if r["model"] in args.models]
        if args.latent_dims:
            results = [r for r in results if r["latent_dim"] in args.latent_dims]

    if results:
        print("Plotting R² vs latent dimension...")
        plot_r2_vs_latent_dim(results, output_dir)
    else:
        print("No results.json found — skipping R² vs latent_dim plot.")

    # --- Load data for reconstruction plots ---
    print(f"Loading stride data from {args.data}...")
    strides, metadata = load_strides(args.data, with_metadata=True)
    _, x_val, _, _ = stride_train_val_split(
        strides, metadata, val_fraction=0.1, seed=args.seed,
    )
    print(f"  Val strides: {x_val.shape}")

    # --- Determine which models/dims to plot reconstructions for ---
    if args.models and args.latent_dims:
        pairs = [(m, d) for m in args.models for d in args.latent_dims]
    elif results:
        # Use the best latent_dim per model (highest R²)
        best = {}
        for r in results:
            name = r["model"]
            r2 = r.get("val_r2", r.get("eval_metrics", {}).get("r2", -1))
            if name not in best or r2 > best[name][1]:
                best[name] = (r["latent_dim"], r2)
        pairs = [(name, dim) for name, (dim, _) in best.items()]
    else:
        # Auto-detect from checkpoints
        pairs = _detect_available(ckpt_dir)

    if not pairs:
        print("No models found to plot.")
        return

    print(f"Loading models for reconstruction plots: {pairs}")
    models = {}
    for name, dim in pairs:
        label = f"{name}_d{dim}"
        m = _load_model(name, dim, ckpt_dir)
        if m is not None:
            models[label] = m

    if not models:
        print("No models loaded successfully.")
        return

    # Pick sample strides
    sample_indices = np.random.choice(x_val.shape[0], size=min(args.n_samples, x_val.shape[0]), replace=False)
    sample_indices.sort()

    # Plots 2-4
    print("Plotting stride reconstructions...")
    plot_stride_reconstructions(x_val, models, sample_indices, output_dir)

    print("Plotting mean gait profile comparison...")
    plot_mean_profile_comparison(x_val, models, output_dir)

    print("Plotting per-muscle R²...")
    plot_per_muscle_r2(x_val, models, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
