"""Cross-experiment EDA: aggregate results.json files, save metadata CSV, and plot comparisons.

Loads results.json from one or more experiment directories, merges them into a
single DataFrame (saved as CSV), and produces:

  1. R² vs latent_dim — all models on one plot
  2. Per-muscle R² heatmap — models × muscles
  3. Latent diagnostics — std_mean, std_min, range per model
  4. Architecture vs regularisation table (FlatAE / FlatVAE / StrideVAE comparison)
  5. Model ranking table printed to stdout

Usage:
    # After stride_sweep_v3 and stride_flatvae have finished:
    python -m leaps.scripts.analyze_runs \\
        --experiment-dirs experiments/stride_sweep_v3 experiments/stride_flatvae \\
        --output-dir experiments/eda

    # Single experiment:
    python -m leaps.scripts.analyze_runs \\
        --experiment-dirs experiments/stride_flatvae \\
        --output-dir experiments/stride_flatvae/eda
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from leaps.data.metadata import EMG_CHANNEL_LABELS, EMG_CHANNELS


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Cross-experiment EDA: aggregate results and plot comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--experiment-dirs", nargs="+", required=True,
        help="One or more experiment directories containing results.json.",
    )
    p.add_argument(
        "--output-dir", default="experiments/eda",
        help="Directory to save plots and aggregated CSV.",
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Filter to specific models (default: all found).",
    )
    p.add_argument(
        "--latent-dims", type=int, nargs="+", default=None,
        help="Filter to specific latent dims (default: all found).",
    )
    return p.parse_args(argv)


# ── Data loading ──────────────────────────────────────────────────────────────


def load_all_results(experiment_dirs: list[str], models_filter=None, dims_filter=None) -> pd.DataFrame:
    """Load and merge results.json from multiple experiment directories into a DataFrame."""
    rows = []
    for exp_dir in experiment_dirs:
        results_path = Path(exp_dir) / "results.json"
        if not results_path.exists():
            print(f"  WARNING: no results.json in {exp_dir} — skipping")
            continue
        with open(results_path) as f:
            results = json.load(f)
        for r in results:
            if models_filter and r["model"] not in models_filter:
                continue
            if dims_filter and r["latent_dim"] not in dims_filter:
                continue

            row = {
                "experiment": Path(exp_dir).name,
                "model": r["model"],
                "latent_dim": r["latent_dim"],
                "val_r2": r.get("val_r2", float("nan")),
                "val_mse": r.get("val_mse", float("nan")),
                "latent_std_mean": r.get("latent_std_mean", float("nan")),
                "latent_std_min": r.get("latent_std_min", float("nan")),
                "latent_std_max": r.get("latent_std_max", float("nan")),
                "latent_min": r.get("latent_min", float("nan")),
                "latent_max": r.get("latent_max", float("nan")),
                "checkpoint": r.get("checkpoint", ""),
            }
            # Per-muscle R²
            for ch in r.get("per_muscle_r2", {}):
                row[f"r2_{ch}"] = r["per_muscle_r2"][ch]

            # Hyperparameters (flatten into columns)
            hp = r.get("hyperparameters", {})
            row["epochs"] = hp.get("epochs", None)
            row["batch_size"] = hp.get("batch_size", None)
            row["beta"] = hp.get("beta", None)
            row["mmd_weight"] = hp.get("mmd_weight", None)
            row["lnorm_weight"] = hp.get("lnorm_weight", None)

            rows.append(row)

    df = pd.DataFrame(rows)
    # Drop duplicate (experiment, model, latent_dim) — keep last (most recent run)
    df = df.drop_duplicates(subset=["experiment", "model", "latent_dim"], keep="last")
    df = df.sort_values(["model", "latent_dim"]).reset_index(drop=True)
    return df


# ── Style helpers ─────────────────────────────────────────────────────────────


MODEL_STYLE = {
    "StridePCA":     ("o", "#2196F3", "PCA (linear)"),
    "StrideNMF":     ("s", "#FF9800", "NMF (synergies)"),
    "StrideCNMF":    ("D", "#9C27B0", "CNMF"),
    "StrideAE":      ("^", "#4CAF50", "AE Conv1D"),
    "StrideVAE":     ("v", "#F44336", "VAE Conv1D"),
    "StrideWAE":     ("<", "#795548", "WAE Conv1D"),
    "StrideFlatAE":  ("P", "#009688", "AE MLP  ← best"),
    "StrideFlatVAE": ("*", "#E91E63", "VAE MLP  ← ablation"),
    "StrideMAE":     ("h", "#607D8B", "MAE Conv1D"),
}


def _style(model_name):
    return MODEL_STYLE.get(model_name, ("x", "gray", model_name))


# ── Plot 1: R² vs latent_dim ──────────────────────────────────────────────────


def plot_r2_vs_latent_dim(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: validation R² vs latent dimension, all models."""
    fig, (ax_r2, ax_mse) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, grp in df.groupby("model"):
        grp = grp.sort_values("latent_dim")
        marker, color, label = _style(model_name)
        ax_r2.plot(grp["latent_dim"], grp["val_r2"],
                   marker=marker, color=color, label=label, linewidth=2, markersize=8)
        ax_mse.plot(grp["latent_dim"], grp["val_mse"],
                    marker=marker, color=color, label=label, linewidth=2, markersize=8)

    ax_r2.axhline(0.80, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="R²=0.80 target")
    ax_r2.set_xlabel("Latent Dimension", fontsize=12)
    ax_r2.set_ylabel("R² (Validation)", fontsize=12)
    ax_r2.set_title("Reconstruction Quality vs Latent Dimension", fontsize=13)
    ax_r2.legend(fontsize=8, loc="lower right")
    ax_r2.grid(True, alpha=0.3)
    ax_r2.set_ylim(0, 1.05)

    ax_mse.set_xlabel("Latent Dimension", fontsize=12)
    ax_mse.set_ylabel("MSE (Validation)", fontsize=12)
    ax_mse.set_title("Reconstruction Error vs Latent Dimension", fontsize=13)
    ax_mse.legend(fontsize=8)
    ax_mse.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "r2_vs_latent_dim.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 2: Architecture vs Regularisation (ablation) ────────────────────────


def plot_arch_vs_reg(df: pd.DataFrame, output_dir: Path) -> None:
    """2×2 grid showing the architecture/regularisation decomposition.

    Requires StrideFlatAE, StrideFlatVAE, StrideAE, StrideVAE.
    If any are missing the plot is skipped gracefully.
    """
    ablation_models = ["StrideFlatAE", "StrideFlatVAE", "StrideAE", "StrideVAE"]
    present = [m for m in ablation_models if m in df["model"].values]
    if len(present) < 2:
        print("  Skipping arch vs reg plot — need at least StrideFlatAE + StrideFlatVAE")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name in present:
        grp = df[df["model"] == model_name].sort_values("latent_dim")
        marker, color, label = _style(model_name)
        ax.plot(grp["latent_dim"], grp["val_r2"],
                marker=marker, color=color, label=label, linewidth=2.5, markersize=9)

    ax.axhline(0.80, color="gray", linestyle=":", linewidth=1.5, alpha=0.6, label="R²=0.80 target")

    # Annotate the gaps at d=8
    d = 8
    vals = {}
    for m in present:
        row = df[(df["model"] == m) & (df["latent_dim"] == d)]
        if not row.empty:
            vals[m] = row["val_r2"].iloc[0]

    if "StrideFlatAE" in vals and "StrideFlatVAE" in vals:
        gap = vals["StrideFlatAE"] - vals["StrideFlatVAE"]
        ax.annotate(
            f"KL cost: Δ={gap:.3f}",
            xy=(d, (vals["StrideFlatAE"] + vals["StrideFlatVAE"]) / 2),
            xytext=(d + 0.5, (vals["StrideFlatAE"] + vals["StrideFlatVAE"]) / 2),
            fontsize=9, color="#009688",
            arrowprops=dict(arrowstyle="-", color="#009688"),
        )
    if "StrideFlatVAE" in vals and "StrideVAE" in vals:
        gap = vals["StrideFlatVAE"] - vals["StrideVAE"]
        ax.annotate(
            f"Arch cost (Conv1D): Δ={gap:.3f}",
            xy=(d, (vals["StrideFlatVAE"] + vals["StrideVAE"]) / 2),
            xytext=(d + 0.5, (vals["StrideFlatVAE"] + vals["StrideVAE"]) / 2),
            fontsize=9, color="#E91E63",
            arrowprops=dict(arrowstyle="-", color="#E91E63"),
        )

    ax.set_xlabel("Latent Dimension", fontsize=12)
    ax.set_ylabel("R² (Validation)", fontsize=12)
    ax.set_title("Architecture vs Regularisation Ablation\n"
                 "FlatAE→FlatVAE: cost of KL | FlatVAE→StrideVAE: cost of Conv1D", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    path = output_dir / "arch_vs_reg_ablation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 3: Per-muscle R² heatmap ─────────────────────────────────────────────


def plot_per_muscle_heatmap(df: pd.DataFrame, output_dir: Path, latent_dim: int = 8) -> None:
    """Heatmap: models × muscles, coloured by R². One dim at a time."""
    muscle_cols = [f"r2_{ch}" for ch in EMG_CHANNELS if f"r2_{ch}" in df.columns]
    if not muscle_cols:
        print("  Skipping per-muscle heatmap — no per_muscle_r2 in results")
        return

    sub = df[df["latent_dim"] == latent_dim].copy()
    if sub.empty:
        print(f"  Skipping per-muscle heatmap — no results for d={latent_dim}")
        return

    sub = sub.set_index("model")[muscle_cols]
    sub.columns = [c.replace("r2_", "") for c in sub.columns]
    muscle_labels = [EMG_CHANNEL_LABELS.get(ch, ch) for ch in sub.columns]

    fig, ax = plt.subplots(figsize=(max(12, len(sub.columns) * 1.1), max(4, len(sub) * 0.7)))
    im = ax.imshow(sub.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(sub.columns)))
    ax.set_xticklabels(muscle_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels(sub.index.tolist(), fontsize=9)
    ax.set_title(f"Per-Muscle R² — Latent dim = {latent_dim}", fontsize=12)

    # Annotate cells
    for i in range(len(sub.index)):
        for j in range(len(sub.columns)):
            val = sub.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < val < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="R²", shrink=0.8)
    fig.tight_layout()
    path = output_dir / f"per_muscle_r2_heatmap_d{latent_dim}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 4: Latent diagnostics ────────────────────────────────────────────────


def plot_latent_diagnostics(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter and bar plots of latent space statistics for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Filter to a few key dims for readability
    key_dims = sorted(df["latent_dim"].unique())
    models = df["model"].unique()

    # 1. latent_std_mean vs val_r2
    ax = axes[0]
    for model_name in models:
        grp = df[df["model"] == model_name]
        marker, color, label = _style(model_name)
        ax.scatter(grp["latent_std_mean"], grp["val_r2"],
                   marker=marker, color=color, label=label, s=80, zorder=3)
        for _, row in grp.iterrows():
            ax.annotate(f"d={row['latent_dim']}", (row["latent_std_mean"], row["val_r2"]),
                        fontsize=6, alpha=0.6)
    ax.set_xlabel("Latent std mean", fontsize=11)
    ax.set_ylabel("Val R²", fontsize=11)
    ax.set_title("Latent Usage vs Reconstruction", fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. latent_std_min — collapsed dim detector
    ax = axes[1]
    pivot_min = df.pivot_table(index="model", columns="latent_dim", values="latent_std_min")
    x = np.arange(len(pivot_min.index))
    width = 0.8 / len(pivot_min.columns)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(pivot_min.columns)))
    for i, (dim, col_vals) in enumerate(pivot_min.items()):
        offset = (i - len(pivot_min.columns) / 2 + 0.5) * width
        ax.bar(x + offset, col_vals.values, width, label=f"d={dim}",
               color=colors[i], alpha=0.85)
    ax.axhline(0.1, color="red", linestyle=":", linewidth=1.5, label="Collapse threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_min.index.tolist(), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Min std across dims", fontsize=11)
    ax.set_title("Latent Dim Collapse Check\n(red = collapse threshold 0.1)", fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    # 3. Latent range [min, max] — RL policy reachability
    ax = axes[2]
    for i, (_, row) in enumerate(df.iterrows()):
        model_name = row["model"]
        _, color, _ = _style(model_name)
        ax.plot([row["latent_min"], row["latent_max"]], [i, i],
                color=color, linewidth=3, alpha=0.7)
        ax.text(row["latent_max"] + 0.05, i, f"{model_name} d={row['latent_dim']}",
                fontsize=6, va="center")
    ax.axvline(-1.2, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="±1.2 lnorm boundary")
    ax.axvline(1.2, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Latent value range", fontsize=11)
    ax.set_title("Latent Range per Model\n(policy must cover this range)", fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_yticks([])

    fig.suptitle("Latent Space Diagnostics", fontsize=13)
    fig.tight_layout()
    path = output_dir / "latent_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Ranking table ─────────────────────────────────────────────────────────────


def print_ranking(df: pd.DataFrame) -> None:
    cols = ["model", "latent_dim", "val_r2", "val_mse", "latent_std_mean", "latent_std_min",
            "latent_min", "latent_max", "experiment"]
    cols = [c for c in cols if c in df.columns]
    ranked = df[cols].sort_values("val_r2", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 100)
    print("MODEL RANKING (sorted by val_r2)")
    print("=" * 100)
    print(ranked.to_string(index=True, float_format="{:.4f}".format))
    print("=" * 100)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.experiment_dirs}")
    df = load_all_results(
        args.experiment_dirs,
        models_filter=args.models,
        dims_filter=args.latent_dims,
    )

    if df.empty:
        print("No results found. Have the training runs finished?")
        return

    print(f"  Loaded {len(df)} runs across {df['model'].nunique()} models, "
          f"{df['latent_dim'].nunique()} latent dims, {df['experiment'].nunique()} experiments")

    # Save aggregated CSV — the persistent metadata store
    csv_path = output_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Metadata saved: {csv_path}")

    print_ranking(df)

    print("\nGenerating plots...")
    plot_r2_vs_latent_dim(df, output_dir)
    plot_arch_vs_reg(df, output_dir)

    for dim in sorted(df["latent_dim"].unique()):
        plot_per_muscle_heatmap(df, output_dir, latent_dim=dim)

    plot_latent_diagnostics(df, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
