"""CLI entry point for training stride-level EMG dimensionality reduction models.

Trains models on full stride sequences (101, 11) instead of individual snapshots
(11,). Each model compresses a complete gait cycle into a latent_dim-dimensional
vector. Reports gait-aware metrics tailored for downstream RL control.

The key difference from train.py: models see temporal structure within each stride,
which matters for learning smooth latent action representations for a muscle
humanoid controller.

## Usage examples

    # Full sweep: 5 stride models × 4 latent dims = 20 wandb runs
    python -m leaps.scripts.train_strides \\
        --data /fast/lsivakumar/data/processed/emg_activations.h5 \\
        --wandb --wandb-project leaps \\
        --output-dir experiments/stride_sweep \\
        --epochs 200 --batch-size 64 \\
        --latent-dims 2 4 6 8 \\
        --models StridePCA StrideNMF StrideAE StrideVAE StrideMAE

    # Quick single-model test
    python -m leaps.scripts.train_strides \\
        --data emg_activations.h5 \\
        --models StrideAE --latent-dims 4 --epochs 3
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from leaps.data.stride_dataset import load_strides, stride_train_val_split
from leaps.evaluation.metrics import compute_eval_metrics, per_muscle_r2
from leaps.training.stride_trainer import (
    STRIDE_PYTORCH_MODELS,
    STRIDE_SKLEARN_MODELS,
    build_stride_model,
    train_stride_pytorch_model,
    train_stride_sklearn_model,
)

ALL_STRIDE_MODELS = sorted(STRIDE_SKLEARN_MODELS | STRIDE_PYTORCH_MODELS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train stride-level EMG dimensionality reduction models."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to HDF5 file with per-subject /AB*/strides datasets.",
    )
    parser.add_argument(
        "--output-dir", default="experiments/stride_sweep",
        help="Directory for checkpoints and results.json.",
    )
    parser.add_argument(
        "--latent-dims", type=int, nargs="+", default=[6, 8, 9, 10, 12, 16],
        help="Latent dimensions to sweep (default: 6 8 9 10 12 16).",
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_STRIDE_MODELS, choices=ALL_STRIDE_MODELS,
        help="Stride models to train (default: all).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.05, help="StrideVAE KL weight (default: 0.05).")
    parser.add_argument("--mask-ratio", type=float, default=0.5, help="StrideMAE mask ratio.")
    parser.add_argument("--mmd-weight", type=float, default=10.0, help="StrideWAE MMD weight.")
    parser.add_argument("--cnmf-alpha", type=float, default=0.1, help="CNMF smoothness weight.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="leaps")
    parser.add_argument("--wandb-group", default=None)
    return parser.parse_args(argv)


def _print_summary(results: list[dict]) -> None:
    """Print a comparison table of all stride model results."""
    print("\n" + "=" * 90)
    print("STRIDE-LEVEL SUMMARY")
    print("=" * 90)
    header = (
        f"{'Model':<25} {'Dim':>4} {'MSE':>10} {'R2':>8}"
        f" {'TimErr':>8} {'AmpR':>8} {'CorrD':>8} {'ProfC':>8} {'Smooth':>8}"
    )
    print(header)
    print("-" * 90)
    for r in sorted(results, key=lambda x: (x["model"], x["latent_dim"])):
        e = r.get("eval_metrics", {})
        line = (
            f"{r['model']:<25} {r['latent_dim']:>4}"
            f" {e.get('mse', float('nan')):>10.6f}"
            f" {e.get('r2', float('nan')):>8.4f}"
            f" {e.get('peak_timing_err_mean', float('nan')):>8.2f}"
            f" {e.get('peak_amp_ratio_mean', float('nan')):>8.3f}"
            f" {e.get('corr_matrix_dist', float('nan')):>8.4f}"
            f" {e.get('gait_profile_corr_mean', float('nan')):>8.4f}"
            f" {e.get('latent_smoothness', float('nan')):>8.4f}"
        )
        print(line)
    print("=" * 90)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load stride data with metadata
    print(f"Loading stride data from {args.data}")
    strides, metadata = load_strides(args.data, with_metadata=True)
    print(f"  Strides shape: {strides.shape}")  # (n_strides, 101, 11)
    print(f"  Subjects: {np.unique(metadata['subjects'])}")
    print(f"  Modes: {np.unique(metadata['modes'])}")

    # Subject-aware train/val split
    x_train, x_val, meta_train, meta_val = stride_train_val_split(
        strides, metadata, val_fraction=args.val_fraction, seed=args.seed,
    )
    val_subjects = np.unique(meta_val["subjects"])
    print(f"  Train: {x_train.shape} ({len(np.unique(meta_train['subjects']))} subjects)")
    print(f"  Val:   {x_val.shape} (held-out subjects: {val_subjects})")

    # wandb setup
    wandb = None
    if args.wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            print("WARNING: wandb not installed, falling back to stdout only.")

    group_name = args.wandb_group or f"stride_{time.strftime('%Y%m%d_%H%M%S')}"
    shared_config = {
        "data": args.data,
        "n_strides": strides.shape[0],
        "stride_len": strides.shape[1],
        "n_channels": strides.shape[2],
        "n_train": x_train.shape[0],
        "n_val": x_val.shape[0],
        "val_subjects": val_subjects.tolist(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }

    results = []
    n_channels = strides.shape[2]
    stride_len = strides.shape[1]

    # Training loop
    for model_name in args.models:
        for latent_dim in args.latent_dims:
            run_name = f"{model_name}_d{latent_dim}"
            print(f"\n--- {run_name} ---")

            wb_run = None
            if wandb is not None:
                wb_run = wandb.init(
                    project=args.wandb_project,
                    group=group_name,
                    name=run_name,
                    config={**shared_config, "model": model_name, "latent_dim": latent_dim},
                    reinit="finish_previous",
                )

            extra = {}
            if model_name == "StrideVAE":
                extra["beta"] = args.beta
            elif model_name == "StrideMAE":
                extra["mask_ratio"] = args.mask_ratio
            elif model_name == "StrideWAE":
                extra["mmd_weight"] = args.mmd_weight
            elif model_name == "StrideCNMF":
                extra["alpha"] = args.cnmf_alpha

            model = build_stride_model(
                model_name, latent_dim, n_channels=n_channels, stride_len=stride_len, **extra
            )
            ckpt_dir = output_dir / "checkpoints"

            def make_log_fn(run):
                def log_fn(metrics: dict[str, float], step: int) -> None:
                    if run is not None:
                        run.log(metrics, step=step)
                return log_fn

            # Train
            if model_name in STRIDE_SKLEARN_MODELS:
                train_metrics = train_stride_sklearn_model(
                    model, x_train, x_val, save_dir=ckpt_dir,
                )
            else:
                train_metrics = train_stride_pytorch_model(
                    model, x_train, x_val,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    save_dir=ckpt_dir,
                    log_fn=make_log_fn(wb_run),
                )

            # Comprehensive evaluation on val set
            x_val_hat = model.reconstruct(x_val)
            latent_val = model.transform(x_val)

            # Flatten for metrics (they expect 2D)
            x_val_flat = x_val.reshape(-1, n_channels)
            x_val_hat_flat = x_val_hat.reshape(-1, n_channels)
            # Repeat latent per timepoint for smoothness calculation
            latent_val_expanded = np.repeat(latent_val, stride_len, axis=0)

            eval_metrics = compute_eval_metrics(
                x_val_flat, x_val_hat_flat,
                latent=latent_val_expanded,
                modes=meta_val.get("modes"),
                stride_len=stride_len,
            )

            # For stride models, also compute latent smoothness directly
            # by encoding each timepoint separately (if we had per-timepoint latents).
            # But since stride models produce one latent per stride, smoothness
            # across strides is less meaningful. We'll skip this for now.

            # Log to wandb
            if wb_run is not None:
                from leaps.data.metadata import EMG_CHANNELS

                wb_run.summary.update(train_metrics)
                wb_run.summary.update({f"eval_{k}": v for k, v in eval_metrics.items()})

                ch_r2 = per_muscle_r2(x_val_flat, x_val_hat_flat)
                ch_r2_table = wandb.Table(
                    columns=["muscle", "r2"],
                    data=[[EMG_CHANNELS[i], float(ch_r2[i])] for i in range(len(ch_r2))],
                )
                wb_run.log({
                    "per_muscle_r2": wandb.plot.bar(
                        ch_r2_table, "muscle", "r2",
                        title=f"Per-Muscle R² — {run_name}",
                    )
                })
                wb_run.finish()

            # Print key results
            key_metrics = [
                "mse", "r2", "stride_mse_mean", "stride_mse_std",
                "peak_timing_err_mean", "peak_amp_ratio_mean",
                "corr_matrix_dist", "gait_profile_corr_mean",
            ]
            for k in key_metrics:
                if k in eval_metrics:
                    print(f"  {k}: {eval_metrics[k]:.6f}")

            results.append({
                "model": model_name,
                "latent_dim": latent_dim,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
            })

    _print_summary(results)

    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
