"""CLI entry point for training EMG dimensionality reduction models.

Trains PCA, NMF, Autoencoder, VAE, and Masked Autoencoder on preprocessed
EMG activation data. Sweeps over multiple latent dimensions and logs
everything to Weights & Biases for comparison.

## How it works

    1. Load preprocessed EMG data from HDF5 (/activations → N×11 matrix)
    2. Split into train/val (default 90/10)
    3. For each model × latent_dim combination:
       a. Create a separate wandb run (grouped for easy comparison)
       b. Train the model (sklearn fit or PyTorch training loop)
       c. Log per-epoch metrics (train_loss, val_mse, val_r2)
       d. Log final per-muscle R² bar chart + gait-aware eval metrics
       e. Save checkpoint to disk
    4. Print summary table and save results.json

## wandb integration

    Each model×latent_dim = one wandb "run". All runs in a single sweep
    share a "group" name (auto-generated timestamp or --wandb-group).

    In the wandb dashboard you can:
    - Compare runs side-by-side (val_mse, val_r2 across models/dims)
    - View per-muscle R² bar charts for each run
    - Group by model name to see latent_dim trends
    - Download results for offline analysis

    Epoch-level metrics logged by PyTorch models:
      - train_loss:     MSE loss on training set (every epoch)
      - val_mse:        validation MSE (every 10 epochs + last epoch)
      - val_r2:         validation R² (same schedule)
      - train_recon_loss, train_kl_loss:  VAE-specific (every epoch)

    Final evaluation metrics (logged once at end of training):
      - mse, r2:                    train set reconstruction quality
      - val_mse, val_r2:            val set reconstruction quality
      - stride_mse_mean/std:        per-stride consistency
      - peak_timing_err_mean:       gait event timing accuracy
      - peak_amp_ratio_mean:        muscle burst preservation
      - corr_matrix_dist:           synergy pattern preservation
      - gait_profile_corr_mean:     gait cycle shape preservation
      - latent_smoothness:          latent trajectory smoothness (for RL)
      - per-muscle R² bar chart:    visual breakdown by muscle

## Usage examples

    # Full sweep: 5 models × 5 latent dims = 25 wandb runs
    python -m leaps.scripts.train \\
        --data /fast/lsivakumar/data/processed/emg_activations.h5 \\
        --wandb --wandb-project leaps \\
        --output-dir experiments/sweep \\
        --epochs 200 --batch-size 512 --lr 1e-3 \\
        --latent-dims 2 3 4 5 6 \\
        --models PCA NMF AE VAE MAE

    # Quick single-model test (no wandb, 3 epochs)
    python -m leaps.scripts.train \\
        --data emg_activations.h5 \\
        --models AE --latent-dims 4 --epochs 3

    # On cluster without internet: use offline mode, sync later
    WANDB_MODE=offline python -m leaps.scripts.train \\
        --data emg_activations.h5 --wandb
    # Then when you have internet:
    wandb sync runs/

    # Monitor live on the dashboard:
    #   https://wandb.ai/la2pixel-university-of-t-bingen/leaps
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from leaps.data.dataset import compute_norm_params, load_activations, train_val_split
from leaps.evaluation.metrics import (
    compute_eval_metrics,
    compute_train_metrics,
    per_muscle_r2,
)
from leaps.training.trainer import (
    PYTORCH_MODELS,
    SKLEARN_MODELS,
    build_model,
    train_pytorch_model,
    train_sklearn_model,
)

# All available model names, sorted alphabetically for the CLI help text
ALL_MODELS = sorted(SKLEARN_MODELS | PYTORCH_MODELS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EMG dimensionality reduction models with latent-dim sweep."
    )
    # --- Required ---
    parser.add_argument(
        "--data", required=True,
        help="Path to HDF5 file with /activations dataset "
             "(output of leaps-preprocess).",
    )
    # --- Output ---
    parser.add_argument(
        "--output-dir", default="experiments/sweep",
        help="Directory for checkpoints and results.json (default: experiments/sweep).",
    )
    # --- Sweep grid ---
    parser.add_argument(
        "--latent-dims", type=int, nargs="+", default=[2, 3, 4, 5, 6],
        help="Latent dimensions to sweep (default: 2 3 4 5 6). "
             "Each dim is trained with every model.",
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
        help="Models to train (default: all). "
             "PCA/NMF are instant; AE/VAE/MAE need GPU for speed.",
    )
    # --- PyTorch training hyperparameters ---
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Training epochs for PyTorch models (default: 200). "
             "PCA/NMF ignore this.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096,
        help="Mini-batch size for PyTorch models (default: 4096). "
             "With 5M snapshots, small batches = slow training.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for Adam optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of data held out for validation (default: 0.1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # --- Model-specific ---
    parser.add_argument(
        "--beta", type=float, default=0.05,
        help="VAE KL divergence weight (default: 0.05). "
             "Lower values = weaker regularization, better reconstruction.",
    )
    parser.add_argument(
        "--mask-ratio", type=float, default=0.5,
        help="MAE input masking ratio (default: 0.5). "
             "Fraction of input channels randomly zeroed during training.",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None,
        help="Hidden layer size for AE/VAE/MAE encoder and decoder. "
             "Default: 2×latent_dim (matching Hausdörfer et al. 2024). "
             "Set to e.g. 64 for ablation with larger architectures.",
    )
    parser.add_argument(
        "--mmd-weight", type=float, default=10.0,
        help="WAE MMD penalty weight (default: 10.0).",
    )
    # --- wandb ---
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging. Each model×dim = one run. "
             "View at https://wandb.ai/la2pixel-university-of-t-bingen/leaps",
    )
    parser.add_argument(
        "--wandb-project", default="leaps",
        help="W&B project name (default: leaps).",
    )
    parser.add_argument(
        "--wandb-group", default=None,
        help="W&B run group name. All runs in this sweep share a group. "
             "Default: auto-generated timestamp like sweep_20260216_143000.",
    )
    return parser.parse_args(argv)


def _print_summary(results: list[dict]) -> None:
    """Print a comparison table of all model × latent_dim results."""
    header = f"{'Model':<20} {'Dim':>4} {'MSE':>10} {'R2':>10}"
    val_header = f"{'val_MSE':>10} {'val_R2':>10}"
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{header} {val_header}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: (x["model"], x["latent_dim"])):
        m = r["metrics"]
        line = (
            f"{r['model']:<20} {r['latent_dim']:>4}"
            f" {m.get('mse', float('nan')):>10.6f}"
            f" {m.get('r2', float('nan')):>10.4f}"
        )
        if "val_mse" in m:
            line += (
                f" {m['val_mse']:>10.6f}"
                f" {m['val_r2']:>10.4f}"
            )
        print(line)
    print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 0: Set all random seeds ONCE before anything else
    # -------------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -------------------------------------------------------------------------
    # Step 1: Load preprocessed EMG activations from HDF5
    # -------------------------------------------------------------------------
    # The HDF5 file was created by leaps-preprocess. It contains:
    #   /activations  — (N, 11) float64, all subjects × all modes pooled
    #   /stride_modes — mode labels per stride (not used here, used for eval)
    print(f"Loading data from {args.data}")
    activations = load_activations(args.data)
    print(f"  Raw shape: {activations.shape}")
    print(f"  Raw range: [{activations.min():.3f}, {activations.max():.3f}]")

    # Split BEFORE normalization — norm params come from train set only
    x_train_raw, x_val_raw = train_val_split(
        activations, val_fraction=args.val_fraction, seed=args.seed,
    )

    # Clip to per-channel 99th percentile and rescale to [0, 1].
    # Computed from training data only to avoid data leakage.
    norm_params = compute_norm_params(x_train_raw, lower_pct=0.0, upper_pct=99.0)
    x_train = norm_params.transform(x_train_raw).astype(np.float32)
    x_val = norm_params.transform(x_val_raw).astype(np.float32)

    # Save norm params alongside checkpoints for inference/RL deployment
    norm_path = output_dir / "norm_params.npz"
    norm_params.save(str(norm_path))

    print(f"  Train: {x_train.shape}, Val: {x_val.shape}")
    print(f"  Normalized range: [{x_train.min():.3f}, {x_train.max():.3f}]")

    # -------------------------------------------------------------------------
    # Step 2: Initialize wandb (optional)
    # -------------------------------------------------------------------------
    # wandb tracks experiment metrics in the cloud. Each model×latent_dim
    # gets its own "run", and all runs in this sweep share a "group" so
    # you can compare them in the dashboard.
    #
    # If wandb is not installed or --wandb is not passed, training still
    # works — metrics are printed to stdout and saved to results.json.
    wandb = None
    if args.wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            print("WARNING: wandb not installed, falling back to stdout only.")
            print("  Install with: pip install wandb")

    # Group name ties all runs in this sweep together in the wandb UI.
    # You can filter by group to see only this experiment's runs.
    group_name = args.wandb_group or f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"

    # Config shared across all runs — logged to wandb so you can reproduce.
    shared_config = {
        "data": args.data,
        "n_samples": activations.shape[0],
        "n_channels": activations.shape[1],
        "n_train": x_train.shape[0],
        "n_val": x_val.shape[0],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "beta": args.beta,
        "mask_ratio": args.mask_ratio,
        "hidden_dim": args.hidden_dim,
    }

    results = []
    n_channels = activations.shape[1]

    # -------------------------------------------------------------------------
    # Step 3: Training loop — sweep over models × latent dims
    # -------------------------------------------------------------------------
    # Outer loop: models (PCA, NMF, AE, VAE, MAE)
    # Inner loop: latent dimensions (e.g. 2, 3, 4, 5, 6)
    #
    # sklearn models (PCA, NMF): fit instantly, no epochs needed.
    # PyTorch models (AE, VAE, MAE): train for --epochs with Adam optimizer.
    #   These benefit from GPU — use --batch-size 512+ with a GPU node.
    for model_name in args.models:
        for latent_dim in args.latent_dims:
            run_name = f"{model_name}_d{latent_dim}"
            print(f"\n--- {run_name} ---")

            # Start a new wandb run for this model×dim combination.
            # reinit=True allows multiple wandb.init() calls in one process.
            wb_run = None
            if wandb is not None:
                wb_run = wandb.init(
                    project=args.wandb_project,
                    group=group_name,
                    name=run_name,
                    config={
                        **shared_config,
                        "model": model_name,
                        "latent_dim": latent_dim,
                    },
                    reinit=True,
                )

            # Build the model with the right architecture
            extra = {}
            if model_name in PYTORCH_MODELS and args.hidden_dim is not None:
                extra["hidden_dim"] = args.hidden_dim
            if model_name == "VAE":
                extra["beta"] = args.beta
            elif model_name == "MAE":
                extra["mask_ratio"] = args.mask_ratio
            elif model_name == "WAE":
                extra["mmd_weight"] = args.mmd_weight

            model = build_model(model_name, latent_dim, n_channels=n_channels, **extra)
            ckpt_dir = output_dir / "checkpoints"

            # Create a logging callback that sends metrics to wandb.
            # This is called by the PyTorch training loop every epoch:
            #   log_fn({"train_loss": 0.01, "val_mse": 0.02, ...}, step=epoch)
            # The wandb dashboard then shows live training curves.
            def make_log_fn(run):
                def log_fn(metrics: dict[str, float], step: int) -> None:
                    if run is not None:
                        run.log(metrics, step=step)
                return log_fn

            # --- Train ---
            if model_name in SKLEARN_MODELS:
                # PCA and NMF: single fit() call, returns metrics dict
                metrics = train_sklearn_model(model, x_train, x_val, save_dir=ckpt_dir)
            else:
                # AE, VAE, MAE: PyTorch training loop with epoch-level logging
                # Metrics are logged to wandb every epoch via log_fn callback.
                # Val metrics logged every 10 epochs for efficiency.
                metrics = train_pytorch_model(
                    model,
                    x_train,
                    x_val,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    save_dir=ckpt_dir,
                    log_fn=make_log_fn(wb_run),
                )

            # --- Evaluate: comprehensive gait-aware metrics ---
            x_val_hat = model.reconstruct(x_val)
            latent_val = model.transform(x_val)
            eval_metrics = compute_eval_metrics(
                x_val, x_val_hat, latent=latent_val, stride_len=101,
            )
            ch_r2 = per_muscle_r2(x_val, x_val_hat)

            # --- Log final summary to wandb ---
            if wb_run is not None:
                from leaps.data.metadata import EMG_CHANNELS

                wb_run.summary.update(metrics)
                wb_run.summary.update({f"eval_{k}": v for k, v in eval_metrics.items()})

                # Per-muscle R² bar chart
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

            # Print key results to stdout
            print(f"  Per-muscle R²: {np.array2string(ch_r2, precision=3)}")
            key_metrics = ["mse", "r2", "stride_mse_mean", "stride_mse_std",
                           "peak_timing_err_mean", "peak_amp_ratio_mean",
                           "corr_matrix_dist", "gait_profile_corr_mean",
                           "latent_smoothness"]
            for k in key_metrics:
                if k in eval_metrics:
                    print(f"  {k}: {eval_metrics[k]:.6f}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.6f}")

            results.append(
                {"model": model_name, "latent_dim": latent_dim, "metrics": metrics}
            )

    # -------------------------------------------------------------------------
    # Step 4: Summary and save
    # -------------------------------------------------------------------------
    _print_summary(results)

    # Save results as JSON for programmatic access / plotting later
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
