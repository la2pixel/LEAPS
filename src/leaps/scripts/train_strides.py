"""Train stride-level EMG representation models.

Compresses full gait cycles (101, 11) to a latent vector; the frozen decoder
is used in RL as the action prior.

Usage:
    leaps train-strides --models StridePCA StrideNMF --latent-dims 4 6 8 10
    leaps train-strides --models StrideFlatAE --latent-dims 4 6 8 10 --wandb
    leaps train-strides --models StrideFlatVAE --latent-dims 8 --beta 0.01 --wandb
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from leaps.data.stride_dataset import load_strides, stride_train_val_split
from leaps.data.metadata import EMG_CHANNELS, LEAPS_H5_PATH
from leaps.training.stride_trainer import STRIDE_SKLEARN_MODELS, STRIDE_PYTORCH_MODELS, build_stride_model

ALL_MODELS = sorted(STRIDE_SKLEARN_MODELS | STRIDE_PYTORCH_MODELS)


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and sweep stride-level EMG representation models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    g = p.add_argument_group("data")
    g.add_argument("--data", default=LEAPS_H5_PATH,
                   help="HDF5 file from leaps preprocess (emg_activations_v2.h5).")
    g.add_argument("--subjects", nargs="+", default=None,
                   help="Subjects to include (default: all AB* in file).")
    g.add_argument("--modes", nargs="+", default=None,
                   choices=["treadmill", "levelground", "ramp", "stair"],
                   help="Locomotion modes to include (default: all).")
    g.add_argument("--conditions", nargs="+", default=None,
                   help="Condition labels to include, e.g. '1.20' 'normal' (default: all).")
    g.add_argument("--val-fraction", type=float, default=0.15,
                   help="Fraction of subjects held out for validation.")
    g.add_argument("--seed", type=int, default=42)

    # Sweep
    g = p.add_argument_group("sweep")
    g.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                   help="Models to train.")
    g.add_argument("--latent-dims", type=int, nargs="+", default=[4, 6, 8, 16],
                   help="Latent dimensions to sweep. 4–8 for tight RL action space; "
                        "16 as an upper-bound reference (how much does extra capacity help?).")

    # Training
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=200,
                   help="Max epochs. Early stopping fires after 75 epochs without val improvement.")
    g.add_argument("--batch-size", type=int, default=256,
                   help="Mini-batch size. 256 is a good default for ~50k strides on GPU.")
    g.add_argument("--lr", type=float, default=1e-3,
                   help="Initial learning rate for Adam. ReduceLROnPlateau halves it on plateau.")

    # Model hyperparameters (fixed per run — do not sweep until you have a baseline)
    g = p.add_argument_group("model hyperparameters")
    g.add_argument("--beta", type=float, default=0.01,
                   help="StrideFlatVAE KL weight. Sweep {0.001, 0.01, 0.05}.")
    g.add_argument("--free-bits", type=float, default=0.5,
                   help="StrideFlatVAE minimum KL per latent dim (nats). Prevents posterior "
                        "collapse. Keep fixed at 0.5 — only sweep beta.")
    g.add_argument("--lnorm-weight", type=float, default=0.01,
                   help="Soft latent norm penalty weight (FlatAE/FlatVAE). Keeps z near [-1.2, 1.2].")

    # Output
    g = p.add_argument_group("output")
    g.add_argument("--output-dir", default="experiments/stride_sweep",
                   help="Directory for checkpoints and results.json.")
    g.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    g.add_argument("--wandb-project", default="leaps")
    g.add_argument("--wandb-entity", default=None,
                   help="W&B entity (username or team). If None, uses your default entity.")
    g.add_argument("--wandb-group", default=None,
                   help="W&B run group (default: auto-generated timestamp).")

    return p.parse_args(argv)


# ── Evaluation helpers ────────────────────────────────────────────────────────


def per_muscle_r2(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """R² per EMG channel on flattened data. Shape: (n_channels,)."""
    x_f = x.reshape(-1, x.shape[-1])
    x_hat_f = x_hat.reshape(-1, x_hat.shape[-1])
    ss_res = ((x_f - x_hat_f) ** 2).sum(axis=0)
    ss_tot = ((x_f - x_f.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def latent_diagnostics(z: np.ndarray) -> dict[str, float]:
    """Check latent distribution — critical before plugging into RL.

    A well-behaved latent for RL should have:
      - std per dim near 1.0 (well-used dimensions; std << 1 = collapsed)
      - range within roughly [-3, 3] (RL policy output is typically tanh-bounded)
      - min_std > 0.1 (no completely collapsed dimensions)
    """
    std_per_dim = z.std(axis=0)    # (latent_dim,)
    return {
        "latent_min": float(z.min()),
        "latent_max": float(z.max()),
        "latent_std_mean": float(std_per_dim.mean()),
        "latent_std_min": float(std_per_dim.min()),   # lowest: collapsed dim indicator
        "latent_std_max": float(std_per_dim.max()),
    }


def reconstruction_r2(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Global R² across all channels and timepoints."""
    x_f = x.reshape(-1)
    x_hat_f = x_hat.reshape(-1)
    ss_res = ((x_f - x_hat_f) ** 2).sum()
    ss_tot = ((x_f - x_f.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def reconstruction_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean((x - x_hat) ** 2))


def print_per_muscle_r2(ch_r2: np.ndarray) -> None:
    print("  Per-muscle R²:")
    for i, (name, r2) in enumerate(zip(EMG_CHANNELS, ch_r2)):
        bar = "█" * max(0, int(r2 * 20))
        print(f"    {name:<30s} {r2:+.3f}  {bar}")


# ── Summary table ─────────────────────────────────────────────────────────────


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Model':<20} {'Dim':>4} {'Val R²':>8} {'Val MSE':>10} {'LatStd':>8} {'LatMin':>8} {'LatMax':>8}"
    print(header)
    print("-" * 80)
    for r in sorted(results, key=lambda x: (-x["val_r2"], x["model"], x["latent_dim"])):
        print(
            f"{r['model']:<20} {r['latent_dim']:>4}"
            f" {r['val_r2']:>8.4f} {r['val_mse']:>10.6f}"
            f" {r['latent_std_mean']:>8.3f} {r['latent_min']:>8.3f} {r['latent_max']:>8.3f}"
        )
    print("=" * 80)

    # Flag potential issues
    for r in results:
        if r["latent_std_min"] < 0.1:
            print(f"  WARNING: {r['model']} d={r['latent_dim']} has a collapsed latent "
                  f"dimension (min_std={r['latent_std_min']:.3f}). "
                  f"Consider reducing lnorm_weight or latent_dim.")
        if abs(r["latent_max"]) > 5 or abs(r["latent_min"]) > 5:
            print(f"  WARNING: {r['model']} d={r['latent_dim']} has extreme latent values "
                  f"[{r['latent_min']:.1f}, {r['latent_max']:.1f}]. "
                  f"RL policy may struggle to match this range.")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and split data ──────────────────────────────────────────────────

    print(f"Loading data from {args.data}")
    if args.modes or args.conditions:
        print(f"  Filtering: modes={args.modes}  conditions={args.conditions}")
    strides, metadata = load_strides(
        args.data,
        subjects=args.subjects,
        modes=args.modes,
        conditions=args.conditions,
        with_metadata=True,
    )
    n_strides, stride_len, n_channels = strides.shape
    print(f"  Loaded: {n_strides} strides, {stride_len} timepoints, {n_channels} channels")
    print(f"  Subjects: {sorted(np.unique(metadata['subjects']))}")
    print(f"  Modes: { {m: int((metadata['modes']==m).sum()) for m in np.unique(metadata['modes'])} }")

    x_train, x_val, meta_train, meta_val = stride_train_val_split(
        strides, metadata, val_fraction=args.val_fraction, seed=args.seed,
    )
    val_subjects = sorted(np.unique(meta_val["subjects"]))
    print(f"  Train: {x_train.shape[0]} strides ({len(np.unique(meta_train['subjects']))} subjects)")
    print(f"  Val:   {x_val.shape[0]} strides — held-out subjects: {val_subjects}")

    # Warn if training set is very small (can happen with aggressive filtering)
    if x_train.shape[0] < 500:
        print(f"  WARNING: Only {x_train.shape[0]} training strides — consider relaxing filters.")

    # ── W&B setup ────────────────────────────────────────────────────────────

    wandb = None
    if args.wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            print("WARNING: wandb not installed. Logging to stdout only.")

    # Derive a stable group name from the model(s) being run so related runs cluster together.
    # Single-model runs: StrideFlatAE → "flatae", StrideFlatVAE → "flatvae", etc.
    # Multi-model sweeps: "stride_sweep"
    if args.wandb_group:
        group_name = args.wandb_group
    elif len(args.models) == 1:
        group_name = args.models[0].lower().replace("stride", "")
    else:
        group_name = "stride_sweep"
    shared_config = {
        "n_strides": n_strides, "n_train": x_train.shape[0], "n_val": x_val.shape[0],
        "val_subjects": val_subjects, "stride_len": stride_len, "n_channels": n_channels,
        "modes_filter": args.modes, "conditions_filter": args.conditions,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "beta": args.beta, "lnorm_weight": args.lnorm_weight,
    }

    # ── Sweep ─────────────────────────────────────────────────────────────────

    n_runs = len(args.models) * len(args.latent_dims)
    print(f"\nStarting sweep: {len(args.models)} models × {len(args.latent_dims)} latent dims = {n_runs} runs")
    print(f"Models: {args.models}")
    print(f"Latent dims: {args.latent_dims}\n")

    results = []

    for run_idx, model_name in enumerate(args.models):
        for latent_dim in args.latent_dims:
            if model_name == "StrideFlatVAE":
                run_name = f"{model_name}_d{latent_dim}_b{args.beta:.3f}"
            else:
                run_name = f"{model_name}_d{latent_dim}"
            print(f"\n[{len(results)+1}/{n_runs}] {run_name}")
            print("-" * 50)

            extra: dict = {}
            if model_name == "StrideFlatVAE":
                extra["beta"] = args.beta
                extra["free_bits"] = args.free_bits
            if model_name in STRIDE_PYTORCH_MODELS:
                extra["lnorm_weight"] = args.lnorm_weight

            model = build_stride_model(
                model_name, latent_dim,
                n_channels=n_channels, stride_len=stride_len,
                **extra,
            )

            # W&B run
            wb_run = None
            if wandb is not None:
                wb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,   # None = use default entity from `wandb login`
                    group=group_name,
                    name=run_name,
                    config={**shared_config, "model": model_name, "latent_dim": latent_dim, **extra},
                    reinit="finish_previous",
                )

            def log_fn(metrics: dict[str, float], step: int) -> None:
                if wb_run is not None:
                    wb_run.log(metrics, step=step)

            # Train
            ckpt_ext = ".pkl" if model_name in STRIDE_SKLEARN_MODELS else ".pt"
            ckpt_path = ckpt_dir / f"{run_name}{ckpt_ext}"

            if model_name in STRIDE_SKLEARN_MODELS:
                model.fit(x_train, x_val)
            else:
                model.fit(
                    x_train, x_val,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    log_fn=log_fn,
                )

            # Save checkpoint — this is the best-val-loss state (early stopping restores it)
            model.save(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            # ── Evaluate on val set ──────────────────────────────────────────

            x_val_hat = model.reconstruct(x_val)
            z_val = model.transform(x_val)  # (n_val_strides, latent_dim) — one z per stride

            val_r2 = reconstruction_r2(x_val, x_val_hat)
            val_mse = reconstruction_mse(x_val, x_val_hat)
            ch_r2 = per_muscle_r2(x_val, x_val_hat)
            latent_stats = latent_diagnostics(z_val)

            print(f"  Val R²:  {val_r2:.4f}")
            print(f"  Val MSE: {val_mse:.6f}")
            print_per_muscle_r2(ch_r2)
            print(f"  Latent:  std_mean={latent_stats['latent_std_mean']:.3f}"
                  f"  std_min={latent_stats['latent_std_min']:.3f}"
                  f"  range=[{latent_stats['latent_min']:.2f}, {latent_stats['latent_max']:.2f}]")

            if wb_run is not None:
                wb_run.summary.update({
                    "val_r2": val_r2, "val_mse": val_mse,
                    **{f"val_r2_{EMG_CHANNELS[i]}": float(ch_r2[i]) for i in range(len(ch_r2))},
                    **latent_stats,
                })
                # Per-muscle bar chart
                table = wandb.Table(
                    columns=["muscle", "r2"],
                    data=[[EMG_CHANNELS[i], float(ch_r2[i])] for i in range(len(ch_r2))],
                )
                wb_run.log({"per_muscle_r2": wandb.plot.bar(table, "muscle", "r2",
                                                             title=f"Per-Muscle R² — {run_name}")})
                wb_run.finish()

            run_result = {
                "model": model_name,
                "latent_dim": latent_dim,
                "val_r2": val_r2,
                "val_mse": val_mse,
                "per_muscle_r2": {EMG_CHANNELS[i]: float(ch_r2[i]) for i in range(len(ch_r2))},
                **latent_stats,
                # Hyperparameters — everything needed to reproduce this checkpoint
                "hyperparameters": {
                    "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                    "lnorm_weight": args.lnorm_weight,
                    "modes_filter": args.modes, "conditions_filter": args.conditions,
                    "val_subjects": val_subjects,
                    **extra,
                },
                "checkpoint": str(ckpt_path),
            }
            # Write per-run metadata next to the checkpoint
            with open(ckpt_path.with_suffix(".json"), "w") as f:
                json.dump(run_result, f, indent=2)

            results.append(run_result)

    # ── Final summary ─────────────────────────────────────────────────────────

    print_summary(results)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
