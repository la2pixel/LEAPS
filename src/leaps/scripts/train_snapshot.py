"""Train HausdorferAE on per-frame muscle activation snapshots.

Loads EMG strides (N, 101, 11), maps them to bilateral 18-muscle activations via
EMGToMuscleMapper (with correct ~50% left-leg phase offset), flattens the time axis,
and trains HausdorferAE on the resulting (N*101, 18) snapshot dataset.

This is the first step of the LEAPS pipeline:
    Camargo EMG → HausdorferAE → latent z → frozen decoder in PPO

## Architecture (Hausdörfer et al. 2024, exact)
    Encoder: Linear(18 → 36) → Tanh → Linear(36 → 9)
    Decoder: Linear(9 → 36)  → Tanh → Linear(36 → 18)
    Loss:    MSE + lnorm_weight × Lnorm(z)
    Lnorm:   0 if |z|∞ < 0.8, else exp((z/1.2)^10) - 1.0

## Key diagnostics after training
    - val_r2: fraction of variance explained (target: ≥ 0.90 — snapshot task is easy)
    - latent_std_min > 0.1 (no collapsed dims)
    - pct_inside_0.8: fraction of z values with |z| < 0.8 (target: ≥ 90%)
      If < 80%, increase lnorm_weight; if dims are squeezed to 0, decrease it.

## Usage

    # Train on all conditions (recommended for general prior)
    python -m leaps.scripts.train_snapshot \\
        --data $LEAPS_EMG_H5 \\
        --output-dir experiments/hausdorfer_ae \\
        --epochs 300 --batch-size 4096 --wandb --wandb-project leaps

    # Train on treadmill 1.25 m/s only (RL-target condition)
    python -m leaps.scripts.train_snapshot \\
        --data $LEAPS_EMG_H5 \\
        --modes treadmill --conditions 1.25 \\
        --output-dir experiments/hausdorfer_ae_1.25 \\
        --epochs 300 --batch-size 4096

    # Quick sanity check (CPU, 5 epochs)
    python -m leaps.scripts.train_snapshot \\
        --data $LEAPS_EMG_H5 \\
        --epochs 5 --batch-size 512
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from leaps.data.metadata import LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides, stride_train_val_split
from leaps.models.legacy_models import HausdorferAE


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train HausdorferAE on per-frame muscle activation snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("data")
    g.add_argument("--data", default=LEAPS_H5_PATH,
                   help="HDF5 file from leaps preprocess (emg_activations_v2.h5).")
    g.add_argument("--subjects", nargs="+", default=None,
                   help="Subjects to include (default: all AB* in file).")
    g.add_argument("--modes", nargs="+", default=None,
                   choices=["treadmill", "levelground", "ramp", "stair"],
                   help="Locomotion modes to include (default: all).")
    g.add_argument("--conditions", nargs="+", default=None,
                   help="Condition labels to include, e.g. '1.25' (default: all).")
    g.add_argument("--val-fraction", type=float, default=0.15,
                   help="Fraction of subjects held out for validation.")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("model")
    g.add_argument("--latent-dim", type=int, default=9,
                   help="Latent dimension. Paper default: action_dim // 2 = 9.")
    g.add_argument("--action-dim", type=int, default=18,
                   help="Number of actuators (18 for gait10dof18musc).")
    g.add_argument("--lnorm-weight", type=float, default=1.0,
                   help="Weight on Lnorm penalty. Paper uses 1.0. "
                        "Increase if |z| > 0.8 too often; decrease if dims collapse.")

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=300,
                   help="Training epochs. 300 is sufficient — model has ~1k params.")
    g.add_argument("--batch-size", type=int, default=4096,
                   help="Mini-batch size. Large batches (4096) stabilise Lnorm gradient.")
    g.add_argument("--lr", type=float, default=3e-4,
                   help="Adam learning rate.")

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", default="experiments/hausdorfer_ae",
                   help="Directory for checkpoint and results.json.")
    g.add_argument("--checkpoint-name", default="hausdorfer_ae.pt",
                   help="Checkpoint filename inside output-dir.")
    g.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    g.add_argument("--wandb-project", default="leaps")
    g.add_argument("--wandb-entity", default=None)
    g.add_argument("--wandb-run-name", default=None,
                   help="W&B run name (default: auto-generated timestamp).")

    return p.parse_args(argv)


# ── EMG strides → snapshots ───────────────────────────────────────────────────


def strides_to_snapshots(
    strides: np.ndarray,
    default_activation: float = 0.05,
    offset_pct: float = 50.0,
) -> np.ndarray:
    """Vectorized map: (N, 101, 11) EMG strides → (N*101, 18) bilateral muscle snapshots.

    Fully numpy — no Python loops. ~100x faster than calling map_stride_with_phase_offset
    per stride (eliminates the 51k × 101 × 2 Python loop).

    Returns:
        (N*101, 18) float32 array, values in [0, 1].
    """
    N, T, _ = strides.shape
    offset_steps = int(round(offset_pct / 100.0 * T))

    # Left-leg EMG = right-leg EMG shifted forward by offset_steps in time
    # np.roll(..., -offset_steps, axis=1)[i,t,:] == strides[i,(t+offset_steps)%T,:]
    emg_r = strides                                    # (N, T, 11)
    emg_l = np.roll(strides, -offset_steps, axis=1)   # (N, T, 11)

    def _map_9(emg: np.ndarray) -> np.ndarray:
        """Vectorized EMG (N, T, 11) → 9-muscle activations (N, T, 9)."""
        act = np.full((*emg.shape[:2], 9), default_activation, dtype=np.float32)
        act[..., 0] = (emg[..., 6] + emg[..., 7]) / 2         # hamstrings
        act[..., 1] = emg[..., 6] * 0.5                        # bifemsh
        act[..., 2] = np.clip(0.25 + emg[..., 9] * 0.15, 0, 1)  # glut_max
        # act[..., 3] = default_activation                      # iliopsoas
        act[..., 4] = emg[..., 5]                               # rect_fem
        act[..., 5] = (emg[..., 3] + emg[..., 4]) / 2          # vasti
        act[..., 6] = emg[..., 0]                               # gastroc
        act[..., 7] = emg[..., 2]                               # soleus
        act[..., 8] = emg[..., 1]                               # tib_ant
        return np.clip(act, 0.0, 1.0)

    muscles_r = _map_9(emg_r)  # (N, T, 9)
    muscles_l = _map_9(emg_l)  # (N, T, 9)

    snapshots = np.concatenate([muscles_r, muscles_l], axis=-1)  # (N, T, 18)
    return snapshots.reshape(-1, 18).astype(np.float32)


# ── Evaluation helpers ────────────────────────────────────────────────────────


def reconstruction_r2(x: np.ndarray, x_hat: np.ndarray) -> float:
    x_f = x.ravel()
    x_hat_f = x_hat.ravel()
    ss_res = float(np.sum((x_f - x_hat_f) ** 2))
    ss_tot = float(np.sum((x_f - x_f.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-8)


def per_muscle_r2(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """R² per muscle channel. x shape: (N, 18)."""
    ss_res = ((x - x_hat) ** 2).sum(axis=0)
    ss_tot = ((x - x.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def latent_diagnostics(z: np.ndarray, lnorm_threshold: float = 0.8) -> dict[str, float]:
    """Check latent distribution for RL readiness.

    A well-trained HausdorferAE should have:
        - pct_inside_0.8 ≥ 90%  (Lnorm dead-zone: policy can move freely here)
        - latent_std_min > 0.1  (no completely collapsed dimensions)
        - latent range ≈ [-1.2, 1.2] (beyond 1.2 is exponentially penalised)
    """
    std_per_dim = z.std(axis=0)
    pct_inside = float(np.mean(np.abs(z) < lnorm_threshold) * 100)
    return {
        "latent_min": float(z.min()),
        "latent_max": float(z.max()),
        "latent_std_mean": float(std_per_dim.mean()),
        "latent_std_min": float(std_per_dim.min()),
        "latent_std_max": float(std_per_dim.max()),
        "pct_inside_0.8": pct_inside,
    }


def print_per_muscle_r2(ch_r2: np.ndarray) -> None:
    from leaps.envs.emg_mapping import MODEL_ACTUATORS
    print("  Per-muscle R²:")
    for name, r2 in zip(MODEL_ACTUATORS, ch_r2):
        bar = "█" * max(0, int(r2 * 20))
        print(f"    {name:<20s} {r2:+.3f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / args.checkpoint_name

    # ── Load strides ─────────────────────────────────────────────────────────

    print(f"Loading strides from {args.data}")
    if args.modes or args.conditions:
        print(f"  Filter: modes={args.modes}  conditions={args.conditions}")
    strides, metadata = load_strides(
        args.data,
        subjects=args.subjects,
        modes=args.modes,
        conditions=args.conditions,
        with_metadata=True,
    )
    n_strides, stride_len, n_emg_ch = strides.shape
    print(f"  Loaded: {n_strides} strides × {stride_len} frames × {n_emg_ch} EMG channels")
    mode_counts = {m: int((metadata["modes"] == m).sum()) for m in np.unique(metadata["modes"])}
    print(f"  Modes: {mode_counts}")

    # ── Subject-level train/val split ─────────────────────────────────────────

    s_train, s_val, meta_train, meta_val = stride_train_val_split(
        strides, metadata, val_fraction=args.val_fraction, seed=args.seed,
    )
    val_subjects = sorted(np.unique(meta_val["subjects"]))
    print(f"  Train: {s_train.shape[0]} strides ({len(np.unique(meta_train['subjects']))} subjects)")
    print(f"  Val:   {s_val.shape[0]} strides — held-out subjects: {val_subjects}")

    # ── Map EMG → bilateral muscle snapshots ──────────────────────────────────

    print("\nMapping EMG → bilateral muscle snapshots...")
    t0 = time.time()
    x_train = strides_to_snapshots(s_train)  # (N_train*101, 18)
    x_val   = strides_to_snapshots(s_val)    # (N_val*101,   18)
    print(f"  x_train: {x_train.shape}  x_val: {x_val.shape}  ({time.time()-t0:.1f}s)")
    print(f"  Value range: [{x_train.min():.3f}, {x_train.max():.3f}]")

    # ── W&B ──────────────────────────────────────────────────────────────────

    wb_run = None
    if args.wandb:
        try:
            import wandb as _wandb
            run_name = args.wandb_run_name or f"hausdorfer_ae_d{args.latent_dim}"
            wb_run = _wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group="hausdorfer_ae",
                name=run_name,
                config={
                    "model": "HausdorferAE",
                    "action_dim": args.action_dim,
                    "latent_dim": args.latent_dim,
                    "lnorm_weight": args.lnorm_weight,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "n_strides": n_strides,
                    "n_snapshots_train": len(x_train),
                    "n_snapshots_val": len(x_val),
                    "val_subjects": val_subjects,
                    "modes_filter": args.modes,
                    "conditions_filter": args.conditions,
                    "seed": args.seed,
                },
            )
        except ImportError:
            print("WARNING: wandb not installed — logging to stdout only.")

    def log_fn(metrics: dict[str, float], step: int) -> None:
        if wb_run is not None:
            wb_run.log(metrics, step=step)

    # ── Build and train model ─────────────────────────────────────────────────

    device_str = f"cuda ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}")

    print(f"\nBuilding HausdorferAE: action_dim={args.action_dim} latent_dim={args.latent_dim}"
          f" lnorm_weight={args.lnorm_weight}")
    model = HausdorferAE(
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
        lnorm_weight=args.lnorm_weight,
    )
    n_params = sum(p.numel() for p in model._module.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"\nTraining: {args.epochs} epochs  batch={args.batch_size}  lr={args.lr}")

    model.fit(
        x_train, x_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_fn=log_fn,
        ckpt_path=ckpt_path,
        min_delta=1e-5,
    )

    model.save(ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # ── Evaluate ─────────────────────────────────────────────────────────────

    print("\nEvaluating on val set...")
    x_val_hat = model.reconstruct(x_val)
    z_val = model.transform(x_val)  # (N_val_snapshots, latent_dim)

    val_r2 = reconstruction_r2(x_val, x_val_hat)
    val_mse = float(np.mean((x_val - x_val_hat) ** 2))
    ch_r2 = per_muscle_r2(x_val, x_val_hat)
    latent_stats = latent_diagnostics(z_val)

    print(f"  Val R²:  {val_r2:.4f}")
    print(f"  Val MSE: {val_mse:.6f}")
    print_per_muscle_r2(ch_r2)
    print(f"  Latent range:  [{latent_stats['latent_min']:.3f}, {latent_stats['latent_max']:.3f}]")
    print(f"  Latent std:    mean={latent_stats['latent_std_mean']:.3f}"
          f"  min={latent_stats['latent_std_min']:.3f}"
          f"  max={latent_stats['latent_std_max']:.3f}")
    print(f"  Inside |z|<0.8: {latent_stats['pct_inside_0.8']:.1f}%"
          f"  (target: ≥90% for Lnorm dead-zone to be effective)")

    # ── Warn if latent is problematic ────────────────────────────────────────

    if latent_stats["latent_std_min"] < 0.1:
        print(f"\n  WARNING: min latent std = {latent_stats['latent_std_min']:.3f} < 0.1."
              f" At least one dim is collapsed. Try reducing lnorm_weight.")
    if latent_stats["pct_inside_0.8"] < 80.0:
        print(f"\n  WARNING: only {latent_stats['pct_inside_0.8']:.1f}% of z values are"
              f" inside the |z|<0.8 Lnorm dead-zone. Increase lnorm_weight so the policy"
              f" can explore freely without hitting the Lnorm wall.")
    if val_r2 < 0.85:
        print(f"\n  WARNING: val R² = {val_r2:.4f} < 0.85. The snapshot AE task is simple"
              f" (1k params, ~5M training frames) — low R² suggests a training issue.")

    # ── Save results ─────────────────────────────────────────────────────────

    from leaps.envs.emg_mapping import MODEL_ACTUATORS
    results = {
        "model": "HausdorferAE",
        "action_dim": args.action_dim,
        "latent_dim": args.latent_dim,
        "lnorm_weight": args.lnorm_weight,
        "val_r2": val_r2,
        "val_mse": val_mse,
        "per_muscle_r2": {MODEL_ACTUATORS[i]: float(ch_r2[i]) for i in range(len(ch_r2))},
        **latent_stats,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_subjects": val_subjects,
            "modes_filter": args.modes,
            "conditions_filter": args.conditions,
            "seed": args.seed,
        },
        "data": {
            "n_strides": n_strides,
            "n_snapshots_train": int(len(x_train)),
            "n_snapshots_val": int(len(x_val)),
        },
        "checkpoint": str(ckpt_path),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if wb_run is not None:
        wb_run.summary.update({
            "val_r2": val_r2,
            "val_mse": val_mse,
            **{f"val_r2_{MODEL_ACTUATORS[i]}": float(ch_r2[i]) for i in range(len(ch_r2))},
            **latent_stats,
        })
        wb_run.finish()


if __name__ == "__main__":
    main()
