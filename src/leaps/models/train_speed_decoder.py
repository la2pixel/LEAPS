"""Train and freeze a speed-conditioned synergy decoder for RL.

Motivation: the existing decoder_k*/ pools all 4 modes and every treadmill
speed into one basis. Real muscle coordination at 0.5 m/s and 1.4 m/s isn't
just a scaled version of the same pattern -- pooling forces one latent code
to represent both. Phase-conditioning (see get_synergies.py's AE_phase
column) has no online equivalent during an RL rollout, since the sim has no
ground-truth gait-cycle-percent signal -- but speed does: every sconewalk_*
env targets one fixed, known speed (h0918 = 1.2 m/s), so a speed-conditioned
decoder can genuinely be deployed by passing that constant every step. With
sconewalk_*_gaussianVel_vcond-v1 the target speed varies per episode instead
of being fixed, and the wrapper passes the live target_vel every step
(speed_cond="env" in latent_env.py).

Restricted to treadmill-mode strides only (the only mode with a real speed
label -- see leaps/data/stride_dataset.py, NaN for levelground/ramp/stair).

Defaults match decoder_k6_AB06_corrected: k=6, subject AB06 only, corrected
latent-norm recipe (threshold 0.6, scale 0.9). --save-plain also freezes the
speed-blind twin trained on the same treadmill-only pool -- the honest
"static prior" decoder for the uncond_real RL arm, so "no speed
conditioning" isn't confounded with "different training data".
"""

import argparse
import os

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import torch

from leaps.data.metadata import LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides
from leaps.synergy_common import CACHE_DIR, r2, train_ae

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description="Train + freeze a speed-conditioned synergy decoder for RL.")
parser.add_argument("--k", type=int, default=6, help="latent dim (default 6, matching decoder_k6_AB06_corrected)")
parser.add_argument("--subjects", nargs="+", default=["AB06"],
                    help="Camargo subject IDs to pool (default: AB06 only, matching the single-subject decoder)")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--threshold", type=float, default=0.6, help="latent_norm_penalty dead zone (corrected recipe)")
parser.add_argument("--scale", type=float, default=0.9, help="latent_norm_penalty ramp scale (corrected recipe)")
parser.add_argument("--val-fraction", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--suffix", default="_corrected", help="appended to output dir names")
parser.add_argument("--save-plain", action="store_true",
                    help="also freeze+save the speed-blind twin (same treadmill-only pool), for the uncond_real RL arm")
parser.add_argument("--skip-freeze", action="store_true")
args = parser.parse_args()

subj_tag = "_".join(args.subjects) if len(args.subjects) <= 2 else f"{len(args.subjects)}subj"

strides, meta = load_strides(LEAPS_H5_PATH, subjects=args.subjects, modes=["treadmill"], with_metadata=True)
speed_per_stride = meta["speeds"]
assert not np.isnan(speed_per_stride).any(), "treadmill-only filter should guarantee real speed labels"

# Stride-level random split (subject-level is meaningless with one subject --
# stride_train_val_split() would put every stride in val; see
# train_single_subject_decoder.py for the same reasoning).
rng = np.random.default_rng(args.seed)
idx = rng.permutation(len(strides))
n_val = max(1, int(len(strides) * args.val_fraction))
val_idx, train_idx = idx[:n_val], idx[n_val:]
strides_train, strides_val = strides[train_idx], strides[val_idx]
speed_train_per_stride, speed_val_per_stride = speed_per_stride[train_idx], speed_per_stride[val_idx]

# Same per-stride-P99 normalization as get_synergies.py, fit on this
# (treadmill-only) training split -- kept separate from the pooled
# decoder_k*/ divisors since the training population differs.
stride_p99s = np.percentile(strides_train, 99, axis=1)
global_divisors = stride_p99s.mean(axis=0) + 1e-8


def to_unit(strides_3d: np.ndarray) -> np.ndarray:
    return np.clip(strides_3d / global_divisors, 0.0, 1.0).astype(np.float32)


X_train = to_unit(strides_train).reshape(-1, 11)
X_val = to_unit(strides_val).reshape(-1, 11)

# Broadcast per-stride speed to per-frame (101 frames/stride, matching the
# stride-major/frame-minor flatten above), min-max normalized to [0,1] on
# train speeds only.
speed_lo, speed_hi = float(speed_train_per_stride.min()), float(speed_train_per_stride.max())


def speed_to_unit_frames(speed_per_stride: np.ndarray) -> np.ndarray:
    unit = (speed_per_stride - speed_lo) / (speed_hi - speed_lo)
    return np.repeat(unit, 101).astype(np.float32).reshape(-1, 1)


speed_train_frames = speed_to_unit_frames(speed_train_per_stride)
speed_val_frames = speed_to_unit_frames(speed_val_per_stride)

rng_sub = np.random.default_rng(0)
sub_idx = rng_sub.choice(len(X_train), size=min(50_000, len(X_train)), replace=False)
X_train_sub = X_train[sub_idx]
speed_train_sub = speed_train_frames[sub_idx]

print(f"pool: subjects={args.subjects} treadmill-only -- {len(X_train)} train frames, {len(X_val)} val frames, "
      f"speed range [{speed_lo:.2f}, {speed_hi:.2f}] m/s")

ae_kw = dict(epochs=args.epochs, batch_size=512, seed=0, threshold=args.threshold, scale=args.scale)

# Plain (speed-blind) AE on this same treadmill-only pool -- the fair
# baseline for the R2 comparison (isolates "speed conditioning" from
# "treadmill-only subsetting", since decoder_k*/ was trained on all 4 modes)
# and, with --save-plain, the decoder for the uncond_real RL arm.
model_plain = train_ae(X_train_sub, dim_latent=args.k, **ae_kw)
model_plain.eval()
with torch.no_grad():
    val_hat_plain, _ = model_plain(torch.from_numpy(X_val))
r2_plain = r2(X_val, val_hat_plain.numpy())

model_speed = train_ae(X_train_sub, dim_latent=args.k, speed_train=speed_train_sub, **ae_kw)
model_speed.eval()
with torch.no_grad():
    val_hat_speed, _ = model_speed(torch.from_numpy(X_val), None, torch.from_numpy(speed_val_frames))
r2_speed = r2(X_val, val_hat_speed.numpy())

print(f"k={args.k}, treadmill-only: AE held-out R2 = {r2_plain:.3f}, AE_speed held-out R2 = {r2_speed:.3f}")


def freeze_and_save(model, name, *, speed_cond: bool):
    for p in model.decoder.parameters():
        p.requires_grad = False
    decoder_dir = CACHE_DIR / name
    decoder_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.decoder.state_dict(), decoder_dir / "decoder.pt")
    with torch.no_grad():
        if speed_cond:
            _, z_train = model(torch.from_numpy(X_train_sub), None, torch.from_numpy(speed_train_sub))
        else:
            _, z_train = model(torch.from_numpy(X_train_sub))
    z_train = z_train.numpy()
    norm = dict(
        p01=np.zeros(11, dtype=np.float32), p99=global_divisors.astype(np.float32),
        z_lo=z_train.min(axis=0).astype(np.float32), z_hi=z_train.max(axis=0).astype(np.float32),
    )
    if speed_cond:
        norm["speed_lo"] = np.float32(speed_lo)
        norm["speed_hi"] = np.float32(speed_hi)
    np.savez(decoder_dir / "norm.npz", **norm)
    print(f"  frozen -> {decoder_dir}  (speed_cond={speed_cond})")


if args.skip_freeze:
    print("--skip-freeze: nothing written to disk.")
else:
    freeze_and_save(model_speed, f"decoder_k{args.k}_speedcond_{subj_tag}{args.suffix}", speed_cond=True)
    if args.save_plain:
        freeze_and_save(model_plain, f"decoder_k{args.k}_treadmillpool_{subj_tag}{args.suffix}", speed_cond=False)
