"""Train a phase-conditioned VAE and check whether z ~ N(0,I) sampled at a
fixed phase decodes into cross-muscle-correlated activation -- i.e.
whether it's a viable source of Lattice-style structured exploration
noise, learned from real EMG instead of hand-engineered.

Diagnostic/validation only. Does not touch decoder_k*/ (SynergyAE's
frozen decoders, what LatentActionPriorWrapper actually loads for RL) and
is not wired into MPO's action-noise mechanism
(deprl/vendor/tonic/torch/models/actors.py::GaussianPolicyHead, confirmed
to sample independent per-actuator Gaussian noise, no cross-actuator
structure at all) -- that integration is a separate, larger decision, not
made here. Nothing is saved to disk; rerun to reproduce.
"""

import argparse
import os

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import torch

from leaps.synergy_common import prepare_training_arrays, r2, train_cvae

torch.set_num_threads(1)

EMG_CHANNELS = [
    "gastrocmed", "tibialisanterior", "soleus", "vastusmedialis", "vastuslateralis",
    "rectusfemoris", "bicepsfemoris", "semitendinosus", "gracilis", "gluteusmedius",
    "rightexternaloblique",
]

parser = argparse.ArgumentParser(description="Train + validate a phase-conditioned VAE for structured exploration noise.")
parser.add_argument("--dim-latent", type=int, default=6)
parser.add_argument("--beta", type=float, default=1.0, help="KL weight")
parser.add_argument("--beta-warmup-epochs", type=int, default=0, help="linearly ramp KL weight from 0 to --beta over this many epochs (0 = constant beta)")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--n-phase-check", type=int, default=5, help="how many phase points to report sampled-correlation structure for")
parser.add_argument("--n-samples", type=int, default=2000, help="prior samples per phase point for the correlation check")
args = parser.parse_args()

data = prepare_training_arrays()
X_train_sub, phase_train_sub = data["X_train_sub"], data["phase_train_sub"]
X_val, phase_val = data["X_val"], data["phase_val"]

model = train_cvae(
    X_train_sub, phase_train_sub, dim_latent=args.dim_latent,
    epochs=args.epochs, beta=args.beta, beta_warmup_epochs=args.beta_warmup_epochs,
)
model.eval()

# Reconstruction R2 (posterior mean, not a stochastic sample) -- comparable
# to the AE/AE_phase columns in get_synergies.py's own sweep table.
with torch.no_grad():
    mu_val, _ = model.encode(torch.from_numpy(X_val), torch.from_numpy(phase_val))
    val_hat = model.decode(mu_val, torch.from_numpy(phase_val))
print(f"CVAE K={args.dim_latent} beta={args.beta}: held-out R2 (posterior mean) = {r2(X_val, val_hat.numpy()):.3f}")

# Structured-noise check: at several fixed phase points, sample from the
# prior (z ~ N(0,I), no real EMG frame involved) and report the resulting
# cross-channel correlation. Independent-Gaussian exploration noise (MPO's
# current mechanism) has ~0 off-diagonal correlation by construction; real
# muscle synergies should show clear agonist/antagonist structure if the
# decoder learned anything meaningful.
print(f"\nPer-phase sampled-noise cross-channel correlation (n={args.n_samples} samples/phase, prior z~N(0,I)):")
phase_fracs = np.linspace(0.0, 1.0, args.n_phase_check, endpoint=False)
strongest_pairs_seen = {}
for frac in phase_fracs:
    phase_pt = torch.tensor([[np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)]], dtype=torch.float32)
    samples = model.sample(phase_pt, n=args.n_samples).numpy()  # (n_samples, 11)
    corr = np.corrcoef(samples, rowvar=False)
    iu = np.triu_indices(len(EMG_CHANNELS), k=1)
    off_diag = corr[iu]
    top = np.argsort(-np.abs(off_diag))[:3]
    top_desc = ", ".join(
        f"{EMG_CHANNELS[iu[0][t]]}~{EMG_CHANNELS[iu[1][t]]}={off_diag[t]:+.2f}" for t in top
    )
    print(
        f"  phase={frac*100:4.0f}%:  mean|corr|={np.abs(off_diag).mean():.3f}  "
        f"max|corr|={np.abs(off_diag).max():.3f}  top pairs: {top_desc}"
    )

print(
    "\nRead: mean|corr| near 0 across all phases means the decoder isn't inducing any "
    "cross-muscle structure (no better than independent noise -- the idea doesn't pan out "
    "as-is). Consistently nonzero, phase-varying correlation with sensible agonist/antagonist "
    "pairs is the signal that would justify scoping the MPO-side integration next."
)
