"""Shared data/model/cache helpers for the get_synergy notebook.
"""

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides, stride_train_val_split

torch.set_num_threads(1)

CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "synergy"


def cached(name: str, fn, *, force: bool = False):
    """Return fn()'s result, computing it once and reusing the pickled copy after."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.pkl"
    if not force and path.exists():
        return pickle.loads(path.read_bytes())
    result = fn()
    path.write_bytes(pickle.dumps(result))
    return result


def r2(x, x_hat):
    ss_res = ((x - x_hat) ** 2).sum()
    ss_tot = ((x - x.mean(axis=0)) ** 2).sum()
    return 1 - ss_res / ss_tot


@lru_cache(maxsize=1)
def load_split(seed: int = 42, val_fraction: float = 0.2):
    """Subject-level train/val split of all strides+metadata. Cached in-process."""
    strides, meta = load_strides(LEAPS_H5_PATH, with_metadata=True)
    return stride_train_val_split(strides, meta, val_fraction=val_fraction, seed=seed)


@dataclass
class UnitScaler:
    """Maps a muscle-activation pool to/from [0,1] using its 1st/99th percentiles."""

    p01: np.ndarray
    p99: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, pct: float = 1.0) -> "UnitScaler":
        return cls(np.percentile(X, pct, axis=0), np.percentile(X, 100 - pct, axis=0))

    @property
    def _range(self) -> np.ndarray:
        return self.p99 - self.p01 + 1e-8

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.p01) / self._range, 0, 1).astype(np.float32)

    def from_unit(self, x: np.ndarray) -> np.ndarray:
        return (x * self._range + self.p01).astype(np.float32)


class SynergyAE(nn.Module):
    """Finalized architecture: tanh hidden, linear latent, sigmoid output.

    Sigmoid output because SCONE muscle excitations are [0,1] (see
    depRL/deprl/env_wrappers/scone_wrapper.py); linear latent per the
    reference paper's spec.

    phase_aware=True concatenates a 2-dim sin/cos gait-phase encoding onto
    both the encoder's and decoder's input, so the same muscle-activation
    frame from different points in the gait cycle isn't forced through the
    same latent code. Off by default -- reproduces the original
    architecture exactly.

    speed_aware=True concatenates a 1-dim normalized walking-speed scalar
    the same way -- pooling across the Camargo dataset's full 0.5-1.4 m/s
    treadmill speed range otherwise forces one synergy basis to cover
    gaits that are biomechanically different at the muscle-coordination
    level, not just scaled versions of each other. Unlike phase (which has
    no online equivalent during an RL rollout -- the sim has no
    ground-truth "% of gait cycle" signal), a target speed is a known
    constant per sconewalk_* environment, so a speed-conditioned decoder is
    actually deployable at RL time: pass the fixed target speed every step.

    subject_aware=True concatenates a subject_dim-wide per-subject
    "fingerprint" the same way (default 55 -- the upper triangle of an
    11x11 cross-channel EMG correlation matrix, see
    leaps/scripts/extract_subject_fingerprints.py). Deployable at RL time
    the same way speed is: the fingerprint is a fixed, known-in-advance
    constant for whichever person's data a run is being conditioned on
    (including, for a genuinely new person never seen during decoder
    training, a fingerprint computed from a small amount of their own raw
    EMG -- no retraining required). This is the mechanism the
    subject-conditioned personalization framework is built on: one shared
    decoder learns to use the fingerprint as a real conditioning signal
    across many training subjects, instead of one decoder per person.

    depth controls hidden-layer count on each side (default 1, matching the
    original architecture exactly). depth=2 matches the reference LAP repo's
    nonLinearAE (github.com/OliEfr/latent-action-priors, utils.py) -- two
    Tanh hidden layers of width dim_latent*2 instead of one, same width
    convention both places.
    """

    def __init__(
        self, dim_a: int, dim_latent: int,
        phase_aware: bool = False, speed_aware: bool = False,
        subject_aware: bool = False, subject_dim: int = 55,
        depth: int = 1,
    ):
        super().__init__()
        h = 2 * dim_latent
        phase_dim = 2 if phase_aware else 0
        speed_dim = 1 if speed_aware else 0
        subj_dim = subject_dim if subject_aware else 0
        cond_dim = phase_dim + speed_dim + subj_dim
        self.phase_aware = phase_aware
        self.speed_aware = speed_aware
        self.subject_aware = subject_aware
        self.depth = depth

        enc_layers = [nn.Linear(dim_a + cond_dim, h), nn.Tanh()]
        for _ in range(depth - 1):
            enc_layers += [nn.Linear(h, h), nn.Tanh()]
        enc_layers += [nn.Linear(h, dim_latent)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = [nn.Linear(dim_latent + cond_dim, h), nn.Tanh()]
        for _ in range(depth - 1):
            dec_layers += [nn.Linear(h, h), nn.Tanh()]
        dec_layers += [nn.Linear(h, dim_a), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def _cond(self, phase, speed, subject):
        parts = []
        if self.phase_aware:
            parts.append(phase)
        if self.speed_aware:
            parts.append(speed)
        if self.subject_aware:
            parts.append(subject)
        return torch.cat(parts, dim=-1) if parts else None

    def forward(self, a, phase=None, speed=None, subject=None):
        cond = self._cond(phase, speed, subject)
        enc_in = torch.cat([a, cond], dim=-1) if cond is not None else a
        z = self.encoder(enc_in)
        dec_in = torch.cat([z, cond], dim=-1) if cond is not None else z
        return self.decoder(dec_in), z


def latent_norm_penalty(a_l: torch.Tensor, threshold: float = 0.5, scale: float = 0.6) -> torch.Tensor:
    worst = a_l.abs().max(dim=-1).values
    penalty = torch.where(worst < threshold, torch.zeros_like(worst), torch.exp((worst / scale) ** 10) - 1.0)
    return penalty.mean()


def train_ae(
    X_train: np.ndarray,
    dim_latent: int,
    *,
    phase_train: np.ndarray = None,
    speed_train: np.ndarray = None,
    subject_train: np.ndarray = None,
    subject_noise_std: float = 0.0,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    penalty_weight: float = 1.0,
    threshold: float = 0.5,
    scale: float = 0.6,
    depth: int = 1,
) -> SynergyAE:
    """phase_train, if given, is a (len(X_train), 2) sin/cos gait-phase
    array aligned row-for-row with X_train -- trains a phase_aware AE.
    speed_train, if given, is a (len(X_train), 1) normalized-speed array,
    same alignment -- trains a speed_aware AE. subject_train, if given, is
    a (len(X_train), subject_dim) array -- each row the fingerprint of
    whichever subject that frame came from -- trains a subject_aware AE.
    All three can be passed together.

    subject_noise_std adds i.i.d. Gaussian noise (this std, per dimension)
    to the subject fingerprint on every training batch, never at eval time.
    With only as many distinct fingerprints as training subjects (e.g. 20),
    a small conditioning-input network has an easy shortcut available --
    memorize those exact points instead of learning a mapping that
    generalizes to a nearby-but-unseen fingerprint. Perturbing the
    conditioning input during training removes that shortcut: the network
    can only do well on average if it's robust to small movements around
    each training fingerprint, which is exactly the robustness a genuinely
    new subject's fingerprint needs to land inside."""
    torch.manual_seed(seed)
    model = SynergyAE(
        dim_a=X_train.shape[1], dim_latent=dim_latent,
        phase_aware=phase_train is not None, speed_aware=speed_train is not None,
        subject_aware=subject_train is not None,
        subject_dim=subject_train.shape[1] if subject_train is not None else 55,
        depth=depth,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tensors = [torch.from_numpy(X_train)]
    if phase_train is not None:
        tensors.append(torch.from_numpy(phase_train))
    if speed_train is not None:
        tensors.append(torch.from_numpy(speed_train))
    if subject_train is not None:
        tensors.append(torch.from_numpy(subject_train))
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for batch in loader:
            idx = 1
            a = batch[0]
            phase = None
            speed = None
            subject = None
            if phase_train is not None:
                phase = batch[idx]; idx += 1
            if speed_train is not None:
                speed = batch[idx]; idx += 1
            if subject_train is not None:
                subject = batch[idx]; idx += 1
                if subject_noise_std > 0:
                    subject = subject + subject_noise_std * torch.randn_like(subject)
            a_hat, a_l = model(a, phase, speed, subject)
            loss = ((a - a_hat) ** 2).sum(dim=-1).mean() + penalty_weight * latent_norm_penalty(a_l, threshold, scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


class ConditionalVAE(nn.Module):
    """Phase-conditioned VAE: same tanh/sigmoid style as SynergyAE, but a
    real stochastic latent (KL(q(z|a,phase) || N(0,I))) instead of the
    deterministic AE's soft latent-norm penalty.

    Built for one thing SynergyAE structurally can't do: generate a full,
    phase-appropriate, cross-muscle-correlated activation sample from
    scratch -- z ~ N(0,I); decoder(z, phase) -- no encoder or real EMG
    frame needed at sample time. Candidate source of structured
    exploration noise (cf. Lattice, Chiappa et al. 2023), as opposed to
    MPO's current per-actuator independent Gaussian (confirmed in
    deprl/vendor/tonic/torch/models/actors.py::GaussianPolicyHead -- zero
    cross-actuator correlation by construction).
    """

    def __init__(self, dim_a: int, dim_latent: int, phase_dim: int = 2):
        super().__init__()
        h = 2 * dim_latent
        self.dim_latent = dim_latent
        self.phase_dim = phase_dim
        self.enc_torso = nn.Sequential(nn.Linear(dim_a + phase_dim, h), nn.Tanh())
        self.mu_layer = nn.Linear(h, dim_latent)
        self.logvar_layer = nn.Linear(h, dim_latent)
        self.decoder = nn.Sequential(nn.Linear(dim_latent + phase_dim, h), nn.Tanh(), nn.Linear(h, dim_a), nn.Sigmoid())

    def encode(self, a, phase):
        h = self.enc_torso(torch.cat([a, phase], dim=-1))
        return self.mu_layer(h), self.logvar_layer(h)

    def decode(self, z, phase):
        return self.decoder(torch.cat([z, phase], dim=-1))

    def forward(self, a, phase):
        mu, logvar = self.encode(a, phase)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z, phase), mu, logvar

    def sample(self, phase: torch.Tensor, n: int = 1) -> torch.Tensor:
        """Draw n phase-appropriate activation samples per phase row,
        straight from the prior -- the exploration-noise entry point."""
        with torch.no_grad():
            phase_rep = phase.repeat_interleave(n, dim=0)
            z = torch.randn(phase_rep.shape[0], self.dim_latent)
            return self.decode(z, phase_rep)


def train_cvae(
    X_train: np.ndarray,
    phase_train: np.ndarray,
    dim_latent: int,
    *,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    beta: float = 1.0,
    beta_warmup_epochs: int = 0,
) -> ConditionalVAE:
    """epochs=0 skips training entirely and returns the randomly-initialized
    model as-is -- useful as an architecture-only control (e.g. to check
    whether decoded-sample correlation structure is learned or just an
    artifact of squashing through a narrow shared hidden layer).

    beta_warmup_epochs linearly ramps the KL weight from 0 up to `beta`
    over that many epochs (standard VAE warm-up, avoids the KL term
    dominating before reconstruction has started resolving -- a common
    cause of an over-regularized, blurry decoder). 0 disables warmup
    (beta is constant from epoch 0), matching the original behavior.
    """
    torch.manual_seed(seed)
    model = ConditionalVAE(dim_a=X_train.shape[1], dim_latent=dim_latent, phase_dim=phase_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(phase_train)),
        batch_size=batch_size, shuffle=True,
    )
    for epoch in range(epochs):
        beta_t = beta * min(1.0, (epoch + 1) / beta_warmup_epochs) if beta_warmup_epochs > 0 else beta
        for a, phase in loader:
            a_hat, mu, logvar = model(a, phase)
            recon = ((a - a_hat) ** 2).sum(dim=-1).mean()
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()
            loss = recon + beta_t * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def prepare_training_arrays(seed: int = 42, val_fraction: float = 0.2, n_sub: int = 50_000) -> dict:
    """Per-stride-P99 normalization to [0,1] + sin/cos gait-phase encoding,
    matching get_synergies.py's own pipeline exactly (same normalization
    convention, same phase derivation) so results from both stay
    comparable. Phase is free: reshape(-1, 11) below flattens
    (n_strides, 101, 11) stride-major/frame-minor, so row i's phase is
    exactly (i % 101)/100 -- each stride is already time-normalized to 101
    points, 0-100% of the gait cycle, verified directly against the source
    h5.
    """
    strides, meta = load_strides(LEAPS_H5_PATH, modes=ALL_MODES, with_metadata=True)
    strides_train, strides_val, meta_train, meta_val = stride_train_val_split(
        strides, meta, val_fraction=val_fraction, seed=seed
    )
    stride_p99s = np.percentile(strides_train, 99, axis=1)
    global_divisors = stride_p99s.mean(axis=0) + 1e-8

    def to_unit(strides_3d: np.ndarray) -> np.ndarray:
        return np.clip(strides_3d / global_divisors, 0.0, 1.0).astype(np.float32)

    def phase_sincos(n_strides: int) -> np.ndarray:
        frac = np.tile(np.linspace(0.0, 1.0, 101, dtype=np.float32), n_strides)
        return np.stack([np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)], axis=1)

    X_train = to_unit(strides_train).reshape(-1, 11)
    X_val = to_unit(strides_val).reshape(-1, 11)
    phase_train = phase_sincos(strides_train.shape[0])
    phase_val = phase_sincos(strides_val.shape[0])

    rng_sub = np.random.default_rng(0)
    sub_idx = rng_sub.choice(len(X_train), size=min(n_sub, len(X_train)), replace=False)

    return dict(
        X_train=X_train, X_val=X_val, phase_train=phase_train, phase_val=phase_val,
        X_train_sub=X_train[sub_idx], phase_train_sub=phase_train[sub_idx],
        global_divisors=global_divisors, meta_val=meta_val,
    )
