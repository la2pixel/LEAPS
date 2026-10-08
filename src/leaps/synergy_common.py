"""Synergy autoencoder, its loss and training loop, shared by the decoder training and analysis scripts."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides, stride_train_val_split
from leaps.paths import RESULTS_DIR

torch.set_num_threads(1)

CACHE_DIR = RESULTS_DIR / "synergy"


def r2(x, x_hat):
    ss_res = ((x - x_hat) ** 2).sum()
    ss_tot = ((x - x.mean(axis=0)) ** 2).sum()
    return 1 - ss_res / ss_tot


class SynergyAE(nn.Module):
    """tanh hidden layers of width 2k, linear latent, sigmoid output.
    depth = hidden layers per side. phase_aware concatenates a sin/cos
    gait-phase encoding onto the encoder and decoder inputs."""

    def __init__(self, dim_a: int, dim_latent: int, phase_aware: bool = False, depth: int = 1):
        super().__init__()
        h = 2 * dim_latent
        cond_dim = 2 if phase_aware else 0
        self.phase_aware = phase_aware
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

    def forward(self, a, phase=None):
        enc_in = torch.cat([a, phase], dim=-1) if self.phase_aware else a
        z = self.encoder(enc_in)
        dec_in = torch.cat([z, phase], dim=-1) if self.phase_aware else z
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
    X_val: np.ndarray = None,
    history: list = None,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    penalty_weight: float = 1.0,
    threshold: float = 0.5,
    scale: float = 0.6,
    depth: int = 1,
) -> SynergyAE:
    """phase_train, if given, is a (len(X_train), 2) sin/cos gait-phase array
    aligned with X_train. history, if given, gets one (train_loss, val_loss)
    per epoch appended; val_loss is the reconstruction term on X_val."""
    torch.manual_seed(seed)
    model = SynergyAE(dim_a=X_train.shape[1], dim_latent=dim_latent, phase_aware=phase_train is not None, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tensors = [torch.from_numpy(X_train)]
    if phase_train is not None:
        tensors.append(torch.from_numpy(phase_train))
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        batch_losses = []
        for batch in loader:
            a = batch[0]
            phase = batch[1] if phase_train is not None else None
            a_hat, a_l = model(a, phase)
            loss = ((a - a_hat) ** 2).sum(dim=-1).mean() + penalty_weight * latent_norm_penalty(a_l, threshold, scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        if history is not None:
            with torch.no_grad():
                x = torch.from_numpy(X_val)
                val_loss = ((x - model(x)[0]) ** 2).sum(dim=-1).mean().item()
            history.append((float(np.mean(batch_losses)), val_loss))
    return model


def prepare_training_arrays(seed: int = 42, val_fraction: float = 0.2, n_sub: int = 50_000) -> dict:
    """Per-stride-P99 normalization to [0, 1] and sin/cos gait phase per
    frame, as in get_synergies.py."""
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
