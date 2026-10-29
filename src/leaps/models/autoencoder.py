"""Synergy autoencoder, its latent penalty and training loop."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)


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
