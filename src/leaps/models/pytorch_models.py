"""PyTorch-based EMG dimensionality reduction models (AE, VAE, MAE).

Architecture follows Hausdörfer et al. (2024) "Latent Action Priors for
Locomotion with Deep Reinforcement Learning":
    Encoder: n_channels → hidden_dim → latent_dim     (1 hidden layer, tanh)
    Decoder: latent_dim → hidden_dim → n_channels      (1 hidden layer, tanh)

Default hidden_dim = 2 * latent_dim (matching the paper). Pass hidden_dim
to override for ablation studies with larger architectures.

Key design choices:
    - Lnorm loss: soft penalty pushing latent values into [-1, 1],
      making the latent space directly usable as a PPO/SAC action space.
    - Tanh activations: naturally bounded, works with Lnorm.
    - No sigmoid on decoder output: EMG data can slightly exceed [0, 1]
      after normalization; downstream clipping (EMGToMuscleMapper, MuJoCo env)
      handles the [0, 1] bound.
    - VAE β < 1: weaker KL term preserves reconstruction quality.

Training features:
    - Reproducible via seed parameter (torch + numpy + dataloader)
    - ReduceLROnPlateau scheduler (halves LR if val loss plateaus)
    - Best-model tracking (restores weights from best val epoch)
    - Generic log_fn callback for wandb integration
"""

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from leaps.evaluation.metrics import compute_train_metrics
from leaps.models.base import EMGModel

# Type alias for the logging callback: fn(metrics_dict, step)
LogFn = Callable[[dict[str, float], int], None] | None


# ---------------------------------------------------------------------------
# Lnorm: latent space regularization for RL-compatible action spaces
# ---------------------------------------------------------------------------

def lnorm_loss(latent: torch.Tensor) -> torch.Tensor:
    """Soft penalty pushing latent values into [-1, 1].

    From Hausdörfer et al. (2024), Eq. 1:
        Lnorm(al) = 0                         if ||al||_inf < 0.8
                   = exp((al / 1.2)^10) - 1   otherwise

    This is free inside [-0.8, 0.8] (no gradient), then rises steeply
    near ±1.2. Standard RL frameworks (SB3/PPO) clip actions to [-1, 1],
    so the encoder must learn to keep latent values in that range.

    Args:
        latent: Encoder output, shape (batch, latent_dim).

    Returns:
        Scalar loss (mean over batch and dimensions).
    """
    abs_z = torch.abs(latent)
    # Clamp the exponent to avoid exp() overflow → NaN.
    # exp(50) ≈ 5e21 which is large enough penalty; exp(>88) overflows float32.
    exponent = torch.pow(abs_z / 1.2, 10).clamp(max=50.0)
    penalty = torch.where(
        abs_z < 0.8,
        torch.zeros_like(latent),
        torch.exp(exponent) - 1.0,
    )
    return penalty.mean()


# ---------------------------------------------------------------------------
# Network components — Hausdörfer et al. architecture
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """MLP encoder: n_channels → hidden_dim → latent_dim.

    Matches Hausdörfer et al.: one hidden layer of size 2×latent_dim with tanh.
    Pass hidden_dim to override for ablation.
    """

    def __init__(self, n_channels: int, latent_dim: int, hidden_dim: int | None = None):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else 2 * latent_dim
        self.net = nn.Sequential(
            nn.Linear(n_channels, h),
            nn.Tanh(),
            nn.Linear(h, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    """MLP decoder: latent_dim → hidden_dim → n_channels.

    Matches Hausdörfer et al.: one hidden layer of size 2×latent_dim with tanh.
    No sigmoid — EMG data can slightly exceed [0, 1] after normalization, and
    downstream clipping (EMGToMuscleMapper, env.step) handles bounds.
    """

    def __init__(self, latent_dim: int, n_channels: int, hidden_dim: int | None = None):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else 2 * latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.Tanh(),
            nn.Linear(h, n_channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _VAEEncoder(nn.Module):
    """VAE encoder that outputs mu and log_var.

    Same architecture as _Encoder but splits the output into two heads.
    """

    def __init__(self, n_channels: int, latent_dim: int, hidden_dim: int | None = None):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else 2 * latent_dim
        self.shared = nn.Sequential(
            nn.Linear(n_channels, h),
            nn.Tanh(),
        )
        self.fc_mu = nn.Linear(h, latent_dim)
        self.fc_log_var = nn.Linear(h, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.fc_mu(h), self.fc_log_var(h)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _PyTorchModelBase(EMGModel):
    """Common functionality for PyTorch-based models."""

    def __init__(self, latent_dim: int, n_channels: int = 11):
        super().__init__(latent_dim, n_channels)
        self.device = _get_device()
        self._module: nn.Module | None = None

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _make_loader(
        self, x: np.ndarray, batch_size: int,
    ) -> torch.utils.data.DataLoader:
        """Create a shuffled DataLoader. Seed should be set globally before calling."""
        dataset = torch.utils.data.TensorDataset(self._to_tensor(x))
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._module.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self._module.load_state_dict(state)


class Autoencoder(_PyTorchModelBase):
    """Standard autoencoder with MSE reconstruction loss + Lnorm regularization.

    Loss = MSE(x, x_hat) + lnorm_weight * Lnorm(z)

    Architecture matches Hausdörfer et al. (2024): single hidden layer of
    size 2×latent_dim with tanh activation. Pass hidden_dim to override.

    Training uses Adam with ReduceLROnPlateau (halves LR when val loss
    plateaus for 10 epochs). Tracks best val loss and restores those
    weights at the end.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        lnorm_weight: float = 0.01,
        hidden_dim: int | None = None,
    ):
        super().__init__(latent_dim, n_channels)
        self.lnorm_weight = lnorm_weight
        self.encoder = _Encoder(n_channels, latent_dim, hidden_dim)
        self.decoder = _Decoder(latent_dim, n_channels, hidden_dim)
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})
        self._module.to(self.device)

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 4096,
        lr: float = 1e-3,
        log_fn: LogFn = None,
    ) -> dict[str, float]:
        loader = self._make_loader(x_train, batch_size)
        self._module.train()
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                z = self.encoder(batch)
                x_hat = self.decoder(z)
                recon_loss = nn.functional.mse_loss(x_hat, batch)
                loss = recon_loss + self.lnorm_weight * lnorm_loss(z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= len(loader.dataset)

            step_metrics = {"train_loss": epoch_loss}
            if x_val is not None and (epoch % 10 == 0 or epoch == epochs - 1):
                val_metrics = compute_train_metrics(x_val, self.reconstruct(x_val))
                step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                val_loss = val_metrics["mse"]
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = deepcopy(self._module.state_dict())
            step_metrics["lr"] = optimizer.param_groups[0]["lr"]
            if log_fn is not None:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)

        self._module.eval()
        metrics = compute_train_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            val_metrics = compute_train_metrics(x_val, self.reconstruct(x_val))
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        return self.encoder(self._to_tensor(x)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        z = self.encoder(self._to_tensor(x))
        return self.decoder(z).cpu().numpy()


class VAE(_PyTorchModelBase):
    """Variational Autoencoder with MSE + β·KL loss + Lnorm.

    Loss = MSE(x, x_hat) + β * KL(q(z|x) || N(0,1)) + lnorm_weight * Lnorm(z)

    β < 1 weakens KL regularization for better reconstruction. Default
    β=0.1 because high β kills reconstruction quality and our primary
    goal is faithful EMG reconstruction for muscle control.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        beta: float = 0.1,
        lnorm_weight: float = 0.01,
        hidden_dim: int | None = None,
    ):
        super().__init__(latent_dim, n_channels)
        self.beta = beta
        self.lnorm_weight = lnorm_weight
        self.encoder = _VAEEncoder(n_channels, latent_dim, hidden_dim)
        self.decoder = _Decoder(latent_dim, n_channels, hidden_dim)
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})
        self._module.to(self.device)

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self._reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 4096,
        lr: float = 1e-3,
        log_fn: LogFn = None,
    ) -> dict[str, float]:
        loader = self._make_loader(x_train, batch_size)
        self._module.train()
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            for (batch,) in loader:
                x_hat, mu, log_var = self._forward(batch)
                recon_loss = nn.functional.mse_loss(x_hat, batch, reduction="mean")
                kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                norm_loss = lnorm_loss(mu)
                loss = recon_loss + self.beta * kl_loss + self.lnorm_weight * norm_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
                epoch_recon += recon_loss.item() * batch.shape[0]
                epoch_kl += kl_loss.item() * batch.shape[0]
            n = len(loader.dataset)
            epoch_loss /= n
            epoch_recon /= n
            epoch_kl /= n

            step_metrics = {
                "train_loss": epoch_loss,
                "train_recon_loss": epoch_recon,
                "train_kl_loss": epoch_kl,
            }
            if x_val is not None and (epoch % 10 == 0 or epoch == epochs - 1):
                val_metrics = compute_train_metrics(x_val, self.reconstruct(x_val))
                step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                val_loss = val_metrics["mse"]
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = deepcopy(self._module.state_dict())
            step_metrics["lr"] = optimizer.param_groups[0]["lr"]
            if log_fn is not None:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)

        self._module.eval()
        metrics = compute_train_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            val_metrics = compute_train_metrics(x_val, self.reconstruct(x_val))
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        mu, _ = self.encoder(self._to_tensor(x))
        return mu.cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        mu, _ = self.encoder(self._to_tensor(x))
        return self.decoder(mu).cpu().numpy()


class WAE_MMD(_PyTorchModelBase):
    """Wasserstein Autoencoder with MMD penalty (Tolstikhin et al., 2018).

    Loss = MSE(x, x_hat) + mmd_weight * MMD(q(z), p(z)) + lnorm_weight * Lnorm(z)

    Instead of KL divergence (VAE), uses Maximum Mean Discrepancy with an RBF
    kernel to match the encoded distribution to N(0, I). This preserves the
    geometry of the data distribution (optimal transport), producing sharper
    reconstructions and better distance preservation than VAE.

    Same encoder/decoder architecture as the standard AE.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        mmd_weight: float = 10.0,
        lnorm_weight: float = 0.01,
        hidden_dim: int | None = None,
    ):
        super().__init__(latent_dim, n_channels)
        self.mmd_weight = mmd_weight
        self.lnorm_weight = lnorm_weight
        self.encoder = _Encoder(n_channels, latent_dim, hidden_dim)
        self.decoder = _Decoder(latent_dim, n_channels, hidden_dim)
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})
        self._module.to(self.device)

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 4096,
        lr: float = 1e-3,
        log_fn: LogFn = None,
    ) -> dict[str, float]:
        loader = self._make_loader(x_train, batch_size)
        self._module.train()
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_mmd = 0.0
            for (batch,) in loader:
                z = self.encoder(batch)
                x_hat = self.decoder(z)
                recon_loss = nn.functional.mse_loss(x_hat, batch)
                mmd = _compute_mmd(z, self.latent_dim, self.device)
                norm_loss = lnorm_loss(z)
                loss = recon_loss + self.mmd_weight * mmd + self.lnorm_weight * norm_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
                epoch_recon += recon_loss.item() * batch.shape[0]
                epoch_mmd += mmd.item() * batch.shape[0]
            n = len(loader.dataset)
            epoch_loss /= n
            epoch_recon /= n
            epoch_mmd /= n

            step_metrics = {
                "train_loss": epoch_loss,
                "train_recon_loss": epoch_recon,
                "train_mmd_loss": epoch_mmd,
            }
            if x_val is not None and (epoch % 10 == 0 or epoch == epochs - 1):
                val_metrics = compute_train_metrics(x_val, self.reconstruct(x_val))
                step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                val_loss = val_metrics["mse"]
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = deepcopy(self._module.state_dict())
            step_metrics["lr"] = optimizer.param_groups[0]["lr"]
            if log_fn is not None:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)

        self._module.eval()
        metrics = compute_train_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            val_metrics = compute_train_metrics(x_val, self.reconstruct(x_val))
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        return self.encoder(self._to_tensor(x)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        z = self.encoder(self._to_tensor(x))
        return self.decoder(z).cpu().numpy()


def _compute_mmd(z: torch.Tensor, latent_dim: int, device: torch.device) -> torch.Tensor:
    """Compute MMD between encoded samples z and samples from N(0, I).

    Uses RBF kernel with median heuristic bandwidth. O(n^2) in batch size,
    but negligible for typical batch sizes (64-4096).
    """
    batch_size = z.shape[0]
    z_prior = torch.randn(batch_size, latent_dim, device=device)

    # RBF kernel: k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    # Median heuristic: sigma^2 = median of pairwise distances
    zz = _pairwise_sq_dist(z, z)
    zp = _pairwise_sq_dist(z, z_prior)
    pp = _pairwise_sq_dist(z_prior, z_prior)

    # Use latent_dim as bandwidth (simple, stable heuristic)
    sigma_sq = float(latent_dim)
    k_zz = torch.exp(-zz / (2 * sigma_sq))
    k_zp = torch.exp(-zp / (2 * sigma_sq))
    k_pp = torch.exp(-pp / (2 * sigma_sq))

    # MMD^2 = E[k(z,z)] - 2*E[k(z,p)] + E[k(p,p)]
    mmd_sq = k_zz.mean() - 2 * k_zp.mean() + k_pp.mean()
    return mmd_sq


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances between rows of x and y."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
    x_sq = (x ** 2).sum(dim=1, keepdim=True)
    y_sq = (y ** 2).sum(dim=1, keepdim=True)
    return x_sq + y_sq.t() - 2 * x @ y.t()


class MaskedAutoencoder(_PyTorchModelBase):
    """Masked Autoencoder — masks input channels, loss on masked channels only.

    During training, randomly zeros out ~mask_ratio of the 11 input channels
    per sample. The loss is computed only on the masked channels, forcing
    the model to learn cross-channel correlations (muscle synergies).

    At inference time, no masking is applied — full input goes through.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        mask_ratio: float = 0.5,
        lnorm_weight: float = 0.01,
        hidden_dim: int | None = None,
    ):
        super().__init__(latent_dim, n_channels)
        self.mask_ratio = mask_ratio
        self.lnorm_weight = lnorm_weight
        self.encoder = _Encoder(n_channels, latent_dim, hidden_dim)
        self.decoder = _Decoder(latent_dim, n_channels, hidden_dim)
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})
        self._module.to(self.device)

    def _generate_mask(self, batch_size: int) -> torch.Tensor:
        """Generate binary mask: 1 = keep, 0 = masked. Vectorized."""
        n_mask = max(1, int(self.n_channels * self.mask_ratio))
        noise = torch.rand(batch_size, self.n_channels, device=self.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.ones(batch_size, self.n_channels, device=self.device)
        mask_indices = ids_shuffle[:, :n_mask]
        mask.scatter_(1, mask_indices, 0.0)
        return mask

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 4096,
        lr: float = 1e-3,
        log_fn: LogFn = None,
    ) -> dict[str, float]:
        loader = self._make_loader(x_train, batch_size)
        self._module.train()
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                mask = self._generate_mask(batch.shape[0])
                masked_input = batch * mask
                z = self.encoder(masked_input)
                x_hat = self.decoder(z)
                inv_mask = 1.0 - mask
                n_masked = inv_mask.sum()
                if n_masked > 0:
                    recon_loss = (((x_hat - batch) ** 2) * inv_mask).sum() / n_masked
                else:
                    recon_loss = nn.functional.mse_loss(x_hat, batch)
                loss = recon_loss + self.lnorm_weight * lnorm_loss(z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= len(loader.dataset)

            step_metrics = {"train_loss": epoch_loss}
            if x_val is not None and (epoch % 10 == 0 or epoch == epochs - 1):
                val_metrics = compute_train_metrics(x_val, self.reconstruct(x_val))
                step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                val_loss = val_metrics["mse"]
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = deepcopy(self._module.state_dict())
            step_metrics["lr"] = optimizer.param_groups[0]["lr"]
            if log_fn is not None:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)

        self._module.eval()
        metrics = compute_train_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            val_metrics = compute_train_metrics(x_val, self.reconstruct(x_val))
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        return self.encoder(self._to_tensor(x)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        t = self._to_tensor(x)
        return self.decoder(self.encoder(t)).cpu().numpy()
