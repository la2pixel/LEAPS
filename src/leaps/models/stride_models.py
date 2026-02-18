"""Stride-level EMG dimensionality reduction models.

These models operate on full stride sequences (101, 11) instead of individual
snapshots (11,). The latent representation is latent_dim-dimensional (one
vector per stride), and the decoder reconstructs the full (101, 11) stride.

Sklearn models (StridePCA, StrideNMF):
    Flatten (101, 11) → (1111,), apply sklearn, unflatten.

PyTorch models (StrideAE, StrideVAE, StrideMAE):
    1D convolutions over the 101 timepoints preserve temporal structure.
    Architecture:
        Encoder: Conv1d(11→32, k=7, s=2) → Conv1d(32→64, k=5, s=2) → FC → latent_dim
        Decoder: FC → ConvTranspose1d(64→32) → ConvTranspose1d(32→11) → sigmoid → (101, 11)

Design choices from "Latent Action Priors for Locomotion" paper:
    - Lnorm loss: keeps latent space in [-1, 1] for RL compatibility
    - Sigmoid output: decoder outputs bounded to [0, 1]
    - VAE β=0.1: mild KL to preserve reconstruction quality
    - MAE block masking: masks contiguous time blocks (not random timepoints)
      to force learning of longer-range temporal dynamics
"""

import pickle
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import NMF, PCA

from leaps.evaluation.metrics import compute_train_metrics
from leaps.models.base import EMGModel
from leaps.models.pytorch_models import lnorm_loss

LogFn = Callable[[dict[str, float], int], None] | None

STRIDE_LEN = 101


# ---------------------------------------------------------------------------
# Sklearn stride models
# ---------------------------------------------------------------------------


class StridePCAModel(EMGModel):
    """PCA on flattened strides — (101, 11) → flatten to 1111 → PCA → unflatten."""

    def __init__(self, latent_dim: int, n_channels: int = 11, stride_len: int = STRIDE_LEN):
        super().__init__(latent_dim, n_channels)
        self.stride_len = stride_len
        self._flat_dim = stride_len * n_channels
        self._model = PCA(n_components=latent_dim)

    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        x_flat = x_train.reshape(x_train.shape[0], -1)
        self._model.fit(x_flat)
        x_hat = self.reconstruct(x_train)
        metrics = _stride_metrics(x_train, x_hat)
        metrics["explained_variance_ratio"] = float(
            np.sum(self._model.explained_variance_ratio_)
        )
        if x_val is not None:
            val_metrics = _stride_metrics(x_val, self.reconstruct(x_val))
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        return metrics

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._model.transform(x.reshape(x.shape[0], -1))

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        x_flat = x.reshape(x.shape[0], -1)
        x_hat_flat = self._model.inverse_transform(self._model.transform(x_flat))
        return x_hat_flat.reshape(-1, self.stride_len, self.n_channels)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)


class StrideNMFModel(EMGModel):
    """Standard NMF on flattened strides.

    Factorizes each stride (flattened to 1111-dim) as a non-negative
    linear combination of basis strides:  X ≈ W @ H

    - W (n_strides, k): activation coefficients (latent representation)
    - H (k, 1111): basis strides (learned components)

    Non-negativity is natural for EMG (muscle activations are non-negative).
    This is the standard muscle synergy approach applied at the stride level.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
        max_iter: int = 500,
        seed: int = 42,
    ):
        super().__init__(latent_dim, n_channels)
        self.stride_len = stride_len
        self._model = NMF(
            n_components=latent_dim,
            init="nndsvda",
            max_iter=max_iter,
            random_state=seed,
        )

    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        x_flat = np.clip(x_train.reshape(x_train.shape[0], -1), 0, None)
        self._model.fit(x_flat)
        x_hat = self.reconstruct(x_train)
        metrics = _stride_metrics(x_train, x_hat)
        metrics["reconstruction_err"] = float(self._model.reconstruction_err_)
        if x_val is not None:
            val_metrics = _stride_metrics(x_val, self.reconstruct(x_val))
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        return metrics

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._model.transform(np.clip(x.reshape(x.shape[0], -1), 0, None))

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        latent = self.transform(x)
        x_hat_flat = latent @ self._model.components_
        return x_hat_flat.reshape(-1, self.stride_len, self.n_channels)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)


class StrideCNMFModel(EMGModel):
    """Smoothness-Constrained NMF (CNMF) on flattened strides.

    Adds an L2 (Frobenius norm) penalty on the activation coefficients W
    to the standard NMF objective:

        min  ½·‖X − W·H‖²_F  +  α·‖W‖²_F    s.t. W, H ≥ 0

    The smoothness constraint (α) improves convergence and reduces
    sensitivity to local optima — a known issue with NMF's non-convex
    objective. Uses multiplicative update solver to match the iterative
    scheme from the muscle synergy literature.

    Reference: Lee & Seung (1999), with smoothness extension from
    muscle synergy extraction studies.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
        alpha: float = 0.1,
        max_iter: int = 500,
        seed: int = 42,
    ):
        super().__init__(latent_dim, n_channels)
        self.stride_len = stride_len
        self.alpha = alpha
        self._model = NMF(
            n_components=latent_dim,
            init="nndsvda",
            solver="mu",           # multiplicative updates (matches paper)
            alpha_W=alpha,         # L2 penalty on activation coefficients
            l1_ratio=0.0,          # pure L2 (Frobenius norm, not L1 sparsity)
            max_iter=max_iter,
            random_state=seed,
        )

    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        x_flat = np.clip(x_train.reshape(x_train.shape[0], -1), 0, None)
        self._model.fit(x_flat)
        x_hat = self.reconstruct(x_train)
        metrics = _stride_metrics(x_train, x_hat)
        metrics["reconstruction_err"] = float(self._model.reconstruction_err_)
        if x_val is not None:
            val_metrics = _stride_metrics(x_val, self.reconstruct(x_val))
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        return metrics

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._model.transform(np.clip(x.reshape(x.shape[0], -1), 0, None))

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        latent = self.transform(x)
        x_hat_flat = latent @ self._model.components_
        return x_hat_flat.reshape(-1, self.stride_len, self.n_channels)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)


# ---------------------------------------------------------------------------
# PyTorch building blocks
# ---------------------------------------------------------------------------


def _compute_conv_output_len(input_len: int) -> int:
    """Compute the temporal length after the encoder's conv layers.

    Conv1d(k=7, s=2, p=3): L_out = floor((L_in + 2*3 - 7) / 2) + 1
    Conv1d(k=5, s=2, p=2): L_out = floor((L_in + 2*2 - 5) / 2) + 1
    """
    l1 = (input_len + 2 * 3 - 7) // 2 + 1  # conv1
    l2 = (l1 + 2 * 2 - 5) // 2 + 1         # conv2
    return l2


class _StrideEncoder(nn.Module):
    """1D conv encoder: (batch, stride_len, n_channels) → (batch, latent_dim)."""

    def __init__(self, n_channels: int, latent_dim: int, stride_len: int = STRIDE_LEN):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)

        conv_out_len = _compute_conv_output_len(stride_len)
        flat_size = 64 * conv_out_len

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self._conv_out_len = conv_out_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, stride_len, n_channels) → channels first for Conv1d
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)


class _StrideDecoder(nn.Module):
    """1D transposed conv decoder: (batch, latent_dim) → (batch, stride_len, n_channels).

    Uses sigmoid on final output to constrain reconstructed EMG to [0, 1].
    """

    def __init__(self, latent_dim: int, n_channels: int, stride_len: int = STRIDE_LEN):
        super().__init__()
        conv_out_len = _compute_conv_output_len(stride_len)
        flat_size = 64 * conv_out_len

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, flat_size),
            nn.ReLU(),
        )

        # Mirror the encoder: ConvTranspose1d reverses Conv1d
        # We need to recover stride_len exactly, so we compute output_padding.
        # conv1: L_in=stride_len → L_out1 = (stride_len + 2*3 - 7)//2 + 1
        # deconv2 needs to go from conv_out_len → L_out1
        l_after_conv1 = (stride_len + 2 * 3 - 7) // 2 + 1
        # output_padding for deconv2: L_out1 - ((conv_out_len - 1)*2 - 2*2 + 5)
        deconv2_out = (conv_out_len - 1) * 2 - 2 * 2 + 5
        op2 = l_after_conv1 - deconv2_out

        # deconv1 needs to go from l_after_conv1 → stride_len
        deconv1_out = (l_after_conv1 - 1) * 2 - 2 * 3 + 7
        op1 = stride_len - deconv1_out

        self.deconv2 = nn.ConvTranspose1d(
            64, 32, kernel_size=5, stride=2, padding=2, output_padding=op2,
        )
        self.deconv1 = nn.ConvTranspose1d(
            32, n_channels, kernel_size=7, stride=2, padding=3, output_padding=op1,
        )
        self._conv_out_len = conv_out_len

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.shape[0], 64, self._conv_out_len)
        x = torch.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv1(x))  # constrain to [0, 1]
        return x.transpose(1, 2)  # → (batch, stride_len, n_channels)


class _StrideVAEEncoder(nn.Module):
    """VAE encoder with conv backbone, outputs mu and log_var."""

    def __init__(self, n_channels: int, latent_dim: int, stride_len: int = STRIDE_LEN):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)

        conv_out_len = _compute_conv_output_len(stride_len)
        flat_size = 64 * conv_out_len

        self.fc_shared = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        h = self.fc_shared(x)
        return self.fc_mu(h), self.fc_log_var(h)


# ---------------------------------------------------------------------------
# Shared PyTorch base
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _StrideModelBase(EMGModel):
    """Common functionality for stride-level PyTorch models."""

    # Early stopping: stop if val loss hasn't improved for this many val checks.
    # Checks happen every 5 epochs, so patience=15 = 75 epochs without improvement.
    EARLY_STOP_PATIENCE: int = 15

    def __init__(self, latent_dim: int, n_channels: int = 11, stride_len: int = STRIDE_LEN):
        super().__init__(latent_dim, n_channels)
        self.stride_len = stride_len
        self.device = _get_device()
        self._module: nn.Module | None = None

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _make_loader(self, x: np.ndarray, batch_size: int) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(self._to_tensor(x))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _check_early_stop(
        self, val_loss: float, best_val_loss: float, no_improve_count: int,
    ) -> tuple[float, int, bool]:
        """Check early stopping condition. Returns (best_loss, count, should_stop)."""
        if val_loss < best_val_loss:
            return val_loss, 0, False
        no_improve_count += 1
        return best_val_loss, no_improve_count, no_improve_count >= self.EARLY_STOP_PATIENCE

    @staticmethod
    def _print_epoch(epoch: int, epochs: int, metrics: dict[str, float]) -> None:
        """Print a compact progress line."""
        parts = [f"Epoch {epoch+1:>3d}/{epochs}"]
        for k in ["train_loss", "val_mse", "val_r2"]:
            if k in metrics:
                parts.append(f"{k}={metrics[k]:.5f}")
        print(f"  {' | '.join(parts)}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._module.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self._module.load_state_dict(state)


# ---------------------------------------------------------------------------
# Stride Autoencoder
# ---------------------------------------------------------------------------


class StrideAutoencoder(_StrideModelBase):
    """Stride-level autoencoder with 1D convolutions + Lnorm.

    Loss = MSE(stride, stride_hat) + lnorm_weight * Lnorm(z)
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
        lnorm_weight: float = 0.01,
    ):
        super().__init__(latent_dim, n_channels, stride_len)
        self.lnorm_weight = lnorm_weight
        self.encoder = _StrideEncoder(n_channels, latent_dim, stride_len)
        self.decoder = _StrideDecoder(latent_dim, n_channels, stride_len)
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})
        self._module.to(self.device)

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 150,
        batch_size: int = 256,
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
        no_improve = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                z = self.encoder(batch)
                x_hat = self.decoder(z)
                recon_loss = nn.functional.mse_loss(x_hat, batch)
                loss = recon_loss + self.lnorm_weight * lnorm_loss(z)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._module.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= len(loader.dataset)

            step_metrics = {"train_loss": epoch_loss}
            if x_val is not None and (epoch % 5 == 0 or epoch == epochs - 1):
                val_hat = self.reconstruct(x_val)
                val_metrics = _stride_metrics(x_val, val_hat)
                step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                val_loss = val_metrics["mse"]
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = deepcopy(self._module.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                self._print_epoch(epoch, epochs, step_metrics)
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            step_metrics["lr"] = optimizer.param_groups[0]["lr"]
            if log_fn is not None:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)

        self._module.eval()
        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            val_metrics = _stride_metrics(x_val, self.reconstruct(x_val))
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


# ---------------------------------------------------------------------------
# Stride VAE
# ---------------------------------------------------------------------------


class StrideVAE(_StrideModelBase):
    """Stride-level VAE with 1D conv encoder and β·KL + Lnorm.

    Loss = MSE + β * KL + lnorm_weight * Lnorm(mu)

    Default β=0.1 (mild regularization) to prioritize reconstruction.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
        beta: float = 0.1,
        lnorm_weight: float = 0.01,
    ):
        super().__init__(latent_dim, n_channels, stride_len)
        self.beta = beta
        self.lnorm_weight = lnorm_weight
        self.encoder = _StrideVAEEncoder(n_channels, latent_dim, stride_len)
        self.decoder = _StrideDecoder(latent_dim, n_channels, stride_len)
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
        epochs: int = 150,
        batch_size: int = 256,
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
        no_improve = 0

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
                nn.utils.clip_grad_norm_(self._module.parameters(), max_norm=1.0)
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
            if x_val is not None and (epoch % 5 == 0 or epoch == epochs - 1):
                val_hat = self.reconstruct(x_val)
                val_metrics = _stride_metrics(x_val, val_hat)
                step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                val_loss = val_metrics["mse"]
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = deepcopy(self._module.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                self._print_epoch(epoch, epochs, step_metrics)
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            step_metrics["lr"] = optimizer.param_groups[0]["lr"]
            if log_fn is not None:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)

        self._module.eval()
        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            val_metrics = _stride_metrics(x_val, self.reconstruct(x_val))
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


# ---------------------------------------------------------------------------
# Stride MAE — block masking for temporal dynamics
# ---------------------------------------------------------------------------


class StrideWAE_MMD(_StrideModelBase):
    """Stride-level WAE-MMD: 1D conv encoder/decoder + MMD penalty.

    Loss = MSE(stride, stride_hat) + mmd_weight * MMD(q(z), N(0,I)) + lnorm_weight * Lnorm(z)

    Uses the same conv backbone as StrideAutoencoder but replaces the
    deterministic-only training with an MMD regularizer that matches the
    latent distribution to a Gaussian prior. This gives a smooth, well-covered
    latent space (like VAE) with sharper reconstructions and better distance
    preservation (optimal transport).
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
        mmd_weight: float = 10.0,
        lnorm_weight: float = 0.01,
    ):
        super().__init__(latent_dim, n_channels, stride_len)
        self.mmd_weight = mmd_weight
        self.lnorm_weight = lnorm_weight
        self.encoder = _StrideEncoder(n_channels, latent_dim, stride_len)
        self.decoder = _StrideDecoder(latent_dim, n_channels, stride_len)
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})
        self._module.to(self.device)

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 150,
        batch_size: int = 256,
        lr: float = 1e-3,
        log_fn: LogFn = None,
    ) -> dict[str, float]:
        from leaps.models.pytorch_models import _compute_mmd

        loader = self._make_loader(x_train, batch_size)
        self._module.train()
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

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
                nn.utils.clip_grad_norm_(self._module.parameters(), max_norm=1.0)
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
            if x_val is not None and (epoch % 5 == 0 or epoch == epochs - 1):
                val_hat = self.reconstruct(x_val)
                val_metrics = _stride_metrics(x_val, val_hat)
                step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                val_loss = val_metrics["mse"]
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = deepcopy(self._module.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                self._print_epoch(epoch, epochs, step_metrics)
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            step_metrics["lr"] = optimizer.param_groups[0]["lr"]
            if log_fn is not None:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)

        self._module.eval()
        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            val_metrics = _stride_metrics(x_val, self.reconstruct(x_val))
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


class StrideFlatAE(_StrideModelBase):
    """Flatten-MLP Autoencoder adapted from Hausdörfer et al. (2024).

    Pipeline:
        1. Flatten stride (101, 11) → 1111-dim vector
        2. Encoder: 1111 → 512 → ReLU → 256 → ReLU → latent_dim → Tanh
        3. Decoder: latent_dim → 256 → ReLU → 512 → ReLU → 1111 → Sigmoid
        4. Reshape back to (101, 11)

    Design choices:
        - Tanh on latent: bounds z to [-1, 1] for RL policy compatibility.
          The RL policy will output z directly via tanh activation.
        - Sigmoid on output: constrains reconstructions to [0, 1], matching
          the range of preprocessed EMG activations.
        - MSE loss only: no regularization beyond the architecture itself.
          Hausdörfer uses Lnorm but tanh already handles bounding.
        - Two hidden layers (512, 256): one more than Hausdörfer's original
          (which has action_dim ~30). We need the extra capacity for our
          1111-dim input, but keep it simple — no batch norm, no dropout.

    For RL integration:
        - decode(z) maps z ∈ [-1,1]^latent_dim → activations ∈ [0,1]^1111
        - Freeze decoder weights, let policy learn which z to output
        - Optionally blend with residual: a = (1-w)*decode(z) + w*a_residual
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
    ):
        super().__init__(latent_dim, n_channels, stride_len)
        flat_dim = stride_len * n_channels  # 101 * 11 = 1111

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, flat_dim),
        )
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})
        self._module.to(self.device)

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        log_fn: LogFn = None,
    ) -> dict[str, float]:
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._to_tensor(x_train_flat)),
            batch_size=batch_size, shuffle=True,
        )
        self._module.train()
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                z = self.encoder(batch)
                x_hat = self.decoder(z)
                loss = nn.functional.mse_loss(x_hat, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= len(loader.dataset)

            step_metrics = {"train_loss": epoch_loss}
            if x_val is not None and (epoch % 5 == 0 or epoch == epochs - 1):
                val_hat = self.reconstruct(x_val)
                val_metrics = _stride_metrics(x_val, val_hat)
                step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                val_loss = val_metrics["mse"]
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = deepcopy(self._module.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                self._print_epoch(epoch, epochs, step_metrics)
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            step_metrics["lr"] = optimizer.param_groups[0]["lr"]
            if log_fn is not None:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)

        self._module.eval()
        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            val_metrics = _stride_metrics(x_val, self.reconstruct(x_val))
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        x_flat = self._to_tensor(x.reshape(x.shape[0], -1))
        return self.encoder(x_flat).cpu().numpy()

    @torch.no_grad()
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent z → muscle activations clipped to [0, 1].

        This is the RL interface: policy outputs z ∈ [-1,1]^latent_dim,
        frozen decoder produces muscle activations for the humanoid.
        Output shape: (batch, 1111) or (1111,) matching input z shape.
        """
        self._module.eval()
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        return np.clip(self.decoder(z_t).cpu().numpy(), 0.0, 1.0)

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        x_flat = self._to_tensor(x.reshape(x.shape[0], -1))
        z = self.encoder(x_flat)
        x_hat = np.clip(self.decoder(z).cpu().numpy(), 0.0, 1.0)
        return x_hat.reshape(-1, self.stride_len, self.n_channels)


class StrideMaskedAutoencoder(_StrideModelBase):
    """Stride-level Masked Autoencoder — masks contiguous time BLOCKS.

    Unlike the snapshot MAE (which masks random channels), this masks
    contiguous blocks of timepoints. This is more challenging because
    the conv encoder can't fill in gaps from immediate neighbors — it
    must learn longer-range temporal dynamics.

    For example, with block_size=15 and mask_ratio=0.3, we mask ~2
    contiguous blocks of 15 timepoints each (30 out of 101 total).
    Loss is computed only on masked timepoints.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
        mask_ratio: float = 0.3,
        block_size: int = 15,
        lnorm_weight: float = 0.01,
    ):
        super().__init__(latent_dim, n_channels, stride_len)
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.lnorm_weight = lnorm_weight
        self.encoder = _StrideEncoder(n_channels, latent_dim, stride_len)
        self.decoder = _StrideDecoder(latent_dim, n_channels, stride_len)
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})
        self._module.to(self.device)

    def _generate_block_mask(self, batch_size: int) -> torch.Tensor:
        """Generate contiguous block mask: 1 = keep, 0 = masked.

        Masks contiguous blocks of `block_size` timepoints. Each block
        is placed at a random start position. Number of blocks chosen
        so ~mask_ratio of timepoints are masked.

        Shape: (batch_size, stride_len, 1) — broadcasts over channels.
        """
        n_total_mask = max(self.block_size, int(self.stride_len * self.mask_ratio))
        n_blocks = max(1, n_total_mask // self.block_size)

        mask = torch.ones(batch_size, self.stride_len, 1, device=self.device)
        for _ in range(n_blocks):
            # Random start position for each sample in the batch
            max_start = self.stride_len - self.block_size
            starts = torch.randint(0, max_start + 1, (batch_size,), device=self.device)
            for b in range(batch_size):
                mask[b, starts[b]:starts[b] + self.block_size, :] = 0.0
        return mask

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 150,
        batch_size: int = 256,
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
        no_improve = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                mask = self._generate_block_mask(batch.shape[0])
                masked_input = batch * mask
                z = self.encoder(masked_input)
                x_hat = self.decoder(z)
                # Loss only on masked timepoints (all channels at those times)
                inv_mask = 1.0 - mask
                n_masked = inv_mask.sum() * self.n_channels
                if n_masked > 0:
                    recon_loss = (((x_hat - batch) ** 2) * inv_mask).sum() / n_masked
                else:
                    recon_loss = nn.functional.mse_loss(x_hat, batch)
                loss = recon_loss + self.lnorm_weight * lnorm_loss(z)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._module.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= len(loader.dataset)

            step_metrics = {"train_loss": epoch_loss}
            if x_val is not None and (epoch % 5 == 0 or epoch == epochs - 1):
                val_hat = self.reconstruct(x_val)
                val_metrics = _stride_metrics(x_val, val_hat)
                step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                val_loss = val_metrics["mse"]
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = deepcopy(self._module.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                self._print_epoch(epoch, epochs, step_metrics)
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            step_metrics["lr"] = optimizer.param_groups[0]["lr"]
            if log_fn is not None:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)

        self._module.eval()
        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            val_metrics = _stride_metrics(x_val, self.reconstruct(x_val))
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _stride_metrics(x: np.ndarray, x_hat: np.ndarray) -> dict[str, float]:
    """Compute MSE and R² for 3D stride arrays by flattening to 2D."""
    x_flat = x.reshape(-1, x.shape[-1])
    x_hat_flat = x_hat.reshape(-1, x_hat.shape[-1])
    return compute_train_metrics(x_flat, x_hat_flat)
