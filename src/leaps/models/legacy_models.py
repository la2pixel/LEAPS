"""LEAPS action-representation models.

Stuff that I tried last year

  - PCAModel/NMFModel and StridePCAModel/StrideNMFModel were near-identical,
    differing only in a stride <-> flat-vector reshape. Merged into one
    PCAModel/NMFModel pair, parametrised by an optional `stride_len`.
  - lnorm_loss (soft wall) and _lnorm_loss_exact (dead-zone wall) were two
    copies of the same penalty. Merged into one lnorm_loss(z, dead_zone=...).
  - StrideFlatAE/StrideFlatVAE shared a torch training-infra base
    (_StrideModelBase) that HausdorferAE duplicated instead of reusing.
    HausdorferAE now shares _TorchModelBase too (tensor conversion, loader
    construction, save/load, freeze, get_decoder_module, epoch logging);
    each model still owns its own fit() since the training-loop details
    genuinely differ (VAE's KL term, HausdorferAE's checkpoint-on-improve).

Two paradigms, preserved as-is:

Snapshot-level -- HausdorferAE
  Input:  one action frame (action_dim,) -- individual timestep.
  Use:    exact replication of Hausdoerfer et al. 2024.

Stride-level -- everything else
  Input:  full gait cycle (101, 11) -- time-normalised frames x EMG channels.
  Use:    policy outputs z per stride; frozen decoder -> 101-frame activation profile.

Decoder output: linear (no sigmoid). Caller clips to [0, 1] before actuator use. Lnorm penalty: soft wall keeping latent inside +/-1.2 (Hausdoerfer et al. 2024).

Dont use this script now :(
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import NMF, PCA

from leaps.models.metrics import compute_train_metrics

# py3.9 can't do `Callable[...] | None` as a real (non-annotation) expression --
# Optional() is the same thing, just the version that works at runtime here.
LogFn = Optional[Callable[[dict[str, float], int], None]]

STRIDE_LEN = 101  #time normalized strides


class ActionModel(ABC):
    """base interface for all action representation models.
       all models compress 11-channel activations to a latent_dim-dimensional
       representation that reconstructs original signal.
    """

    def __init__(self, latent_dim: int, n_channels: int = 11):
        self.latent_dim = latent_dim
        self.n_channels = n_channels

    @abstractmethod
    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        """model training.
        args:
            x_train: shape (N, n_channels).
            x_val: shape (M, n_channels) which is optional

        returns:
            dict of training metrics
        """

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """encodes data to latent space
        args:
            x: i/p(N, n_channels).

        returns:
           latent representation, shape (N, latent_dim).
        """

    @abstractmethod
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """encode, decode data

        args:
            x: i/p(N, n_channels).
        returns:
            reconstructed signal(N, n_channels).
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model from disk."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── Losses ───────────────────────────────────────────────────────────────────


def lnorm_loss(latent: torch.Tensor, dead_zone: float = 0.0) -> torch.Tensor:
    """Penalty pushing latent values into [-1, 1] (Hausdoerfer et al. 2024).

    Per-element: exp((|z|/1.2)^10) - 1, i.e. ~0 near the origin and a steep
    wall approaching |z|=1.2.

    dead_zone=0.0 (default): soft wall from the first moment, used by
        StrideFlatAE/StrideFlatVAE.
    dead_zone=0.8: exact paper version used by HausdorferAE -- penalty is
        masked to 0 for |z| < 0.8, letting the RL policy move freely inside
        the latent ball before the wall engages.
    """
    abs_z = torch.abs(latent)
    penalty = torch.exp(torch.pow(abs_z / 1.2, 10).clamp(max=50.0)) - 1.0
    if dead_zone > 0:
        penalty = penalty.masked_fill(abs_z < dead_zone, 0.0)
    return penalty.mean()


def _flatten_metrics(x: np.ndarray, x_hat: np.ndarray) -> dict[str, float]:
    """MSE/R2 on the trailing channel dim, regardless of 2D (N, C) or 3D (N, T, C) input."""
    return compute_train_metrics(
        x.reshape(-1, x.shape[-1]),
        x_hat.reshape(-1, x_hat.shape[-1]),
    )


# ── Sklearn baselines: PCA / NMF (snapshot- or stride-level) ────────────────


class PCAModel(ActionModel):
    """PCA dimensionality reduction for EMG signals.

    Snapshot-level by default (operates on (N, n_channels) frames). Pass
    stride_len to operate on (N, stride_len, n_channels) strides instead --
    each stride is flattened to a stride_len*n_channels vector before PCA and
    unflattened after inverse_transform.

    PCA is the theoretical upper bound for linear compression: it maximises
    explained variance with a closed-form optimal solution. Use it as a
    ceiling -- if a neural model can't beat PCA R^2 at the same latent_dim,
    training failed.

    Note: PCA output is unconstrained. Reconstructions can be slightly
    negative for channels that are mostly silent.
    """

    def __init__(self, latent_dim: int, n_channels: int = 11, stride_len: int | None = None):
        super().__init__(latent_dim, n_channels)
        self.stride_len = stride_len
        self._model = PCA(n_components=latent_dim)

    def _flat(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1) if self.stride_len else x

    def _unflat(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(-1, self.stride_len, self.n_channels) if self.stride_len else x

    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        self._model.fit(self._flat(x_train))
        metrics = _flatten_metrics(x_train, self.reconstruct(x_train))
        metrics["explained_variance_ratio"] = float(
            np.sum(self._model.explained_variance_ratio_)
        )
        if x_val is not None:
            metrics.update(
                {f"val_{k}": v for k, v in _flatten_metrics(x_val, self.reconstruct(x_val)).items()}
            )
        return metrics

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._model.transform(self._flat(x))

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self._unflat(self._model.inverse_transform(self.transform(x)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)


class NMFModel(ActionModel):
    """Non-Negative Matrix Factorization for muscle synergy extraction.

    Snapshot-level by default. Pass stride_len to factorise flattened strides
    instead (X ~= W @ H, W/H >= 0 -- W is per-stride activation weights, H is
    basis strides / synergy templates).

    Non-negativity is biologically grounded: EMG can't go negative. The k
    components are interpretable as co-activation templates (push-off
    synergy, swing synergy, etc.).
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int | None = None,
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

    def _flat(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1) if self.stride_len else x

    def _unflat(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(-1, self.stride_len, self.n_channels) if self.stride_len else x

    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        x_clipped = np.clip(self._flat(x_train), 0, None)
        self._model.fit(x_clipped)
        metrics = _flatten_metrics(x_train, self.reconstruct(x_train))
        metrics["reconstruction_err"] = float(self._model.reconstruction_err_)
        if x_val is not None:
            metrics.update(
                {f"val_{k}": v for k, v in _flatten_metrics(x_val, self.reconstruct(x_val)).items()}
            )
        return metrics

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._model.transform(np.clip(self._flat(x), 0, None))

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self._unflat(self.transform(x) @ self._model.components_)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)


class StridePCAModel(PCAModel):
    """PCA on flattened strides -- thin alias of PCAModel with stride_len required."""

    def __init__(self, latent_dim: int, n_channels: int = 11, stride_len: int = STRIDE_LEN):
        super().__init__(latent_dim, n_channels, stride_len=stride_len)


class StrideNMFModel(NMFModel):
    """NMF on flattened strides -- thin alias of NMFModel with stride_len required."""

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
        max_iter: int = 500,
        seed: int = 42,
    ):
        super().__init__(latent_dim, n_channels, stride_len=stride_len, max_iter=max_iter, seed=seed)


# ── Shared PyTorch infra ─────────────────────────────────────────────────────


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchModelBase(ActionModel):
    """Common infrastructure for all torch-based models (stride- and snapshot-level).

    Early stopping: val loss checked every VAL_CHECK_INTERVAL epochs; stops if
    no improvement for EARLY_STOP_PATIENCE consecutive checks.
    """

    EARLY_STOP_PATIENCE: int = 15
    VAL_CHECK_INTERVAL: int = 5

    def __init__(self, latent_dim: int, n_channels: int = 11):
        super().__init__(latent_dim, n_channels)
        self.device = _get_device()
        self._module: nn.Module | None = None

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _make_loader(self, x_flat: np.ndarray, batch_size: int) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._to_tensor(x_flat)),
            batch_size=batch_size,
            shuffle=True,
        )

    def _run_val(
        self,
        x_val: np.ndarray,
        best_val_loss: float,
        best_state: dict | None,
        no_improve: int,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> tuple[float, dict | None, int]:
        """Val loss is pure MSE (not full training loss) to track reconstruction on unseen data."""
        val_hat = self.reconstruct(x_val)
        val_loss = float(np.mean((x_val - val_hat) ** 2))
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            return val_loss, deepcopy(self._module.state_dict()), 0
        return best_val_loss, best_state, no_improve + 1

    @staticmethod
    def _log(epoch: int, epochs: int, metrics: dict[str, float]) -> None:
        parts = [f"Epoch {epoch + 1:>3d}/{epochs}"]
        for k in ["train_loss", "val_mse", "val_r2"]:
            if k in metrics:
                parts.append(f"{k}={metrics[k]:.5f}")
        print(f"  {' | '.join(parts)}")

    def freeze(self) -> None:
        """Freeze all parameters -- call before RL training."""
        for p in self._module.parameters():
            p.requires_grad_(False)
        self._module.eval()

    def get_decoder_module(self) -> nn.Module:
        """Return the decoder submodule for embedding directly in an RL agent."""
        return self.decoder

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._module.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self._module.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )


# ── StrideFlatAE ───────────────────────────────────────────────────────────


class StrideFlatAE(_TorchModelBase):
    """Flatten-MLP autoencoder faithful to Hausdoerfer et al. (2024).

    Loss = MSE(x, x_hat) + lnorm_weight * Lnorm(z)

    Architecture:
        Encoder: Flatten(1111) -> Linear(512) -> Tanh -> Linear(256) -> Tanh -> Linear(latent)
        Decoder: Linear(256)   -> Tanh -> Linear(512) -> Tanh -> Linear(1111) -> Reshape(101,11)

    Tanh (not ReLU): paper uses tanh throughout; compatible with Lnorm range
    and Xavier init. Decoder output is linear -- caller clips to [0, 1] before
    actuator use.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
        lnorm_weight: float = 0.01,
    ):
        super().__init__(latent_dim, n_channels)
        self.stride_len = stride_len
        self.lnorm_weight = lnorm_weight
        flat_dim = stride_len * n_channels

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.Tanh(),
            nn.Linear(512, 256),      nn.Tanh(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.Tanh(),
            nn.Linear(256, 512),        nn.Tanh(),
            nn.Linear(512, flat_dim),
        )
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})

        # Xavier uniform init is designed for tanh (Kaiming uniform, the PyTorch default, targets ReLU).
        for m in self._module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

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
        x_flat = x_train.reshape(x_train.shape[0], -1)
        loader = self._make_loader(x_flat, batch_size)
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )
        best_val_loss, best_state, no_improve = float("inf"), None, 0

        self._module.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                z = self.encoder(batch)
                x_hat = self.decoder(z)
                loss = nn.functional.mse_loss(x_hat, batch) + self.lnorm_weight * lnorm_loss(z)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._module.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= len(loader.dataset)

            step_metrics = {"train_loss": epoch_loss, "lr": optimizer.param_groups[0]["lr"]}
            if x_val is not None and (epoch % self.VAL_CHECK_INTERVAL == 0 or epoch == epochs - 1):
                best_val_loss, best_state, no_improve = self._run_val(
                    x_val, best_val_loss, best_state, no_improve, scheduler,
                )
                step_metrics.update(_flatten_metrics(x_val, self.reconstruct(x_val)))
                self._log(epoch, epochs, {**step_metrics, "val_mse": best_val_loss})
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
            if log_fn:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)
        self._module.eval()

        metrics = _flatten_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            metrics.update({f"val_{k}": v for k, v in _flatten_metrics(x_val, self.reconstruct(x_val)).items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        return self.encoder(self._to_tensor(x.reshape(x.shape[0], -1))).cpu().numpy()

    @torch.no_grad()
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent z -> (batch, 1111) muscle activations (unbounded; caller clips to [0,1])."""
        self._module.eval()
        return self.decoder(self._to_tensor(z)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        z = self.encoder(self._to_tensor(x.reshape(x.shape[0], -1)))
        return self.decoder(z).cpu().numpy().reshape(-1, self.stride_len, self.n_channels)


# ── StrideFlatVAE ────────────────────────────────────────────────────────────


class StrideFlatVAE(_TorchModelBase):
    """Flatten-MLP VAE. Same MLP backbone as StrideFlatAE, adds reparameterisation + KL.

    Loss = MSE(x, x_hat) + beta * sum_i max(KL_i, free_bits) + lnorm_weight * Lnorm(mu)

    KL forces the aggregate posterior toward N(0,I), so z=0 always decodes to
    a valid stride -- useful for RL initialisation where the policy starts
    near z=0.

    free_bits (default 0.5 nats): prevents posterior collapse -- each latent
    dim is forced to encode at least free_bits nats regardless of beta. Only
    sweep beta.

    At RL inference, transform() returns mu -- deterministic, no sampling noise.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
        beta: float = 0.01,
        free_bits: float = 0.5,
        lnorm_weight: float = 0.01,
    ):
        super().__init__(latent_dim, n_channels)
        self.stride_len = stride_len
        self.beta = beta
        self.free_bits = free_bits
        self.lnorm_weight = lnorm_weight
        flat_dim = stride_len * n_channels

        self.encoder_body = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.Tanh(),
            nn.Linear(512, 256),      nn.Tanh(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.Tanh(),
            nn.Linear(256, 512),        nn.Tanh(),
            nn.Linear(512, flat_dim),
        )
        self._module = nn.ModuleDict({
            "encoder_body": self.encoder_body,
            "fc_mu": self.fc_mu,
            "fc_log_var": self.fc_log_var,
            "decoder": self.decoder,
        })
        for m in self._module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self._module.to(self.device)

    def _encode(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x_flat)
        return self.fc_mu(h), self.fc_log_var(h)

    def _reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        log_fn: LogFn = None,
    ) -> dict[str, float]:
        x_flat = x_train.reshape(x_train.shape[0], -1)
        loader = self._make_loader(x_flat, batch_size)
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )
        best_val_loss, best_state, no_improve = float("inf"), None, 0

        self._module.train()
        for epoch in range(epochs):
            epoch_loss = epoch_recon = epoch_kl = 0.0
            for (batch,) in loader:
                mu, log_var = self._encode(batch)
                z = self._reparameterise(mu, log_var)
                x_hat = self.decoder(z)

                recon = nn.functional.mse_loss(x_hat, batch)
                # free_bits clamps KL per dim to prevent posterior collapse (sigma->0).
                kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
                kl = kl_per_dim.clamp(min=self.free_bits).sum(dim=1).mean()
                loss = recon + self.beta * kl + self.lnorm_weight * lnorm_loss(mu)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._module.parameters(), max_norm=1.0)
                optimizer.step()
                n = batch.shape[0]
                epoch_loss += loss.item() * n
                epoch_recon += recon.item() * n
                epoch_kl += kl.item() * n

            n_total = len(loader.dataset)
            step_metrics = {
                "train_loss": epoch_loss / n_total,
                "train_recon": epoch_recon / n_total,
                "train_kl": epoch_kl / n_total,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if x_val is not None and (epoch % self.VAL_CHECK_INTERVAL == 0 or epoch == epochs - 1):
                best_val_loss, best_state, no_improve = self._run_val(
                    x_val, best_val_loss, best_state, no_improve, scheduler,
                )
                step_metrics.update({f"val_{k}": v for k, v in _flatten_metrics(x_val, self.reconstruct(x_val)).items()})
                self._log(epoch, epochs, step_metrics)
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
            if log_fn:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)
        self._module.eval()

        metrics = _flatten_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            metrics.update({f"val_{k}": v for k, v in _flatten_metrics(x_val, self.reconstruct(x_val)).items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return mu (posterior mean) -- deterministic, no sampling."""
        self._module.eval()
        mu, _ = self._encode(self._to_tensor(x.reshape(x.shape[0], -1)))
        return mu.cpu().numpy()

    @torch.no_grad()
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode z -> (batch, 1111) muscle activations (unbounded; caller clips to [0,1])."""
        self._module.eval()
        return self.decoder(self._to_tensor(z)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        mu, _ = self._encode(self._to_tensor(x.reshape(x.shape[0], -1)))
        return self.decoder(mu).cpu().numpy().reshape(-1, self.stride_len, self.n_channels)


# ── HausdorferAE -- snapshot-level, exact paper architecture ────────────────


class HausdorferAE(_TorchModelBase):
    """Snapshot-level MLP autoencoder -- exact Hausdoerfer et al. (2024) architecture.

    Input/output: one action frame (action_dim,) = (18,) for gait10dof18musc.
    Train on individual muscle activation snapshots, not full strides.

    Architecture (one hidden layer as in paper):
        Encoder: Linear(action_dim -> 2*latent_dim) -> Tanh -> Linear(2*latent_dim -> latent_dim)
        Decoder: Linear(latent_dim -> 2*latent_dim) -> Tanh -> Linear(2*latent_dim -> action_dim)

    Defaults for gait10dof18musc (18 muscles):
        action_dim = 18
        latent_dim = 9    (= action_dim // 2, per paper)
        hidden_dim = 18   (= 2 x latent_dim, per paper)

    Loss: MSE(a, a_hat) + lnorm_weight * Lnorm(z), with Lnorm's 0.8 dead zone
    (see lnorm_loss).

    Decoder output is linear (no sigmoid). decode(z) returns values in [0,1]
    physical space (muscle excitation space). The RL interface must scale to
    [-1,1] before env.step() using EMGToMuscleMapper.to_loco_action() -- which
    already handles this.

    ## Data space: train in [0, 1] (physical EMG space)
    Muscle excitations are non-negative by definition. The AE trains in this
    physical space so metrics (MSE, R^2) are interpretable and the pipeline
    stays consistent with EMGToMuscleMapper. The Lnorm penalty acts on the
    latent z, not on the data space -- so the choice of data range doesn't
    affect Lnorm behaviour.

    ## How to prepare training data
    EMG data lives as strides (N_strides, 101, 11). Map to 18 muscles via
    EMGToMuscleMapper, then flatten the time axis:
        mapped = mapper.map_batch(strides.reshape(-1, 11)).reshape(N, 101, 18)
        snapshots = mapped.reshape(-1, 18)    # (N*101, 18) in [0,1] -- train on this

    ## RL interface (after pre-training + freeze)
        z = policy(obs)                               # (latent_dim,) from policy head
        muscle_act = np.clip(ae.decode(z), 0, 1)     # (action_dim,) in [0,1]
        loco_action = mapper.to_loco_action(muscle_act)   # scale to [-1,1] for env.step()
        # DefaultControl maps [-1,1] -> ctrl [0,1] for muscles automatically.
    """

    EARLY_STOP_PATIENCE: int = 10
    VAL_CHECK_INTERVAL: int = 5

    def __init__(
        self,
        action_dim: int = 18,
        latent_dim: int | None = None,
        lnorm_weight: float = 1.0,
    ):
        if latent_dim is None:
            latent_dim = action_dim // 2  # paper default
        super().__init__(latent_dim=latent_dim, n_channels=action_dim)
        self.action_dim = action_dim
        self.lnorm_weight = lnorm_weight
        hidden_dim = 2 * latent_dim  # paper: one hidden layer, size 2*latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),  # linear out -- caller clamps to [0,1]
        )
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})

        for m in self._module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # designed for tanh
                nn.init.zeros_(m.bias)

        self._module.to(self.device)

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 300,
        batch_size: int = 4096,
        lr: float = 3e-4,
        log_fn: LogFn = None,
        ckpt_path: Path | str | None = None,
        min_delta: float = 1e-5,
    ) -> dict[str, float]:
        """Train on action snapshots (N, action_dim).

        Args:
            x_train:    (N, action_dim) muscle activations in [0, 1].
            x_val:      Validation split for early stopping.
            epochs:     300 is sufficient -- the model is tiny (~1k params).
            batch_size: 4096 (large batches stabilise Lnorm, matching paper).
            lr:         3e-4 (Adam, paper default).
            log_fn:     Optional callback(metrics_dict, epoch) for W&B/TensorBoard.
            ckpt_path:  If given, best model is saved here whenever val_mse improves.
            min_delta:  Minimum improvement in val_mse to reset the early-stop counter.
        """
        pin = self.device.type == "cuda"
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.as_tensor(x_train, dtype=torch.float32)
            ),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin,
            num_workers=4,
            persistent_workers=True,
        )
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )
        best_val_loss, best_state, no_improve = float("inf"), None, 0

        self._module.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device, non_blocking=pin)
                z = self.encoder(batch)
                a_hat = self.decoder(z)
                loss = nn.functional.mse_loss(a_hat, batch) + self.lnorm_weight * lnorm_loss(z, dead_zone=0.8)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= len(loader.dataset)

            step_metrics: dict[str, float] = {
                "train_loss": epoch_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if x_val is not None and (epoch % self.VAL_CHECK_INTERVAL == 0 or epoch == epochs - 1):
                val_mse, val_r2 = self._eval_metrics(x_val)
                scheduler.step(val_mse)
                step_metrics.update({"val_mse": val_mse, "val_r2": val_r2})
                if val_mse < best_val_loss - min_delta:
                    best_val_loss = val_mse
                    best_state = deepcopy(self._module.state_dict())
                    no_improve = 0
                    if ckpt_path is not None:
                        self.save(ckpt_path)
                else:
                    no_improve += 1
                if epoch % (self.VAL_CHECK_INTERVAL * 4) == 0:
                    print(
                        f"  Epoch {epoch + 1:>3d}/{epochs}"
                        f" | train_loss={epoch_loss:.5f}"
                        f" | val_mse={val_mse:.5f}"
                        f" | val_r2={val_r2:.4f}"
                    )
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1} (patience={self.EARLY_STOP_PATIENCE} x {self.VAL_CHECK_INTERVAL} = {self.EARLY_STOP_PATIENCE * self.VAL_CHECK_INTERVAL} epochs, min_delta={min_delta})")
                    break
                self._module.train()
            if log_fn:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)
        self._module.eval()

        train_mse, train_r2 = self._eval_metrics(x_train)
        metrics = {"mse": train_mse, "r2": train_r2}
        if x_val is not None:
            val_mse, val_r2 = self._eval_metrics(x_val)
            metrics.update({"val_mse": val_mse, "val_r2": val_r2})
        return metrics

    @torch.no_grad()
    def _eval_metrics(self, x: np.ndarray) -> tuple[float, float]:
        """Return (mse, r2) on x."""
        self._module.eval()
        x_hat = self.reconstruct(x)
        mse = float(np.mean((x - x_hat) ** 2))
        ss_tot = np.sum((x - x.mean()) ** 2)
        r2 = float(1 - np.sum((x - x_hat) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
        return mse, r2

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Encode (N, action_dim) -> (N, latent_dim)."""
        self._module.eval()
        return self.encoder(self._to_tensor(x)).cpu().numpy()

    @torch.no_grad()
    def decode(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """Decode latent z -> muscle activations in [0,1] physical space.

        Caller must scale to [-1,1] before env.step():
            muscle_act = np.clip(ae.decode(z), 0, 1)
            loco_action = mapper.to_loco_action(muscle_act)   # 2*a - 1
            env.step(loco_action)
        """
        self._module.eval()
        if isinstance(z, np.ndarray):
            z = self._to_tensor(z)
        return self.decoder(z.to(self.device)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Encode then decode: (N, action_dim) -> (N, action_dim)."""
        self._module.eval()
        return self.decoder(self.encoder(self._to_tensor(x))).cpu().numpy()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._module.state_dict(),
                "action_dim": self.action_dim,
                "latent_dim": self.latent_dim,
                "lnorm_weight": self.lnorm_weight,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._module.load_state_dict(ckpt["state_dict"])
        self._module.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "HausdorferAE":
        """Load from a checkpoint saved by save() -- infers config from file."""
        device = _get_device()
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            action_dim=ckpt["action_dim"],
            latent_dim=ckpt["latent_dim"],
            lnorm_weight=ckpt["lnorm_weight"],
        )
        model._module.load_state_dict(ckpt["state_dict"])
        model._module.eval()
        return model

    def __repr__(self) -> str:
        return (
            f"HausdorferAE("
            f"action_dim={self.action_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"hidden_dim={2 * self.latent_dim}, "
            f"lnorm_weight={self.lnorm_weight})"
        )
