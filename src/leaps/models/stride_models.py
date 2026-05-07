"""Action representation models for LEAPS.

## Two paradigms

### Snapshot-level — HausdorferAE (bottom of this file)
Input: one action frame (18,) — individual timestep, 18 muscle activations.
Use:   exact replication of Hausdörfer et al. 2024. Policy outputs z per
       timestep → frozen decoder → muscle commands that frame.
Start here — simplest to debug.

### Stride-level — everything else
Input: full gait cycle (101, 11) — 101 time-normalised frames × 11 EMG channels.
Use:   policy outputs one z per stride; decoder returns 101-frame activation
       profile that gets phase-indexed at each sim timestep.
Architecturally cleaner for gait, but harder to debug.

## Decoder output convention (all models)

**Linear output — no sigmoid.**
EMG after preprocessing is in [0, 1]. Sigmoid saturates gradients near 0 and 1
(where most EMG lives), causing training stalls. Callers must clamp outputs to
[0, 1] before passing to the muscle actuator. EMGToMuscleMapper does this.

## Lnorm penalty (Hausdörfer et al. 2024)

Soft wall keeping latent z inside ±1.2:
  L(z_i) = 0                          if |z_i| < 0.8
          = exp((z_i / 1.2)^10) - 1   otherwise
The 0.8 dead-zone gives the RL policy room to explore. The wall at ±1.2
is steep (exponent 10) but differentiable — no hard clipping.

## Stride-level models

Sklearn baselines (closed-form, no GPU):
  - StridePCAModel  — linear, variance-maximising upper bound
  - StrideNMFModel  — non-negative, biologically interpretable synergies
  - StrideCNMFModel — NMF + L2 penalty on activations

PyTorch — Conv1D backbone (StrideAutoencoder, StrideVAE, StrideWAE_MMD, StrideMaskedAutoencoder):
  - Encoder: Conv1d(11→32, k=7, s=2) → Conv1d(32→64, k=5, s=2) → FC → latent
  - Decoder: FC → ConvTranspose1d → ConvTranspose1d → (101, 11)

PyTorch — MLP backbone (StrideFlatAE, StrideFlatVAE):
  - Flattens (101, 11) → 1111, two hidden layers (512, 256), tanh, Xavier init.
  - No temporal inductive bias. Current best: StrideFlatAE d=8 (R²=0.634).

## Training recommendation (stride models)

Minimum 200 epochs. ReduceLROnPlateau checks every 5 epochs (patience=10 = 50
epochs before LR halves). Early stopping at 15 checks = 75 epochs.
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

LogFn = Callable[[dict[str, float], int], None] | None

STRIDE_LEN = 101  # number of time-points per normalised stride


def lnorm_loss(latent: torch.Tensor) -> torch.Tensor:
    """Soft penalty pushing latent values into [-1, 1] (Hausdörfer et al. 2024)."""
    abs_z = torch.abs(latent)
    exponent = torch.pow(abs_z / 1.2, 10).clamp(max=50.0)
    penalty = torch.exp(exponent) - 1.0
    return penalty.mean()


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_sq = (x ** 2).sum(dim=1, keepdim=True)
    y_sq = (y ** 2).sum(dim=1, keepdim=True)
    return x_sq + y_sq.t() - 2 * x @ y.t()


def _compute_mmd(z: torch.Tensor, latent_dim: int, device: torch.device) -> torch.Tensor:
    """MMD between encoded samples z and N(0, I) prior (WAE-MMD)."""
    batch_size = z.shape[0]
    z_prior = torch.randn(batch_size, latent_dim, device=device)
    sigma_sq = float(latent_dim)
    k_zz = torch.exp(-_pairwise_sq_dist(z, z) / (2 * sigma_sq))
    k_zp = torch.exp(-_pairwise_sq_dist(z, z_prior) / (2 * sigma_sq))
    k_pp = torch.exp(-_pairwise_sq_dist(z_prior, z_prior) / (2 * sigma_sq))
    return k_zz.mean() - 2 * k_zp.mean() + k_pp.mean()


# ── Sklearn baselines ──────────────────────────────────────────────────────────


class StridePCAModel(EMGModel):
    """PCA on flattened strides.

    Flattens (101, 11) → 1111, applies sklearn PCA, unflattens.

    PCA is the theoretical upper bound for *linear* compression: it maximises
    explained variance and is the only method here with a closed-form optimal
    solution. Use it as a ceiling metric — if a neural model can't beat PCA R²
    at the same latent_dim, something is wrong with the training.

    Note: PCA is not constrained to [0, 1] output. Reconstructions can be
    slightly negative for channels that are mostly silent.
    """

    def __init__(self, latent_dim: int, n_channels: int = 11, stride_len: int = STRIDE_LEN):
        super().__init__(latent_dim, n_channels)
        self.stride_len = stride_len
        self._model = PCA(n_components=latent_dim)

    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        self._model.fit(x_train.reshape(x_train.shape[0], -1))
        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        metrics["explained_variance_ratio"] = float(np.sum(self._model.explained_variance_ratio_))
        if x_val is not None:
            metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
        return metrics

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._model.transform(x.reshape(x.shape[0], -1))

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        x_flat = x.reshape(x.shape[0], -1)
        return self._model.inverse_transform(self._model.transform(x_flat)).reshape(-1, self.stride_len, self.n_channels)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)


class StrideNMFModel(EMGModel):
    """Non-Negative Matrix Factorisation on flattened strides.

    Factorises X ≈ W @ H  where W, H ≥ 0.
      - W (n_strides, k): per-stride activation weights (the latent)
      - H (k, 1111):      basis strides (the synergy templates)

    NMF is the standard muscle synergy extraction method. Non-negativity is
    biologically grounded: EMG can't go negative. The k components are
    interpretable as co-activation templates (e.g., "push-off synergy",
    "swing synergy").

    Edge case: NMF requires non-negative input. We clip to [0, ∞) at both
    fit and transform. Since data is already [0, 1] from preprocessing, clipping
    only matters for any floating-point underflow below zero.
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
            init="nndsvda",   # deterministic init, better than random for EMG
            max_iter=max_iter,
            random_state=seed,
        )

    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        x_flat = np.clip(x_train.reshape(x_train.shape[0], -1), 0, None)
        self._model.fit(x_flat)
        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        metrics["reconstruction_err"] = float(self._model.reconstruction_err_)
        if x_val is not None:
            metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
        return metrics

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._model.transform(np.clip(x.reshape(x.shape[0], -1), 0, None))

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return (self.transform(x) @ self._model.components_).reshape(-1, self.stride_len, self.n_channels)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)


class StrideCNMFModel(EMGModel):
    """Smoothness-Constrained NMF (CNMF) on flattened strides.

    Adds an L2 penalty on the activation coefficients W to reduce sensitivity
    to the non-convex local optima that plague standard NMF:

        min  ½·‖X − W·H‖²_F  +  α·‖W‖²_F    s.t. W, H ≥ 0

    The multiplicative update solver (Lee & Seung, 1999) is used to match
    the synergy literature. The alpha penalty discourages very large activation
    weights, which tend to cause a few synergies to dominate.

    In practice CNMF often has *worse* reconstruction R² than standard NMF
    because the penalty trades off fit quality for stability. This is expected.
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
            solver="mu",       # multiplicative updates required for alpha_W
            alpha_W=alpha,     # L2 penalty on activation coefficients only
            l1_ratio=0.0,      # pure L2, no L1 sparsity
            max_iter=max_iter,
            random_state=seed,
        )

    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        x_flat = np.clip(x_train.reshape(x_train.shape[0], -1), 0, None)
        self._model.fit(x_flat)
        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        metrics["reconstruction_err"] = float(self._model.reconstruction_err_)
        if x_val is not None:
            metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
        return metrics

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._model.transform(np.clip(x.reshape(x.shape[0], -1), 0, None))

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return (self.transform(x) @ self._model.components_).reshape(-1, self.stride_len, self.n_channels)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)


# ── Conv1D building blocks ─────────────────────────────────────────────────────


def _compute_conv_output_len(input_len: int) -> int:
    """Temporal length after the encoder's two strided conv layers.

    Conv1d(k=7, s=2, p=3): L_out = floor((L + 2·3 - 7) / 2) + 1
    Conv1d(k=5, s=2, p=2): L_out = floor((L + 2·2 - 5) / 2) + 1
    For input_len=101: 51 → 26.
    """
    l1 = (input_len + 2 * 3 - 7) // 2 + 1
    l2 = (l1 + 2 * 2 - 5) // 2 + 1
    return l2


class _StrideEncoder(nn.Module):
    """Shared Conv1D encoder for AE / WAE / MAE.

    Input:  (batch, stride_len, n_channels)  — channels last
    Output: (batch, latent_dim)

    Two strided convolutions downsample the 101-timepoint axis to 26 steps,
    then a two-layer FC compresses to latent_dim.

    The stride=2 convolutions act like learned pooling, preserving local
    temporal structure (e.g. the shape of a push-off burst) while halving
    resolution. The receptive field of the first conv is 7 timesteps = 35ms
    at 200Hz (after downsampling from 1000Hz), covering roughly 3–4% of the
    gait cycle — appropriate for capturing muscle onset/offset patterns.
    """

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
        x = x.transpose(1, 2)          # (B, T, C) → (B, C, T) for Conv1d
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(x.flatten(1))


class _StrideVAEEncoder(nn.Module):
    """VAE variant of the encoder: outputs (mu, log_var) instead of a point."""

    def __init__(self, n_channels: int, latent_dim: int, stride_len: int = STRIDE_LEN):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        conv_out_len = _compute_conv_output_len(stride_len)
        flat_size = 64 * conv_out_len
        self.fc_shared = nn.Sequential(nn.Linear(flat_size, 128), nn.ReLU())
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        h = self.fc_shared(x.flatten(1))
        return self.fc_mu(h), self.fc_log_var(h)


class _StrideDecoder(nn.Module):
    """Shared Conv1D decoder for AE / VAE / WAE / MAE.

    Input:  (batch, latent_dim)
    Output: (batch, stride_len, n_channels)  — unbounded, linear activation.

    No sigmoid on the final layer. The old sigmoid caused vanishing gradients
    near 0 and 1 (where most EMG values live), preventing the model from
    fitting sharp muscle bursts. Callers must clamp to [0, 1] before use.

    ConvTranspose1d layers are the mirror of the encoder's Conv1d layers.
    output_padding is computed to exactly recover the original stride_len.
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
        # Compute output_padding to exactly recover stride_len through the two transposes.
        l_after_conv1 = (stride_len + 2 * 3 - 7) // 2 + 1
        op2 = l_after_conv1 - ((conv_out_len - 1) * 2 - 2 * 2 + 5)
        op1 = stride_len - ((l_after_conv1 - 1) * 2 - 2 * 3 + 7)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=op2)
        self.deconv1 = nn.ConvTranspose1d(32, n_channels, kernel_size=7, stride=2, padding=3, output_padding=op1)
        self._conv_out_len = conv_out_len

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.shape[0], 64, self._conv_out_len)
        x = torch.relu(self.deconv2(x))
        x = self.deconv1(x)             # linear output — no sigmoid
        return x.transpose(1, 2)        # (B, C, T) → (B, T, C)


# ── Shared PyTorch base ────────────────────────────────────────────────────────


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _StrideModelBase(EMGModel):
    """Common infrastructure for all stride-level PyTorch models.

    Early stopping: val loss checked every 5 epochs; stops if no improvement
    for 15 consecutive checks (= 75 epochs). Saves the best val-loss checkpoint.
    """

    EARLY_STOP_PATIENCE: int = 15   # val-checks without improvement before stopping
    VAL_CHECK_INTERVAL: int = 5     # epochs between val checks

    def __init__(self, latent_dim: int, n_channels: int = 11, stride_len: int = STRIDE_LEN):
        super().__init__(latent_dim, n_channels)
        self.stride_len = stride_len
        self.device = _get_device()
        self._module: nn.Module | None = None

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _make_loader(self, x: np.ndarray, batch_size: int) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._to_tensor(x)),
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
        """Evaluate on val set, update scheduler and best-model checkpoint.

        Val loss is always pure MSE — not the full training loss (which may include
        KL, MMD, or Lnorm terms). Reason: we want to track reconstruction quality
        on unseen data, not whether regularisation terms are converging. The
        scheduler and early stopping both track this reconstruction MSE.

        Uses reconstruct() not transform() so the full encode→decode pipeline is
        exercised (VAE uses mu at inference, so no sampling noise in val).
        """
        val_hat = self.reconstruct(x_val)
        val_loss = float(np.mean((x_val - val_hat) ** 2))
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            # deep-copy the state dict — a reference would be mutated by future updates
            return val_loss, deepcopy(self._module.state_dict()), 0
        return best_val_loss, best_state, no_improve + 1

    @staticmethod
    def _log(epoch: int, epochs: int, metrics: dict[str, float]) -> None:
        parts = [f"Epoch {epoch + 1:>3d}/{epochs}"]
        for k in ["train_loss", "val_mse", "val_r2"]:
            if k in metrics:
                parts.append(f"{k}={metrics[k]:.5f}")
        print(f"  {' | '.join(parts)}")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._module.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self._module.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )


# ── StrideAutoencoder ──────────────────────────────────────────────────────────


class StrideAutoencoder(_StrideModelBase):
    """Deterministic Conv1D autoencoder.

    Loss = MSE(x, x̂)  +  lnorm_weight · Lnorm(z)

    Lnorm is a soft penalty that keeps latent values inside [-1.2, 1.2]:
        Lnorm(z) = mean(max(0, |z| - 1.2)²)
    This gives the RL policy a predictable input range without hard-clamping
    the latent (which would kill gradients during policy training).

    This is the simplest neural model and reconstruction quality upper bound
    for the conv architecture — VAE/WAE trade some R² for latent regularity.
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
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        log_fn: LogFn = None,
    ) -> dict[str, float]:
        loader = self._make_loader(x_train, batch_size)
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
                step_metrics.update(_stride_metrics(x_val, self.reconstruct(x_val)))
                self._log(epoch, epochs, {**step_metrics, "val_mse": best_val_loss})
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
            if log_fn:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)
        self._module.eval()

        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        return self.encoder(self._to_tensor(x)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        return self.decoder(self.encoder(self._to_tensor(x))).cpu().numpy()


# ── StrideVAE ─────────────────────────────────────────────────────────────────


class StrideVAE(_StrideModelBase):
    """Variational Conv1D autoencoder.

    Loss = MSE(x, x̂)  +  β · KL(q(z|x) ‖ N(0,I))  +  lnorm_weight · Lnorm(μ)

    The KL term forces the posterior q(z|x) = N(μ, σ²) to stay close to the
    standard Gaussian prior. This makes the latent space smooth and well-covered:
    interpolating between two latent points yields valid gait patterns.

    β scale note: MSE on (101,11) EMG is ~0.01–0.05. KL for latent_dim=4 at
    convergence is typically 1–4 nats. Setting β=0.1 would make KL dominate
    by ~10×, blurring reconstructions. β=0.01 keeps KL as a regulariser
    without overwhelming the reconstruction objective.

    At inference, we use μ (not a sample) to avoid stochasticity in the
    decoded muscle activations.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
        beta: float = 0.01,       # was 0.1 — see scale note above
        lnorm_weight: float = 0.01,
    ):
        super().__init__(latent_dim, n_channels, stride_len)
        self.beta = beta
        self.lnorm_weight = lnorm_weight
        self.encoder = _StrideVAEEncoder(n_channels, latent_dim, stride_len)
        self.decoder = _StrideDecoder(latent_dim, n_channels, stride_len)
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})
        self._module.to(self.device)

    def _reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + σ·ε using the reparameterisation trick."""
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        log_fn: LogFn = None,
    ) -> dict[str, float]:
        loader = self._make_loader(x_train, batch_size)
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )
        best_val_loss, best_state, no_improve = float("inf"), None, 0

        self._module.train()
        for epoch in range(epochs):
            epoch_loss = epoch_recon = epoch_kl = 0.0
            for (batch,) in loader:
                mu, log_var = self.encoder(batch)
                z = self._reparameterise(mu, log_var)
                x_hat = self.decoder(z)
                recon = nn.functional.mse_loss(x_hat, batch)
                # KL divergence: -½ Σ(1 + log σ² - μ² - σ²)
                kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
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
                step_metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
                self._log(epoch, epochs, step_metrics)
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
            if log_fn:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)
        self._module.eval()

        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return μ (mean of posterior) — deterministic, no sampling."""
        self._module.eval()
        mu, _ = self.encoder(self._to_tensor(x))
        return mu.cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct via μ (no stochasticity at inference)."""
        self._module.eval()
        mu, _ = self.encoder(self._to_tensor(x))
        return self.decoder(mu).cpu().numpy()


# ── StrideWAE_MMD ──────────────────────────────────────────────────────────────


class StrideWAE_MMD(_StrideModelBase):
    """Wasserstein Autoencoder with Maximum Mean Discrepancy regularisation.

    Loss = MSE(x, x̂)  +  mmd_weight · MMD(q(z), N(0,I))  +  lnorm_weight · Lnorm(z)

    WAE-MMD is deterministic like the plain AE (no reparameterisation trick),
    but adds a distribution-level penalty. Instead of KL forcing each individual
    posterior q(z|x) toward N(0,I) (VAE), MMD pushes the *aggregate* posterior
    q(z) = ∫ q(z|x) p(x) dx toward N(0,I) using a kernel two-sample test.

    Advantages over VAE for our use case:
      1. Sharper reconstructions — no KL blurring of individual posteriors.
      2. Better distance preservation — nearby strides in EMG space map to
         nearby latent points (important for RL interpolation).
      3. Smoother latent manifold — MMD fills in the latent space more
         uniformly than KL, reducing "holes" between observed strides.

    The RBF kernel bandwidth is set to latent_dim (bandwidth heuristic from
    Tolstikhin et al., 2018). mmd_weight=10 was found empirically to keep the
    MMD contribution comparable to the MSE term.

    Reference: Tolstikhin et al., "Wasserstein Auto-Encoders", ICLR 2018.
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
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        log_fn: LogFn = None,
    ) -> dict[str, float]:


        loader = self._make_loader(x_train, batch_size)
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )
        best_val_loss, best_state, no_improve = float("inf"), None, 0

        self._module.train()
        for epoch in range(epochs):
            epoch_loss = epoch_recon = epoch_mmd = 0.0
            for (batch,) in loader:
                z = self.encoder(batch)
                x_hat = self.decoder(z)
                recon = nn.functional.mse_loss(x_hat, batch)
                mmd = _compute_mmd(z, self.latent_dim, self.device)
                loss = recon + self.mmd_weight * mmd + self.lnorm_weight * lnorm_loss(z)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._module.parameters(), max_norm=1.0)
                optimizer.step()
                n = batch.shape[0]
                epoch_loss += loss.item() * n
                epoch_recon += recon.item() * n
                epoch_mmd += mmd.item() * n
            n_total = len(loader.dataset)
            step_metrics = {
                "train_loss": epoch_loss / n_total,
                "train_recon": epoch_recon / n_total,
                "train_mmd": epoch_mmd / n_total,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if x_val is not None and (epoch % self.VAL_CHECK_INTERVAL == 0 or epoch == epochs - 1):
                best_val_loss, best_state, no_improve = self._run_val(
                    x_val, best_val_loss, best_state, no_improve, scheduler,
                )
                step_metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
                self._log(epoch, epochs, step_metrics)
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
            if log_fn:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)
        self._module.eval()

        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        return self.encoder(self._to_tensor(x)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        return self.decoder(self.encoder(self._to_tensor(x))).cpu().numpy()


# ── StrideFlatAE ───────────────────────────────────────────────────────────────


class StrideFlatAE(_StrideModelBase):
    """Flatten-MLP autoencoder, faithful to Hausdörfer et al. (2024).

    Loss = MSE(x, x̂)  +  lnorm_weight · Lnorm(z)

    where Lnorm is the exact paper penalty (free inside ±0.8, steep wall at ±1.2).

    Architecture:
        Encoder: Flatten(1111) → Linear(512) → Tanh → Linear(256) → Tanh → Linear(latent)
        Decoder: Linear(256)   → Tanh → Linear(512) → Tanh → Linear(1111) → Reshape(101,11)

    **Why tanh, not ReLU:**
        The paper uses tanh throughout. For this model it is the right choice:
        - tanh output is in [-1, 1], naturally compatible with Lnorm's target range
        - Xavier init (designed for tanh) gives well-scaled initial activations
        - EMG input is in [0, 1], so first-layer tanh outputs are mostly positive —
          the network doesn't need negative activations to represent muscle data
        - Saturation risk: real but manageable with 512/256-unit layers and Xavier init.
          The paper uses 2×latent_dim hidden units, which is too small for our 1111-dim
          input; we scale up while keeping tanh.

    **Decoder output:** linear (no sigmoid). Caller clips to [0, 1] before actuator use.
    Sigmoid was removed because it kills gradients near 0 and 1 where most EMG lives.

    **RL interface:**
        Policy outputs z ∈ R^latent_dim, frozen decoder → (101, 11) muscle activations.
        EMGToMuscleMapper clips decoder output to [0, 1] then maps to [-1, 1] for MuJoCo.
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
        flat_dim = stride_len * n_channels  # 101 × 11 = 1111

        # Encoder: 1111 → 512 → 256 → latent_dim
        # 512/256 hidden units is larger than the paper's 2×latent_dim — needed
        # because our input is 1111-dim vs the paper's much smaller action space.
        # No activation on the latent output — z is unconstrained; Lnorm handles range.
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.Tanh(),
            nn.Linear(512, 256),      nn.Tanh(),
            nn.Linear(256, latent_dim),
        )
        # Decoder: latent_dim → 256 → 512 → 1111 (symmetric to encoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.Tanh(),
            nn.Linear(256, 512),        nn.Tanh(),
            nn.Linear(512, flat_dim),   # linear output — caller clips to [0, 1]
        )
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})

        # Xavier uniform init is designed for tanh (preserves variance through layers).
        # PyTorch default (Kaiming uniform) is designed for ReLU — wrong for tanh.
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
        # Flatten (N, 101, 11) → (N, 1111). MLP treats the stride as one long vector.
        # No temporal inductive bias, but fully connected layers still learn cross-time
        # correlations (e.g., gastroc burst at 60% always follows quad burst at 15%).
        x_flat = x_train.reshape(x_train.shape[0], -1)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._to_tensor(x_flat)),
            batch_size=batch_size, shuffle=True,
        )
        # Adam: works well for tanh networks. lr=1e-3 is the standard default.
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        # ReduceLROnPlateau halves LR when val MSE plateaus.
        # patience=10 checks × VAL_CHECK_INTERVAL=5 epochs = 50 epochs before halving.
        # EARLY_STOP_PATIENCE=15 checks = 75 epochs: LR typically halves once before
        # early stopping fires, giving the model time to escape the plateau first.
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
                # lnorm_loss: soft wall penalising |z_i| > 1.2.
                # Keeps z in a range the RL policy's bounded output can reliably cover
                # without hard-clamping z (which would kill gradients during RL training).
                loss = nn.functional.mse_loss(x_hat, batch) + self.lnorm_weight * lnorm_loss(z)
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping at 1.0: tanh layers can produce large gradients
                # in early epochs when the encoder weights are poorly initialised.
                nn.utils.clip_grad_norm_(self._module.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= len(loader.dataset)

            step_metrics = {"train_loss": epoch_loss, "lr": optimizer.param_groups[0]["lr"]}
            if x_val is not None and (epoch % self.VAL_CHECK_INTERVAL == 0 or epoch == epochs - 1):
                best_val_loss, best_state, no_improve = self._run_val(
                    x_val, best_val_loss, best_state, no_improve, scheduler,
                )
                step_metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
                self._log(epoch, epochs, step_metrics)
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
            if log_fn:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            # Restore best val checkpoint — last epoch may have started overfitting
            self._module.load_state_dict(best_state)
        self._module.eval()

        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Encode (N, 101, 11) → (N, latent_dim) latent vectors."""
        self._module.eval()
        return self.encoder(self._to_tensor(x.reshape(x.shape[0], -1))).cpu().numpy()

    @torch.no_grad()
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent z → (batch, 1111) muscle activations (unbounded; caller clips to [0,1]).

        RL interface: policy outputs z at each stride boundary. This frozen decoder
        maps it to the full 101-step EMG profile. LatentActionGymEnv then indexes into
        the profile using the current gait phase (sim_time % stride_duration).
        """
        self._module.eval()
        return self.decoder(torch.as_tensor(z, dtype=torch.float32, device=self.device)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Encode then decode: (N, 101, 11) → (N, 101, 11)."""
        self._module.eval()
        z = self.encoder(self._to_tensor(x.reshape(x.shape[0], -1)))
        return self.decoder(z).cpu().numpy().reshape(-1, self.stride_len, self.n_channels)


# ── StrideFlatVAE ─────────────────────────────────────────────────────────────


class StrideFlatVAE(_StrideModelBase):
    """Flatten-MLP VAE. Same MLP backbone as StrideFlatAE, adds reparameterisation + KL.

    The probabilistic angle: instead of mapping a stride to a single point z,
    the encoder outputs a Gaussian posterior q(z|x) = N(mu, sigma²).

    Why this matters for RL specifically:
        The PPO policy initialises near z=0 and explores by adding Gaussian noise.
        With FlatAE, z=0 may land in a region of latent space that was never in the
        training data — the decoder output there is undefined/extrapolated muscle noise.
        With FlatVAE, the KL term forces the aggregate posterior toward N(0,I), so z=0
        decodes to something close to the mean stride, and any z sampled from N(0,I)
        by the policy will decode to a biomechanically recognisable (if noisy) pattern.

    Posterior collapse problem (previously observed at beta=0.01):
        Standard KL can collapse — the encoder learns sigma→0, effectively ignoring
        the stochastic channel. To prevent this we use the free_bits constraint:
        each latent dimension is forced to encode at least `free_bits` nats of info.
        KL contribution per dim is clamped: KL_i = max(KL_i, free_bits).
        This guarantees sigma > 0 for every dim regardless of beta.
        Kingma et al. (2016) "Improving VAEs using Householder Flow", Appendix.

    Loss = MSE(x, x̂) + beta · Σ_i max(KL_i, free_bits) + lnorm_weight · Lnorm(mu)

    Recommended sweep:
        beta:      {0.001, 0.01, 0.05}   — bigger beta → smoother latent, lower R²
        free_bits: 0.5 nats (fixed)      — prevents collapse, do not sweep this

    Architecture (identical to StrideFlatAE backbone):
        Encoder body: Flatten(1111) → Linear(512) → Tanh → Linear(256) → Tanh
        mu head:      Linear(256, latent_dim)      ← posterior mean
        log_var head: Linear(256, latent_dim)      ← posterior log-variance
        Decoder:      Linear(latent_dim, 256) → Tanh → Linear(512) → Tanh → Linear(1111)

    At RL inference, transform() returns mu — no sampling noise.
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 11,
        stride_len: int = STRIDE_LEN,
        beta: float = 0.01,
        free_bits: float = 0.5,       # minimum KL per latent dim (nats) — prevents collapse
        lnorm_weight: float = 0.01,
    ):
        super().__init__(latent_dim, n_channels, stride_len)
        self.beta = beta
        self.free_bits = free_bits
        self.lnorm_weight = lnorm_weight
        flat_dim = stride_len * n_channels  # 1111

        # Shared encoder body — identical architecture to StrideFlatAE
        self.encoder_body = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.Tanh(),
            nn.Linear(512, 256),      nn.Tanh(),
        )
        # Two separate heads for the Gaussian parameters
        self.fc_mu = nn.Linear(256, latent_dim)       # posterior mean
        self.fc_log_var = nn.Linear(256, latent_dim)  # posterior log-variance
        # Decoder is identical to StrideFlatAE
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.Tanh(),
            nn.Linear(256, 512),        nn.Tanh(),
            nn.Linear(512, flat_dim),   # linear output — caller clips to [0, 1]
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
        """Run encoder body and return (mu, log_var) — the posterior parameters."""
        h = self.encoder_body(x_flat)
        return self.fc_mu(h), self.fc_log_var(h)

    def _reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + sigma * eps, eps ~ N(0, I).

        The reparameterisation trick rewrites the sample as a deterministic function
        of mu, log_var, and a noise variable eps that doesn't depend on the parameters.
        This allows gradients to flow through z to the encoder parameters.
        """
        std = torch.exp(0.5 * log_var)  # sigma = exp(log_sigma) = exp(0.5 * log_var)
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
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._to_tensor(x_flat)),
            batch_size=batch_size, shuffle=True,
        )
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

                # KL per latent dimension: KL_i = -½(1 + log σ²_i - μ²_i - σ²_i)
                # Shape: (batch, latent_dim)
                kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

                # Free bits: clamp each dim's KL to at least free_bits nats.
                # This prevents collapse (sigma→0, KL→0 for that dim) because the
                # gradient of max(KL_i, free_bits) w.r.t. encoder params is zero
                # only when KL_i > free_bits — the encoder is never incentivised
                # to reduce information content below the floor.
                kl_per_dim = kl_per_dim.clamp(min=self.free_bits)

                # Sum over dims (scales with latent_dim), mean over batch.
                kl = kl_per_dim.sum(dim=1).mean()

                # Lnorm on mu (not z): keeps the *mean* of the posterior bounded.
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
                "train_kl": epoch_kl / n_total,   # track KL separately to detect collapse
                "lr": optimizer.param_groups[0]["lr"],
            }
            if x_val is not None and (epoch % self.VAL_CHECK_INTERVAL == 0 or epoch == epochs - 1):
                best_val_loss, best_state, no_improve = self._run_val(
                    x_val, best_val_loss, best_state, no_improve, scheduler,
                )
                step_metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
                self._log(epoch, epochs, step_metrics)
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
            if log_fn:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)
        self._module.eval()

        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Encode (N, 101, 11) → (N, latent_dim) posterior means mu.

        Returns mu, not a sample. At RL inference we want a deterministic mapping
        from observation to latent — sampling would add noise to every policy step.
        The latent space smoothness from KL training means mu alone is reliable.
        """
        self._module.eval()
        mu, _ = self._encode(self._to_tensor(x.reshape(x.shape[0], -1)))
        return mu.cpu().numpy()

    @torch.no_grad()
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode z → (batch, 1111) muscle activations (unbounded; caller clips to [0,1]).

        Same interface as StrideFlatAE.decode — RL can swap models transparently.
        """
        self._module.eval()
        return self.decoder(torch.as_tensor(z, dtype=torch.float32, device=self.device)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Encode via mu (no sampling) then decode: (N, 101, 11) → (N, 101, 11)."""
        self._module.eval()
        mu, _ = self._encode(self._to_tensor(x.reshape(x.shape[0], -1)))
        return self.decoder(mu).cpu().numpy().reshape(-1, self.stride_len, self.n_channels)


# ── StrideMaskedAutoencoder ────────────────────────────────────────────────────


class StrideMaskedAutoencoder(_StrideModelBase):
    """Conv1D autoencoder trained with contiguous block masking.

    At each training step, ~30% of the 101 timepoints are zeroed out in
    contiguous blocks of 15 points (~2 blocks per stride). The loss is computed
    only on the masked timepoints, forcing the encoder to fill in missing
    sections from surrounding context rather than copying visible inputs.

    This learns longer-range temporal dependencies than the plain AE: the model
    must understand that "given push-off at 60%, swing flexion should follow at
    75%" rather than just "copy whatever you see."

    Known limitation: masked positions are filled with zeros, not a learned
    [MASK] token. Since EMG=0 means "silent muscle" — a valid real value — the
    encoder cannot distinguish masked regions from truly silent muscles. For a
    CNN encoder this is a minor issue (context from adjacent frames resolves
    the ambiguity in most cases), but it means the training signal on masked
    regions with naturally low EMG is weak.

    At inference (reconstruct), the full unmasked stride is passed to the
    encoder, so the encoder receives more information than during training.
    This train/inference mismatch is intentional: it makes the latent z at
    inference *richer* than during training, so reconstructions are generally
    better than the masked training loss suggests.
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

    def _block_mask(self, batch_size: int) -> torch.Tensor:
        """Generate a block mask: 1 = visible, 0 = masked.

        Randomly places n_blocks contiguous blocks of block_size timepoints.
        Shape: (batch, stride_len, 1) — broadcasts over channels so all 11
        EMG channels are masked together at the same timepoints.
        """
        n_masked = max(self.block_size, int(self.stride_len * self.mask_ratio))
        n_blocks = max(1, n_masked // self.block_size)
        mask = torch.ones(batch_size, self.stride_len, 1, device=self.device)
        max_start = self.stride_len - self.block_size
        for _ in range(n_blocks):
            starts = torch.randint(0, max_start + 1, (batch_size,), device=self.device)
            for b in range(batch_size):
                mask[b, starts[b]: starts[b] + self.block_size, :] = 0.0
        return mask

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray | None = None,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        log_fn: LogFn = None,
    ) -> dict[str, float]:
        loader = self._make_loader(x_train, batch_size)
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )
        best_val_loss, best_state, no_improve = float("inf"), None, 0

        self._module.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                mask = self._block_mask(batch.shape[0])
                z = self.encoder(batch * mask)   # encode corrupted input
                x_hat = self.decoder(z)
                # Loss only on masked timepoints — forces learning from context.
                inv_mask = 1.0 - mask
                n_masked = inv_mask.sum()
                recon = (((x_hat - batch) ** 2) * inv_mask).sum() / n_masked if n_masked > 0 \
                    else nn.functional.mse_loss(x_hat, batch)
                loss = recon + self.lnorm_weight * lnorm_loss(z)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._module.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= len(loader.dataset)

            step_metrics = {"train_loss": epoch_loss, "lr": optimizer.param_groups[0]["lr"]}
            if x_val is not None and (epoch % self.VAL_CHECK_INTERVAL == 0 or epoch == epochs - 1):
                # Val loss uses full reconstruction MSE (no masking) — this is
                # what matters for downstream RL use.
                best_val_loss, best_state, no_improve = self._run_val(
                    x_val, best_val_loss, best_state, no_improve, scheduler,
                )
                step_metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
                self._log(epoch, epochs, step_metrics)
                if no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
            if log_fn:
                log_fn(step_metrics, epoch)

        if best_state is not None:
            self._module.load_state_dict(best_state)
        self._module.eval()

        metrics = _stride_metrics(x_train, self.reconstruct(x_train))
        if x_val is not None:
            metrics.update({f"val_{k}": v for k, v in _stride_metrics(x_val, self.reconstruct(x_val)).items()})
        return metrics

    @torch.no_grad()
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Encode full (unmasked) stride at inference."""
        self._module.eval()
        return self.encoder(self._to_tensor(x)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct from full (unmasked) stride — no masking at inference."""
        self._module.eval()
        return self.decoder(self.encoder(self._to_tensor(x))).cpu().numpy()


# ── Helper ─────────────────────────────────────────────────────────────────────


def _stride_metrics(x: np.ndarray, x_hat: np.ndarray) -> dict[str, float]:
    """MSE and R² on 3D stride arrays, computed by flattening to 2D."""
    return compute_train_metrics(
        x.reshape(-1, x.shape[-1]),
        x_hat.reshape(-1, x_hat.shape[-1]),
    )


# ── HausdorferAE — snapshot-level, exact paper architecture ───────────────────


def _lnorm_loss_exact(z: torch.Tensor) -> torch.Tensor:
    """Exact Lnorm from Hausdörfer et al. (2024).

    Per-element: 0 if |z| < 0.8, else exp((z/1.2)^10) - 1.
    The 0.8 dead-zone lets the RL policy move freely inside the latent ball.
    The steep wall at ±1.2 keeps z bounded without hard-clamping gradients.
    """
    abs_z = z.abs()
    in_zone = abs_z < 0.8
    penalty = torch.exp(torch.pow(abs_z / 1.2, 10).clamp(max=50.0)) - 1.0
    return penalty.masked_fill(in_zone, 0.0).mean()


class HausdorferAE(EMGModel):
    """Snapshot-level MLP autoencoder — exact Hausdörfer et al. (2024) architecture.

    Input/output: one action frame (action_dim,) = (18,) for gait10dof18musc.
    Train on individual muscle activation snapshots, not full strides.

    Architecture (one hidden layer as in paper):
        Encoder: Linear(action_dim → 2*latent_dim) → Tanh → Linear(2*latent_dim → latent_dim)
        Decoder: Linear(latent_dim → 2*latent_dim) → Tanh → Linear(2*latent_dim → action_dim)

    Defaults for gait10dof18musc (18 muscles):
        action_dim = 18
        latent_dim = 9    (= action_dim // 2, per paper)
        hidden_dim = 18   (= 2 × latent_dim, per paper)

    Loss: MSE(a, â) + lnorm_weight × Lnorm(z)

    Decoder output is linear (no sigmoid). decode(z) returns values in [0,1] physical space
    (muscle excitation space). The RL interface must scale to [-1,1] before env.step()
    using EMGToMuscleMapper.to_loco_action() — which already handles this.

    ## Data space: train in [0, 1] (physical EMG space)
    Muscle excitations are non-negative by definition. The AE trains in this physical
    space so metrics (MSE, R²) are interpretable and the pipeline stays consistent
    with EMGToMuscleMapper. The Lnorm penalty acts on the latent z, not on the data
    space — so the choice of data range doesn't affect Lnorm behaviour.

    ## How to prepare training data
    EMG data lives as strides (N_strides, 101, 11). Map to 18 muscles via
    EMGToMuscleMapper, then flatten the time axis:
        mapped = mapper.map_batch(strides.reshape(-1, 11)).reshape(N, 101, 18)
        snapshots = mapped.reshape(-1, 18)    # (N*101, 18) in [0,1] — train on this

    ## RL interface (after pre-training + freeze)
        z = policy(obs)                               # (latent_dim,) from policy head
        muscle_act = np.clip(ae.decode(z), 0, 1)     # (action_dim,) in [0,1]
        loco_action = mapper.to_loco_action(muscle_act)   # scale to [-1,1] for env.step()
        # DefaultControl maps [-1,1] → ctrl [0,1] for muscles automatically.
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
        self.device = _get_device()
        hidden_dim = 2 * latent_dim  # paper: one hidden layer, size 2×latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),  # linear out — caller clamps to [0,1]
        )
        self._module = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})

        for m in self._module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # designed for tanh
                nn.init.zeros_(m.bias)

        self._module.to(self.device)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

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
            epochs:     300 is sufficient — the model is tiny (≈1 k params).
            batch_size: 4096 (large batches stabilise Lnorm, matching paper).
            lr:         3e-4 (Adam, paper default).
            log_fn:     Optional callback(metrics_dict, epoch) for W&B/TensorBoard.
            ckpt_path:  If given, best model is saved here whenever val_mse improves.
            min_delta:  Minimum improvement in val_mse to reset the early-stop counter.
        """
        pin = self.device.type == "cuda"
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.as_tensor(x_train, dtype=torch.float32)  # CPU — moved to GPU per batch
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
                loss = nn.functional.mse_loss(a_hat, batch) + self.lnorm_weight * _lnorm_loss_exact(z)
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
                    print(f"  Early stopping at epoch {epoch + 1} (patience={self.EARLY_STOP_PATIENCE} × {self.VAL_CHECK_INTERVAL} = {self.EARLY_STOP_PATIENCE * self.VAL_CHECK_INTERVAL} epochs, min_delta={min_delta})")
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
        """Encode (N, action_dim) → (N, latent_dim)."""
        self._module.eval()
        return self.encoder(self._to_tensor(x)).cpu().numpy()

    @torch.no_grad()
    def decode(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """Decode latent z → muscle activations in [0,1] physical space.

        Caller must scale to [-1,1] before env.step():
            muscle_act = np.clip(ae.decode(z), 0, 1)
            loco_action = mapper.to_loco_action(muscle_act)   # 2*a - 1
            env.step(loco_action)
        """
        self._module.eval()
        if isinstance(z, np.ndarray):
            z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        return self.decoder(z.to(self.device)).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Encode then decode: (N, action_dim) → (N, action_dim)."""
        self._module.eval()
        return self.decoder(self.encoder(self._to_tensor(x))).cpu().numpy()

    def freeze(self) -> None:
        """Freeze all parameters — call before RL training.

        After this, only the policy is updated; the decoder maps policy z
        to muscle commands as a fixed, pre-trained transformation.
        """
        for p in self._module.parameters():
            p.requires_grad_(False)
        self._module.eval()

    def get_decoder_module(self) -> nn.Sequential:
        """Return the decoder nn.Sequential for embedding directly in the RL agent."""
        return self.decoder

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
        """Load from a checkpoint saved by save() — infers config from file."""
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
