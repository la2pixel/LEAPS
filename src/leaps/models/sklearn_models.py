"""Sklearn-based EMG dimensionality reduction models (PCA, NMF)."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF, PCA

from leaps.evaluation.metrics import compute_train_metrics
from leaps.models.base import EMGModel


class PCAModel(EMGModel):
    """PCA dimensionality reduction for EMG signals."""

    def __init__(self, latent_dim: int, n_channels: int = 11):
        super().__init__(latent_dim, n_channels)
        self._model = PCA(n_components=latent_dim)

    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        self._model.fit(x_train)
        x_hat = self.reconstruct(x_train)
        metrics = compute_train_metrics(x_train, x_hat)
        metrics["explained_variance_ratio"] = float(
            np.sum(self._model.explained_variance_ratio_)
        )
        if x_val is not None:
            x_val_hat = self.reconstruct(x_val)
            val_metrics = compute_train_metrics(x_val, x_val_hat)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        return metrics

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._model.transform(x)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self._model.inverse_transform(self._model.transform(x))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)


class NMFModel(EMGModel):
    """Non-Negative Matrix Factorization for muscle synergy extraction."""

    def __init__(self, latent_dim: int, n_channels: int = 11, max_iter: int = 500, seed: int = 42):
        super().__init__(latent_dim, n_channels)
        self._model = NMF(
            n_components=latent_dim,
            init="nndsvda",
            max_iter=max_iter,
            random_state=seed,
        )

    def fit(self, x_train: np.ndarray, x_val: np.ndarray | None = None) -> dict[str, float]:
        x_clipped = np.clip(x_train, 0, None)
        self._model.fit(x_clipped)
        x_hat = self.reconstruct(x_train)
        metrics = compute_train_metrics(x_train, x_hat)
        metrics["reconstruction_err"] = float(self._model.reconstruction_err_)
        if x_val is not None:
            x_val_hat = self.reconstruct(x_val)
            val_metrics = compute_train_metrics(x_val, x_val_hat)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        return metrics

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._model.transform(np.clip(x, 0, None))

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        latent = self.transform(x)
        return latent @ self._model.components_

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)
