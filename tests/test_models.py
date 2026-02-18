"""Unit tests for EMG dimensionality reduction models using synthetic data."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from leaps.evaluation.metrics import compute_train_metrics, mse, r_squared
from leaps.models.base import EMGModel
from leaps.models.pytorch_models import Autoencoder, MaskedAutoencoder, VAE
from leaps.models.sklearn_models import NMFModel, PCAModel

N_SAMPLES = 1000
N_CHANNELS = 11
LATENT_DIM = 4


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (N, 11) data resembling EMG activations (non-negative)."""
    rng = np.random.default_rng(42)
    x = rng.random((N_SAMPLES, N_CHANNELS)).astype(np.float32)
    n_val = 100
    return x[n_val:], x[:n_val]


# --- Metric tests ---


class TestMetrics:
    def test_perfect_reconstruction(self):
        x = np.random.rand(100, 11)
        assert mse(x, x) == pytest.approx(0.0)
        assert r_squared(x, x) == pytest.approx(1.0)

    def test_zero_input(self):
        x = np.zeros((100, 11))
        assert mse(x, x) == pytest.approx(0.0)

    def test_compute_train_metrics(self):
        x = np.random.rand(50, 11)
        m = compute_train_metrics(x, x)
        assert set(m.keys()) == {"mse", "r2"}
        assert m["mse"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)


# --- Model interface tests ---


def _check_model_interface(model: EMGModel, x_train: np.ndarray, x_val: np.ndarray):
    """Verify that a model implements the full EMGModel interface correctly."""
    metrics = model.fit(x_train, x_val)
    assert isinstance(metrics, dict)
    assert "mse" in metrics

    z = model.transform(x_val)
    assert z.shape == (x_val.shape[0], model.latent_dim)

    x_hat = model.reconstruct(x_val)
    assert x_hat.shape == x_val.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model_checkpoint"
        model.save(path)
        assert path.exists()

        x_hat_before = model.reconstruct(x_val)
        model.load(path)
        x_hat_after = model.reconstruct(x_val)
        np.testing.assert_allclose(x_hat_before, x_hat_after, atol=1e-6)


class TestPCAModel:
    def test_interface(self, synthetic_data):
        x_train, x_val = synthetic_data
        model = PCAModel(latent_dim=LATENT_DIM)
        _check_model_interface(model, x_train, x_val)

    def test_explained_variance(self, synthetic_data):
        x_train, x_val = synthetic_data
        model = PCAModel(latent_dim=LATENT_DIM)
        metrics = model.fit(x_train, x_val)
        assert "explained_variance_ratio" in metrics
        assert 0.0 < metrics["explained_variance_ratio"] <= 1.0


class TestNMFModel:
    def test_interface(self, synthetic_data):
        x_train, x_val = synthetic_data
        model = NMFModel(latent_dim=LATENT_DIM)
        _check_model_interface(model, x_train, x_val)

    def test_non_negative_latent(self, synthetic_data):
        x_train, x_val = synthetic_data
        model = NMFModel(latent_dim=LATENT_DIM)
        model.fit(x_train)
        z = model.transform(x_val)
        assert np.all(z >= 0)


class TestAutoencoder:
    def test_interface(self, synthetic_data):
        x_train, x_val = synthetic_data
        model = Autoencoder(latent_dim=LATENT_DIM)
        model.fit(x_train, x_val, epochs=5, batch_size=128)
        z = model.transform(x_val)
        assert z.shape == (x_val.shape[0], LATENT_DIM)
        x_hat = model.reconstruct(x_val)
        assert x_hat.shape == x_val.shape

    def test_custom_hidden_dim(self, synthetic_data):
        """Ablation: larger hidden layer should still work."""
        x_train, x_val = synthetic_data
        model = Autoencoder(latent_dim=LATENT_DIM, hidden_dim=64)
        model.fit(x_train, x_val, epochs=3, batch_size=128)
        z = model.transform(x_val)
        assert z.shape == (x_val.shape[0], LATENT_DIM)

    def test_save_load(self, synthetic_data):
        x_train, x_val = synthetic_data
        model = Autoencoder(latent_dim=LATENT_DIM)
        model.fit(x_train, epochs=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ae.pt"
            model.save(path)
            x_hat_before = model.reconstruct(x_val)
            model.load(path)
            x_hat_after = model.reconstruct(x_val)
            np.testing.assert_allclose(x_hat_before, x_hat_after, atol=1e-6)


class TestVAE:
    def test_interface(self, synthetic_data):
        x_train, x_val = synthetic_data
        model = VAE(latent_dim=LATENT_DIM)
        model.fit(x_train, x_val, epochs=5, batch_size=128)
        z = model.transform(x_val)
        assert z.shape == (x_val.shape[0], LATENT_DIM)
        x_hat = model.reconstruct(x_val)
        assert x_hat.shape == x_val.shape


class TestMaskedAutoencoder:
    def test_interface(self, synthetic_data):
        x_train, x_val = synthetic_data
        model = MaskedAutoencoder(latent_dim=LATENT_DIM, mask_ratio=0.5)
        model.fit(x_train, x_val, epochs=5, batch_size=128)
        z = model.transform(x_val)
        assert z.shape == (x_val.shape[0], LATENT_DIM)
        x_hat = model.reconstruct(x_val)
        assert x_hat.shape == x_val.shape


# --- Trainer + factory tests ---


class TestTrainer:
    def test_build_model(self):
        from leaps.training.trainer import build_model

        for name in ["PCA", "NMF", "AE", "VAE", "MAE"]:
            model = build_model(name, latent_dim=3)
            assert isinstance(model, EMGModel)
            assert model.latent_dim == 3

    def test_build_with_hidden_dim(self):
        from leaps.training.trainer import build_model

        model = build_model("AE", latent_dim=4, hidden_dim=64)
        assert isinstance(model, Autoencoder)
        # Verify the hidden layer size is 64, not 2*4=8
        first_layer = model.encoder.net[0]
        assert first_layer.out_features == 64

    def test_build_unknown_raises(self):
        from leaps.training.trainer import build_model

        with pytest.raises(ValueError, match="Unknown model"):
            build_model("UNKNOWN", latent_dim=3)

    def test_train_sklearn(self, synthetic_data):
        from leaps.training.trainer import train_sklearn_model

        x_train, x_val = synthetic_data
        model = PCAModel(latent_dim=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = train_sklearn_model(model, x_train, x_val, save_dir=tmpdir)
            assert "mse" in metrics
            assert Path(tmpdir, "PCAModel_d3.pkl").exists()

    def test_train_pytorch(self, synthetic_data):
        from leaps.training.trainer import train_pytorch_model

        x_train, x_val = synthetic_data
        model = Autoencoder(latent_dim=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = train_pytorch_model(
                model, x_train, x_val, epochs=3, batch_size=128, save_dir=tmpdir
            )
            assert "mse" in metrics
            assert Path(tmpdir, "Autoencoder_d3.pt").exists()

    def test_log_fn_callback(self, synthetic_data):
        x_train, x_val = synthetic_data
        logged = []

        def log_fn(metrics, step):
            logged.append((step, metrics))

        model = Autoencoder(latent_dim=3)
        model.fit(x_train, x_val, epochs=5, batch_size=128, log_fn=log_fn)
        assert len(logged) == 5
        assert all("train_loss" in m for _, m in logged)
        val_logged = [s for s, m in logged if "val_mse" in m]
        assert len(val_logged) >= 2


# --- Dataset tests ---


class TestDataset:
    def test_emg_dataset(self):
        from leaps.data.dataset import EMGDataset

        data = np.random.rand(100, 11).astype(np.float32)
        ds = EMGDataset(data)
        assert len(ds) == 100
        assert ds[0].shape == (11,)

    def test_train_val_split(self):
        from leaps.data.dataset import train_val_split

        x = np.random.rand(1000, 11)
        x_train, x_val = train_val_split(x, val_fraction=0.1, seed=42)
        assert x_train.shape[0] == 900
        assert x_val.shape[0] == 100

    def test_train_val_split_deterministic(self):
        from leaps.data.dataset import train_val_split

        x = np.random.rand(100, 11)
        t1, v1 = train_val_split(x, seed=0)
        t2, v2 = train_val_split(x, seed=0)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(v1, v2)
