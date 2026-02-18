"""Action representation models for EMG dimensionality reduction."""

from leaps.models.base import EMGModel
from leaps.models.pytorch_models import Autoencoder, MaskedAutoencoder, VAE
from leaps.models.sklearn_models import NMFModel, PCAModel
from leaps.models.stride_models import (
    StrideAutoencoder,
    StrideMaskedAutoencoder,
    StrideNMFModel,
    StridePCAModel,
    StrideVAE,
)

__all__ = [
    "EMGModel",
    # Snapshot-level
    "PCAModel",
    "NMFModel",
    "Autoencoder",
    "VAE",
    "MaskedAutoencoder",
    # Stride-level
    "StridePCAModel",
    "StrideNMFModel",
    "StrideAutoencoder",
    "StrideVAE",
    "StrideMaskedAutoencoder",
]
