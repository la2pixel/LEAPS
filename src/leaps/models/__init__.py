"""Action representation models for EMG dimensionality reduction."""

from leaps.models.base import EMGModel
from leaps.models.sklearn_models import NMFModel, PCAModel
from leaps.models.stride_models import (
    HausdorferAE,
    StrideAutoencoder,
    StrideFlatAE,
    StrideFlatVAE,
    StrideMaskedAutoencoder,
    StrideNMFModel,
    StridePCAModel,
    StrideVAE,
    StrideWAE_MMD,
)

__all__ = [
    "EMGModel",
    # Snapshot-level: Hausdörfer-exact latent action prior
    "HausdorferAE",
    # Snapshot-level baselines
    "PCAModel",
    "NMFModel",
    # Stride-level
    "StridePCAModel",
    "StrideNMFModel",
    "StrideAutoencoder",
    "StrideVAE",
    "StrideWAE_MMD",
    "StrideFlatAE",
    "StrideFlatVAE",
    "StrideMaskedAutoencoder",
]
