"""
LEAPS: Learning Humanoid Locomotion from EMG-Based Latent Action Priors and Muscle Synergies
"""

__version__ = "0.1.0"
__author__ = "Lalitha Sivakumar"

from leaps.data import EMGPreprocessor, load_camargo_dataset
from leaps.models import LatentActionPrior, SynergyExtractor

__all__ = [
    "EMGPreprocessor",
    "load_camargo_dataset",
    "LatentActionPrior",
    "SynergyExtractor",
]
