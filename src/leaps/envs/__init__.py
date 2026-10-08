"""Environment wrappers for muscle-actuated locomotion."""

from leaps.envs.emg_mapping import EMGToMuscleMapper
from leaps.envs.latent_env import LatentActionPriorWrapper

__all__ = [
    "EMGToMuscleMapper",
    "LatentActionPriorWrapper",
]
