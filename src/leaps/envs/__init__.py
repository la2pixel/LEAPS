"""Environments and reward/terminal utilities for muscle-actuated locomotion."""

from leaps.envs.adaptive_effort_reward import AdaptiveEffortRewardWrapper
from leaps.envs.emg_mapping import H2190_SCOPE_PROXY, EMGToMuscleMapper
from leaps.envs.latent_env import LatentActionPriorWrapper
from leaps.envs.rewards import (
    ForwardVelocityReward,
    HaeufleReward,
    LinearWalkingReward,
    WalkingReward,
)
from leaps.envs.terminal_state import HeightTerminal, NoTerminal

__all__ = [
    "EMGToMuscleMapper",
    "H2190_SCOPE_PROXY",
    "LatentActionPriorWrapper",
    "AdaptiveEffortRewardWrapper",
    "ForwardVelocityReward",
    "WalkingReward",
    "HaeufleReward",
    "LinearWalkingReward",
    "HeightTerminal",
    "NoTerminal",
]
