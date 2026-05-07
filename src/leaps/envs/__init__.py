"""Environment wrappers for muscle-actuated humanoid simulation."""

from leaps.envs.emg_mapping import EMGToMuscleMapper
from leaps.envs.gait10dof_env import Gait10dof18Musc
from leaps.envs.gym_wrapper import Gait10dof18MuscGymEnv, LatentActionGymEnv
from leaps.envs.latent_control import LatentDecoderControl
from leaps.envs.muscle_env import MuscleHumanoidEnv
from leaps.envs.rewards import ForwardVelocityReward, WalkingReward, LinearWalkingReward, HaeufleReward
from leaps.envs.terminal_state import HeightJointTerminalStateHandler, NoTerminalStateHandler

# Register custom components with LocoMuJoCo's registries.
# Order matters: terminal state and reward handlers must be registered
# before the env, because the env constructor looks them up by name.
HeightJointTerminalStateHandler.register()
NoTerminalStateHandler.register()
ForwardVelocityReward.register()
WalkingReward.register()
LinearWalkingReward.register()
HaeufleReward.register()
LatentDecoderControl.register()   # register before env so conf_ppo.yaml can reference it
Gait10dof18Musc.register()

__all__ = [
    "MuscleHumanoidEnv",
    "EMGToMuscleMapper",
    "Gait10dof18Musc",
    "HeightJointTerminalStateHandler",
    "ForwardVelocityReward",
    "WalkingReward",
    "LinearWalkingReward",
    "HaeufleReward",
    "LatentDecoderControl",
    "Gait10dof18MuscGymEnv",
    "LatentActionGymEnv",
]
