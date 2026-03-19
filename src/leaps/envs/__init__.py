"""Environment wrappers for muscle-actuated humanoid simulation."""

from leaps.envs.emg_mapping import EMGToMuscleMapper
from leaps.envs.gait10dof_env import Gait10dof18Musc
from leaps.envs.gym_wrapper import Gait10dof18MuscGymEnv, LatentActionGymEnv
from leaps.envs.muscle_env import MuscleHumanoidEnv
from leaps.envs.rewards import ForwardVelocityReward, WalkingReward
from leaps.envs.terminal_state import HeightJointTerminalStateHandler

# Register custom components with LocoMuJoCo
HeightJointTerminalStateHandler.register()
ForwardVelocityReward.register()
WalkingReward.register()
Gait10dof18Musc.register()

__all__ = [
    "MuscleHumanoidEnv",
    "EMGToMuscleMapper",
    "Gait10dof18Musc",
    "HeightJointTerminalStateHandler",
    "ForwardVelocityReward",
    "WalkingReward",
    "Gait10dof18MuscGymEnv",
    "LatentActionGymEnv",
]
