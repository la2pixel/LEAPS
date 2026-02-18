"""LocoMuJoCo environment for the gait10dof18musc 2D sagittal-plane humanoid.

This model has 18 muscle actuators (9 per leg), 10 DOFs, and NO free joint.
The root is defined by three separate joints: pelvis_tx (slide), pelvis_ty (slide),
pelvis_tilt (hinge). We subclass LocoEnv directly (not BaseSkeleton) because
BaseSkeleton assumes the full 92-muscle skeleton with arms, box feet, etc.
"""

from pathlib import Path
from typing import Any, List, Tuple, Union

import mujoco
import numpy as np
from mujoco import MjData, MjModel, MjSpec

from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from loco_mujoco.environments import LocoEnv


# Absolute path to the XML model
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "humanoid"
_DEFAULT_XML = str(_MODEL_DIR / "gait10dof18musc.xml")

# 18 muscle actuators in XML order
_ACTUATOR_NAMES = [
    "hamstrings_r", "bifemsh_r", "glut_max_r", "iliopsoas_r", "rect_fem_r", "vasti_r",
    "gastroc_r", "soleus_r", "tib_ant_r",
    "hamstrings_l", "bifemsh_l", "glut_max_l", "iliopsoas_l", "rect_fem_l", "vasti_l",
    "gastroc_l", "soleus_l", "tib_ant_l",
]

# Real DOF joints (exclude constraint joints used for muscle wrapping paths)
_POSITION_JOINTS = [
    "pelvis_tilt",
    "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
    "lumbar_extension",
]

_VELOCITY_JOINTS = [
    "pelvis_tx", "pelvis_ty",  # include root translations in velocity
    "pelvis_tilt",
    "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
    "lumbar_extension",
]


class Gait10dof18Musc(LocoEnv):
    """LocoMuJoCo environment for the 2D sagittal-plane gait10dof18musc humanoid.

    18 muscle actuators, 10 DOFs, no free joint.
    Observation: 8 joint positions + 10 joint velocities = 18 dims.
    """

    mjx_enabled = False

    def __init__(
        self,
        spec: Union[str, MjSpec] = None,
        observation_spec: List[ObservationType] = None,
        actuation_spec: List[str] = None,
        **kwargs,
    ) -> None:
        if spec is None:
            spec = self.get_default_xml_file_path()

        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        if observation_spec is None:
            observation_spec = self._get_observation_specification(spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        super().__init__(
            spec=spec,
            actuation_spec=actuation_spec,
            observation_spec=observation_spec,
            **kwargs,
        )

    # ── Class methods ──────────────────────────────────────────────────

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        return _DEFAULT_XML

    @classmethod
    def generate(cls, task=None, **kwargs):
        """Create environment with sensible defaults for RL."""
        kwargs.setdefault("terminal_state_type", "HeightJointTerminalStateHandler")
        kwargs.setdefault("goal_type", "NoGoal")
        kwargs.setdefault("reward_type", "NoReward")
        return cls(**kwargs)

    # ── Spec methods ───────────────────────────────────────────────────

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[ObservationType]:
        obs = []
        # Joint positions (8) — exclude pelvis_tx for translation invariance
        for jnt in _POSITION_JOINTS:
            obs.append(ObservationType.JointPos(f"q_{jnt}", xml_name=jnt))
        # Joint velocities (10) — include pelvis_tx and pelvis_ty
        for jnt in _VELOCITY_JOINTS:
            obs.append(ObservationType.JointVel(f"dq_{jnt}", xml_name=jnt))
        return obs

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        return list(_ACTUATOR_NAMES)

    # ── Info properties ────────────────────────────────────────────────

    @info_property
    def root_body_name(self) -> str:
        return "pelvis"

    @info_property
    def upper_body_xml_name(self) -> str:
        return "torso"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        return "pelvis_tx"

    @info_property
    def root_height_healthy_range(self) -> tuple:
        return (0.5, 1.3)

    # ── Reset to keyframe pose ───────────────────────────────────────────

    def _reset_carry(
        self, model: MjModel, data: MjData, carry: Any
    ) -> Tuple[MjData, Any]:
        """Load the default keyframe pose on reset.

        LocoMuJoCo's base reset uses mj_resetData which zeros qpos.
        Our model needs the keyframe pose (pelvis_ty=0.95, constraint joints
        at their default values) to start in a valid standing configuration.
        """
        data, carry = super()._reset_carry(model, data, carry)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
        return data, carry

    # ── Override free joint properties (no free joint in 2D model) ─────

    @property
    def free_jnt_qpos_id(self):
        return np.zeros((0, 7), dtype=int)

    @property
    def free_jnt_qvel_id(self):
        return np.zeros((0, 6), dtype=int)
