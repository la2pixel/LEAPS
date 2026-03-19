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


# Real DOF joint angles for a right-heel-strike initial pose.
# The wrapping-path constraint joints (knee translations, vasti/gastroc/iliopsoas
# waypoints) are LEFT at their keyframe-0 defaults — the very stiff equality
# constraints (solimp=0.9999) snap them to the correct positions within 1-2 steps.
# Sign convention: positive hip_flexion = leg forward; knee_angle < 0 = flexion.
_HEELSTRIKE_POSE = {
    "pelvis_tilt":      -0.05,   # slight forward lean
    "hip_flexion_r":    +0.30,   # right leg forward ~17°
    "knee_angle_r":     -0.05,   # nearly extended
    "ankle_angle_r":    +0.10,   # dorsiflexed for heel contact
    "hip_flexion_l":    -0.15,   # left leg behind ~9° (push-off)
    "knee_angle_l":     -0.35,   # 20° flexion (beginning of swing)
    "ankle_angle_l":    -0.40,   # plantarflexed ~23° (push-off)
    "lumbar_extension":  0.0,
}

# Forward walking speed matching EMG recording conditions.
_INIT_FORWARD_VELOCITY = 1.25  # m/s


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

    # ── Reset to heel-strike pose ────────────────────────────────────────

    def _reset_carry(
        self, model: MjModel, data: MjData, carry: Any
    ) -> Tuple[MjData, Any]:
        """Reset to a right-heel-strike pose with forward walking velocity.

        Why: the default keyframe is a symmetric standing pose at rest (v=0).
        EMG data was recorded during steady-state walking at ~1.25 m/s, so
        starting from rest causes an immediate mismatch that leads to falling.

        Steps:
        1. Load keyframe-0 to get correct pelvis height and wrapping-path
           constraint joints at their 0° defaults.
        2. Override the real DOF joints to a heel-strike configuration
           (asymmetric legs, right foot leading).
        3. Set pelvis_tx velocity to match EMG recording speed (1.25 m/s).
        4. The very stiff equality constraints (solimp=0.9999) will snap the
           wrapping-path joints to their correct positions within 1-2 steps —
           no need to compute them manually.
        """
        data, carry = super()._reset_carry(model, data, carry)

        # Step 1: base pose from keyframe (pelvis_ty=0.95, constraint joints)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)

        # Step 2: override real DOF joints for heel-strike asymmetry
        for jnt_name, angle in _HEELSTRIKE_POSE.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = angle

        # Step 3: lower pelvis so the right heel is exactly at ground (y=0).
        # With hip_flexion_r > 0 the leg swings forward, lifting the heel above
        # the default keyframe height. Without this correction the foot falls
        # onto the ground at reset and the braking impulse tips the body backward.
        mujoco.mj_kinematics(model, data)
        calcn_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
        if calcn_r_id >= 0:
            # World z = height (pelvis quat rotates local-y → world-z)
            heel_z = float(data.xpos[calcn_r_id, 2])
            pelvis_ty_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_ty")
            data.qpos[model.jnt_qposadr[pelvis_ty_jid]] -= heel_z

        # Step 4: forward velocity matching EMG recording conditions
        pelvis_tx_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_tx")
        if pelvis_tx_jid >= 0:
            data.qvel[model.jnt_dofadr[pelvis_tx_jid]] = _INIT_FORWARD_VELOCITY

        mujoco.mj_forward(model, data)
        return data, carry

    # ── Override free joint properties (no free joint in 2D model) ─────

    @property
    def free_jnt_qpos_id(self):
        return np.zeros((0, 7), dtype=int)

    @property
    def free_jnt_qvel_id(self):
        return np.zeros((0, 6), dtype=int)
