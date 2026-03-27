"""LocoMuJoCo environment for the gait10dof18musc 2D sagittal-plane humanoid.

This model has 18 muscle actuators (9 per leg), 10 DOFs, and NO free joint.
The root is defined by three separate joints: pelvis_tx (slide), pelvis_ty (slide),
pelvis_tilt (hinge). We subclass LocoEnv directly (not BaseSkeleton) because
BaseSkeleton assumes the full 92-muscle skeleton with arms, box feet, etc.

MJX support (mjx_enabled = True):
    MJX is MuJoCo's JAX backend that allows vmapping hundreds of envs in
    parallel on GPU. PPOJax requires MJX — it vmaps env.step() over 2048
    parallel environments, each running on different portions of the GPU.
    The MJX reset uses _mjx_reset_carry() (JAX-compatible), which mirrors
    the CPU _reset_carry() using jnp array operations instead of mujoco.*
    calls.
"""

from pathlib import Path
from typing import Any, List, Tuple, Union

import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from loco_mujoco.environments import LocoEnv


# Real DOF joint angles for a right-heel-strike initial pose.
# The wrapping-path constraint joints (knee translations, vasti/gastroc/iliopsoas
# waypoints) are LEFT at their keyframe-0 defaults — the very stiff equality
# constraints (solimp=0.9999) snap them to the correct positions within 1-2 steps.
# Sign convention: positive hip_flexion = leg forward; knee_angle < 0 = flexion.
_HEELSTRIKE_POSE = {
    "pelvis_tilt":      -0.05,   # slight forward lean (prevents backward toppling)
    "hip_flexion_r":    +0.30,   # right leg forward ~17° (heel contact)
    "knee_angle_r":     -0.05,   # nearly fully extended (just before heel strike)
    "ankle_angle_r":    +0.10,   # dorsiflexed ~6° so heel hits ground first
    "hip_flexion_l":    -0.15,   # left leg behind ~9° (entering push-off)
    "knee_angle_l":     -0.35,   # 20° knee flexion (loading response)
    "ankle_angle_l":    -0.40,   # plantarflexed ~23° (propulsion phase)
    "lumbar_extension":  0.0,    # neutral spine
}

# Forward walking speed matching EMG recording conditions.
_INIT_FORWARD_VELOCITY = 1.25  # m/s (Camargo dataset: level-ground 1.25 m/s)


# Absolute path to the XML model (MyoConverter output of OpenSim Gait2392)
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "humanoid"
_DEFAULT_XML = str(_MODEL_DIR / "gait10dof18musc.xml")

# 18 muscle actuators in XML order (9 right leg, 9 left leg)
_ACTUATOR_NAMES = [
    "hamstrings_r", "bifemsh_r", "glut_max_r", "iliopsoas_r", "rect_fem_r", "vasti_r",
    "gastroc_r", "soleus_r", "tib_ant_r",
    "hamstrings_l", "bifemsh_l", "glut_max_l", "iliopsoas_l", "rect_fem_l", "vasti_l",
    "gastroc_l", "soleus_l", "tib_ant_l",
]


# Foot geom names in the XML (from contact pair list in gait10dof18musc.xml).
# These are the geoms that touch the ground-plane. For MJX we keep only the
# most important ones (heel + toe) to keep the contact list small.
_FOOT_GEOMS = [
    "calcn_r_geom_1",  # right heel
    "toes_r_geom_1",   # right toe
    "calcn_l_geom_1",  # left heel
    "toes_l_geom_1",   # left toe
]
_GROUND_GEOM = "ground-plane"


class Gait10dof18Musc(LocoEnv):
    """LocoMuJoCo environment for the 2D sagittal-plane gait10dof18musc humanoid.

    18 muscle actuators, 10 DOFs, no free joint.
    Observation: all joints (pos + vel) from spec = 76 dims (38 joints × 2).

    With mjx_enabled = True, this env supports PPOJax's VecEnv (jax.vmap over
    2048 parallel envs on GPU). The MJX reset replicates the heel-strike pose
    using pure JAX array operations (_mjx_reset_carry), which is JIT-able and
    vmap-able without any Python-side MuJoCo calls.
    """

    # MJX enabled: required for PPOJax's VecEnv (jax.vmap over parallel envs).
    # When True, LocoEnv.__init__ calls Mjx.__init__ which creates mjx.Data and
    # mjx.Model (JAX-backed versions of MjData/MjModel).
    mjx_enabled = True

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

        # MJX requires simplified contact model (no mesh-mesh margin/gap).
        # Must be called BEFORE super().__init__() which builds MjModel + mjx.Model.
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(
            spec=spec,
            actuation_spec=actuation_spec,
            observation_spec=observation_spec,
            **kwargs,
        )

        # ── Pre-compute joint indices for MJX reset ─────────────────────
        # mujoco.mj_name2id() cannot be called inside jax.jit (it's a Python
        # side-effect). We call it here (at construction time, outside jit)
        # and store the integer addresses as Python attributes. These become
        # compile-time constants when captured inside _mjx_reset_carry.

        model = self._model  # the CPU MjModel built during super().__init__

        def _qpos_idx(jnt_name: str) -> int:
            """Return the qpos array address for a named joint."""
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            assert jid >= 0, f"Joint '{jnt_name}' not found in model"
            return int(model.jnt_qposadr[jid])

        def _qvel_idx(jnt_name: str) -> int:
            """Return the qvel array address for a named joint (dof address)."""
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            assert jid >= 0, f"Joint '{jnt_name}' not found in model"
            return int(model.jnt_dofadr[jid])

        def _body_id(body_name: str) -> int:
            """Return the body index for a named body."""
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            assert bid >= 0, f"Body '{body_name}' not found in model"
            return int(bid)

        # qpos indices for each real DOF we want to set at reset
        self._qpos_pelvis_tilt   = _qpos_idx("pelvis_tilt")
        self._qpos_pelvis_ty     = _qpos_idx("pelvis_ty")
        self._qpos_hip_r         = _qpos_idx("hip_flexion_r")
        self._qpos_knee_r        = _qpos_idx("knee_angle_r")
        self._qpos_ankle_r       = _qpos_idx("ankle_angle_r")
        self._qpos_hip_l         = _qpos_idx("hip_flexion_l")
        self._qpos_knee_l        = _qpos_idx("knee_angle_l")
        self._qpos_ankle_l       = _qpos_idx("ankle_angle_l")
        self._qpos_lumbar        = _qpos_idx("lumbar_extension")

        # qvel index for forward velocity (pelvis_tx is a slide joint → 1 DOF)
        self._qvel_pelvis_tx     = _qvel_idx("pelvis_tx")

        # Body index for right heel — used to compute ground contact height
        self._body_calcn_r       = _body_id("calcn_r")

    # ── Class methods ──────────────────────────────────────────────────

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        return _DEFAULT_XML

    @classmethod
    def generate(cls, task=None, **kwargs):
        """Create environment with sensible defaults for RL training."""
        # HeightJointTerminalStateHandler: terminates if pelvis_ty leaves [0.5, 1.3]
        # TargetXVelocityReward: r = exp(-(v_x - 1.25)^2), reward for walking at ~1.25 m/s
        # NoGoal: no additional goal observation (simpler obs space for PPO)
        kwargs.setdefault("terminal_state_type", "HeightJointTerminalStateHandler")
        kwargs.setdefault("goal_type", "NoGoal")
        kwargs.setdefault("reward_type", "TargetXVelocityReward")
        kwargs.setdefault("reward_params", {"target_velocity": _INIT_FORWARD_VELOCITY})
        return cls(**kwargs)

    # ── MJX: simplify contacts for parallel simulation ─────────────────

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        """Simplify the contact model for MJX (JAX-based parallel simulation).

        MJX limitations vs CPU MuJoCo:
          1. It builds convex hulls for ALL mesh geoms, even disabled ones.
             Some MyoConverter meshes have degenerate/empty hulls → IndexError.
          2. It performs poorly with many contact pairs.
          3. Non-zero margin/gap on mesh geoms raises NotImplementedError.

        Fix:
          Step 1: Replace ALL mesh geoms with spheres (avoids convex hull build).
                  Contacts are disabled so shape doesn't affect simulation.
          Step 2: Zero margin/gap and disable contacts on all geoms.
          Step 3: Re-enable only the 4 foot-floor pairs as plane-sphere contacts
                  (MJX handles plane-sphere natively and efficiently).
        """
        foot_geom_names = set(_FOOT_GEOMS)

        for g in spec.geoms:
            g.margin = 0.0
            g.gap = 0.0
            g.contype = 0
            g.conaffinity = 0
            # Replace mesh geoms with tiny spheres so MJX skips convex hull build.
            # Foot geoms get a larger radius so plane-sphere contact is plausible.
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size = [0.03 if g.name in foot_geom_names else 0.001, 0.0, 0.0]

        # Re-enable foot-floor contact pairs (plane-sphere, fully supported by MJX).
        for foot_geom in _FOOT_GEOMS:
            spec.add_pair(geomname1=_GROUND_GEOM, geomname2=foot_geom)

        return spec

    # ── Spec methods ───────────────────────────────────────────────────

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[ObservationType]:
        joint_names = [j.name for j in spec.joints]
        obs_pos = [ObservationType.JointPos(f"q_{j}", xml_name=j) for j in joint_names]
        obs_vel = [ObservationType.JointVel(f"dq_{j}", xml_name=j) for j in joint_names]
        return obs_pos + obs_vel

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        return list(_ACTUATOR_NAMES)

    # ── Info properties ────────────────────────────────────────────────

    @info_property
    def root_body_name(self) -> str:
        # The pelvis is the root body. Used for height-based terminal state.
        return "pelvis"

    @info_property
    def upper_body_xml_name(self) -> str:
        return "torso"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        # We don't have a free joint, but TargetXVelocityReward uses this
        # to find the joint whose velocity[0] is the forward speed.
        # pelvis_tx is a slide joint: qvel[dof_adr] = forward velocity.
        return "pelvis_tx"

    @info_property
    def root_height_healthy_range(self) -> tuple:
        # pelvis_ty in the default keyframe is ~0.95 m. Healthy range:
        #   0.5 m: would be nearly on the ground (fallen)
        #   1.3 m: unphysically high (should never happen)
        return (0.5, 1.3)

    # ── CPU reset (non-MJX, used for evaluation / rendering) ──────────

    def _reset_carry(
        self, model: MjModel, data: MjData, carry: Any
    ) -> Tuple[MjData, Any]:
        """Reset to a right-heel-strike pose with forward walking velocity.

        CPU version (numpy-based). Used when running the env without MJX
        (e.g. for recording videos, interactive inspection).

        Why we need a custom reset:
            The default keyframe is a symmetric standing pose at v=0.
            EMG data was recorded during steady-state walking at 1.25 m/s.
            Starting from rest causes an immediate state mismatch → fall.

        Steps:
            1. Load keyframe-0 (sets pelvis_ty=0.95, constraint joint 0°)
            2. Set real DOFs to heel-strike asymmetric pose
            3. Correct pelvis height so right heel is at ground (z=0)
            4. Set pelvis_tx velocity to 1.25 m/s
        """
        data, carry = super()._reset_carry(model, data, carry)

        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)

        for jnt_name, angle in _HEELSTRIKE_POSE.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = angle

        # Run forward kinematics to get body positions before we correct height
        mujoco.mj_kinematics(model, data)
        calcn_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
        if calcn_r_id >= 0:
            heel_z = float(data.xpos[calcn_r_id, 2])
            pelvis_ty_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_ty")
            data.qpos[model.jnt_qposadr[pelvis_ty_jid]] -= heel_z

        pelvis_tx_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_tx")
        if pelvis_tx_jid >= 0:
            data.qvel[model.jnt_dofadr[pelvis_tx_jid]] = _INIT_FORWARD_VELOCITY

        mujoco.mj_forward(model, data)
        return data, carry

    # ── MJX reset (JAX-based, JIT-able, used during PPOJax training) ───

    def _mjx_reset_carry(
        self, model: Model, data: Data, carry: Any
    ) -> Tuple[Data, Any]:
        """JAX-JIT-able heel-strike reset for MJX (used by PPOJax VecEnv).

        This mirrors _reset_carry() but uses pure jnp array operations so
        it can be compiled with jax.jit and vmapped over thousands of parallel
        environments.

        Key difference from CPU reset:
            - No mujoco.mj_* calls (not JAX-compatible)
            - Uses .replace() (functional update — MJX Data is immutable)
            - mjx.forward() replaces mujoco.mj_kinematics() + mj_forward()
            - Joint indices (self._qpos_*, self._qvel_*) were computed at
              construction time and are treated as compile-time constants.

        Math for heel-height correction:
            After setting qpos to the heel-strike pose, the right heel body
            (calcn_r) may be above or below z=0 (ground plane). We read
            data.xpos[calcn_r_id, 2] after mjx.forward(), then subtract
            that height from pelvis_ty so the heel is exactly at z=0.

            pelvis_ty_new = pelvis_ty_old - xpos[calcn_r, z-axis]
        """
        # ── Set heel-strike joint angles ─────────────────────────────────
        # data.qpos is a JAX array (shape [nq,]). .at[idx].set(val) is the
        # JAX functional equivalent of qpos[idx] = val (immutable update).
        qpos = data.qpos
        qpos = qpos.at[self._qpos_pelvis_tilt].set(-0.05)
        qpos = qpos.at[self._qpos_hip_r].set(+0.30)
        qpos = qpos.at[self._qpos_knee_r].set(-0.05)
        qpos = qpos.at[self._qpos_ankle_r].set(+0.10)
        qpos = qpos.at[self._qpos_hip_l].set(-0.15)
        qpos = qpos.at[self._qpos_knee_l].set(-0.35)
        qpos = qpos.at[self._qpos_ankle_l].set(-0.40)
        qpos = qpos.at[self._qpos_lumbar].set(0.0)
        data = data.replace(qpos=qpos)

        # ── Set forward walking velocity ─────────────────────────────────
        # pelvis_tx is a slide joint → its single DOF address gives forward vel.
        qvel = data.qvel
        qvel = qvel.at[self._qvel_pelvis_tx].set(_INIT_FORWARD_VELOCITY)
        data = data.replace(qvel=qvel)

        # ── Run forward kinematics to compute body positions ─────────────
        # mjx.forward() propagates qpos → body positions (xpos), site frames,
        # geom positions, etc. Equivalent to mujoco.mj_kinematics + mj_forward
        # but JAX-differentiable and JIT-compilable.
        data = mjx.forward(model, data)

        # ── Heel-height correction ───────────────────────────────────────
        # After setting the leg angles, the right heel body (calcn_r) may
        # be above the floor (z > 0) because hip_flexion_r > 0 shifts the
        # entire leg forward and upward relative to the keyframe. We read
        # the heel's world-z from xpos (already computed by mjx.forward above)
        # and shift pelvis_ty down by that amount.
        heel_z = data.xpos[self._body_calcn_r, 2]  # JAX scalar: heel height in world frame
        qpos = data.qpos.at[self._qpos_pelvis_ty].set(
            data.qpos[self._qpos_pelvis_ty] - heel_z
        )
        data = data.replace(qpos=qpos)

        # Final forward pass with corrected pelvis height
        data = mjx.forward(model, data)

        # ── Run parent handlers (terminal state, reward, etc.) ────────────
        # The parent _mjx_reset_carry resets stateful objects like the reward
        # function state, domain randomizer, and initial state handler.
        data, carry = super()._mjx_reset_carry(model, data, carry)
        return data, carry

    # ── Override free joint properties (no free joint in 2D model) ─────

    @property
    def free_jnt_qpos_id(self):
        # The base class assumes a 6-DOF free joint (7 qpos values: xyz + quat).
        # Our model uses separate slide/hinge joints, so there is no free joint.
        # Returning empty arrays prevents the base class from trying to index
        # into qpos with free-joint offsets (which would crash or produce NaN).
        return np.zeros((0, 7), dtype=int)

    @property
    def free_jnt_qvel_id(self):
        return np.zeros((0, 6), dtype=int)
