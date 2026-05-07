"""LocoMuJoCo environment for the gait10dof18musc 2D sagittal-plane humanoid.

We subclass LocoEnv directly..
MJX support (mjx_enabled = True)
    
"""

from pathlib import Path
from typing import Any, List, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from loco_mujoco.environments import LocoEnv


# Reference State Initialization (RSI): 4 representative walking poses.
# Covers one full gait cycle at roughly equal spacing (~0%, 30%, 60%, 80%).
# Sign convention: positive hip_flexion = leg forward; knee_angle < 0 = flexion.
# Auxiliary constraint joints (knee translations, muscle wrapping waypoints) are
# left at keyframe-0 defaults — equality constraints correct them within 1-2 steps.
_INIT_POSES = [
    # 0 — right heel strike (0% gait cycle)
    {
        "pelvis_tilt":      -0.05,
        "hip_flexion_r":    +0.30,   # right leg forward ~17°
        "knee_angle_r":     -0.05,   # nearly extended (heel contact)
        "ankle_angle_r":    +0.10,   # dorsiflexed: heel hits first
        "hip_flexion_l":    -0.15,   # left leg trailing ~9°
        "knee_angle_l":     -0.35,   # loading response flexion
        "ankle_angle_l":    -0.40,   # plantarflexed: push-off
        "lumbar_extension":  0.0,
    },
    # 1 — right mid-stance (~30%): CoM over right foot, left leg swinging forward
    {
        "pelvis_tilt":       0.00,
        "hip_flexion_r":     0.00,   # hip neutral
        "knee_angle_r":     -0.15,   # slight flexion under load
        "ankle_angle_r":    -0.08,   # mild plantarflexion (heel-to-ball)
        "hip_flexion_l":    +0.10,   # swing leg moving forward
        "knee_angle_l":     -0.50,   # clearance flexion
        "ankle_angle_l":    -0.15,   # foot clearing ground
        "lumbar_extension":  0.0,
    },
    # 2 — left heel strike (~60%): mirror of pose 0
    {
        "pelvis_tilt":      -0.05,
        "hip_flexion_r":    -0.15,
        "knee_angle_r":     -0.35,
        "ankle_angle_r":    -0.40,
        "hip_flexion_l":    +0.30,
        "knee_angle_l":     -0.05,
        "ankle_angle_l":    +0.10,
        "lumbar_extension":  0.0,
    },
    # 3 — left mid-stance (~80%): mirror of pose 1
    {
        "pelvis_tilt":       0.00,
        "hip_flexion_r":    +0.10,
        "knee_angle_r":     -0.50,
        "ankle_angle_r":    -0.15,
        "hip_flexion_l":     0.00,
        "knee_angle_l":     -0.15,
        "ankle_angle_l":    -0.08,
        "lumbar_extension":  0.0,
    },
]

# for each joint, add gaussian noise std added on top of each pose at reset.
# 0 for pelvis_tx and auxiliary joints.
_INIT_NOISE_STD = {
    "pelvis_ty":        0.02,   # ±2 cm height can vary
    "pelvis_tilt":      0.03,   # ±1.7° trunk lean
    "hip_flexion_r":    0.05,   # ±2.9° hip angle
    "knee_angle_r":     0.05,
    "ankle_angle_r":    0.03,
    "hip_flexion_l":    0.05,
    "knee_angle_l":     0.05,
    "ankle_angle_l":    0.03,
    "lumbar_extension": 0.02,
}

# Default forward walking speed m/s
_DEFAULT_INIT_VELOCITY = 1.25  


# Absolute path to the XML model 
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "humanoid"
_DEFAULT_XML = str(_MODEL_DIR / "gait10dof18musc_fixed.xml")

_ACTUATOR_NAMES = [
    "hamstrings_r", "bifemsh_r", "glut_max_r", "iliopsoas_r", "rect_fem_r", "vasti_r",
    "gastroc_r", "soleus_r", "tib_ant_r",
    "hamstrings_l", "bifemsh_l", "glut_max_l", "iliopsoas_l", "rect_fem_l", "vasti_l",
    "gastroc_l", "soleus_l", "tib_ant_l",
]


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

    Args:
        init_velocity: Forward velocity set at reset (m/s). Default 1.25 matches
            the Camargo dataset recording speed and LocoMuJoCo walk task speed.
        init_noise_scale: Multiplier on all per-joint reset noise stds. 0 = no
            noise (deterministic reset), 1 = default, >1 = more randomisation.
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
        init_velocity: float = 1.25,
        init_noise_scale: float = 1.0,
        healthy_pelvis_height_min: float = 0.7,
        healthy_pelvis_height_max: float = 1.3,
        **kwargs,
    ) -> None:
        self._init_velocity = float(init_velocity)
        self._init_noise_scale = float(init_noise_scale)
        self._healthy_pelvis_height_range = (
            float(healthy_pelvis_height_min),
            float(healthy_pelvis_height_max),
        )
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

        # ── Precompute RSI pose arrays ───────────────────────────────────
        # Build one qpos vector per initial pose (from keyframe-0 defaults).
        # Auxiliary joints keep their keyframe values; eq constraints correct them at reset.
        tmp_data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, tmp_data, 0)

        pose_arrays = []
        for pose_dict in _INIT_POSES:
            q = tmp_data.qpos.copy()
            for jnt_name, angle in pose_dict.items():
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                if jid >= 0:
                    q[int(model.jnt_qposadr[jid])] = angle
            pose_arrays.append(q)

        self._init_poses_jnp = jnp.array(np.stack(pose_arrays))  # (4, nq)
        self._n_init_poses = len(_INIT_POSES)

        # Noise std vector: non-zero only at real DOF qpos indices.
        noise_std = np.zeros(model.nq)
        for jnt_name, std in _INIT_NOISE_STD.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                noise_std[int(model.jnt_qposadr[jid])] = std
        self._init_noise_std = noise_std * self._init_noise_scale          # CPU numpy
        self._init_noise_std_jnp = jnp.array(noise_std) * self._init_noise_scale  # JAX

        # ── Precompute equality constraint parameters ────────────────────
        # All 28 auxiliary joints (knee translations, muscle wrapping waypoints)
        # are governed by joint-joint polynomial equality constraints of the form:
        #   qpos[adr1] = c0 + c1*x + c2*x² + c3*x³ + c4*x⁴
        # where x = qpos[adr2] (the driving real DOF).
        # We extract these once at init (outside JIT) so both the CPU reset
        # (_reset_carry) and the JAX reset (_mjx_reset_carry) can apply them
        # using plain array indexing — no mujoco.* calls needed at runtime.
        self._eq_constraints: list[tuple[int, int, np.ndarray]] = []
        for i in range(model.neq):
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
                j1 = int(model.eq_obj1id[i])
                j2 = int(model.eq_obj2id[i])
                adr1 = int(model.jnt_qposadr[j1])
                adr2 = int(model.jnt_qposadr[j2])
                coef = model.eq_data[i, :5].copy()  # polynomial coefficients c0..c4
                self._eq_constraints.append((adr1, adr2, coef))

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
        kwargs.setdefault("reward_type", "WalkingReward")
        kwargs.setdefault("reward_params", {"alive_bonus": 0.5})
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
        return self._healthy_pelvis_height_range

    # ── Equality constraint helpers ────────────────────────────────────

    def _apply_eq_constraints_cpu(self, data: MjData) -> None:
        """Set all constraint joint qpos values from their polynomial equality constraints.

        Called after setting real DOFs so that the auxiliary joints (knee translations,
        muscle wrapping waypoints) start at their geometrically correct positions
        rather than the keyframe defaults.
        """
        for adr1, adr2, coef in self._eq_constraints:
            x = data.qpos[adr2]
            data.qpos[adr1] = coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3 + coef[4]*x**4

    def _apply_eq_constraints_jax(self, data: Data) -> Data:
        """JAX version of _apply_eq_constraints_cpu — JIT-safe, vmap-safe.

        All loop bounds and indices are Python-level integers (computed at init),
        so JAX traces through the loop as a sequence of scalar operations.
        """
        qpos = data.qpos
        for adr1, adr2, coef in self._eq_constraints:
            x = qpos[adr2]
            val = coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3 + coef[4]*x**4
            qpos = qpos.at[adr1].set(val)
        return data.replace(qpos=qpos)

    # ── CPU reset (non-MJX, used for evaluation / rendering) ──────────

    def _reset_carry(
        self, model: MjModel, data: MjData, carry: Any
    ) -> Tuple[MjData, Any]:
        """RSI reset: randomly pick one of 4 gait phases + Gaussian noise.

        CPU version (numpy-based). Used for rendering and evaluation.

        Steps:
            1. Load keyframe-0 defaults
            2. Randomly choose one of the 4 representative gait poses
            3. Apply equality constraints (auxiliary joints)
            4. Correct pelvis height so the right heel is at z=0
            5. Add per-joint Gaussian noise
            6. Set pelvis_tx velocity to 1.25 m/s
        """
        data, carry = super()._reset_carry(model, data, carry)

        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)

        # Randomly pick one of the representative gait poses
        pose_idx = int(np.random.randint(self._n_init_poses))
        for jnt_name, angle in _INIT_POSES[pose_idx].items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = angle

        self._apply_eq_constraints_cpu(data)

        # Heel height correction: shift pelvis_ty so right heel is at z=0
        mujoco.mj_kinematics(model, data)
        calcn_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
        if calcn_r_id >= 0:
            heel_z = float(data.xpos[calcn_r_id, 2])
            pelvis_ty_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_ty")
            data.qpos[model.jnt_qposadr[pelvis_ty_jid]] -= heel_z

        # Add Gaussian noise to real DOFs (zero elsewhere via _init_noise_std)
        data.qpos += np.random.normal(0.0, self._init_noise_std)

        pelvis_tx_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_tx")
        if pelvis_tx_jid >= 0:
            data.qvel[model.jnt_dofadr[pelvis_tx_jid]] = self._init_velocity

        mujoco.mj_forward(model, data)
        return data, carry

    # ── MJX reset (JAX-based, JIT-able, used during PPOJax training) ───

    def _mjx_reset_carry(
        self, model: Model, data: Data, carry: Any
    ) -> Tuple[Data, Any]:
        """RSI reset for MJX — JIT-able, vmap-able over parallel envs.

        Randomly picks one of 4 gait poses using carry.key, then adds
        per-joint Gaussian noise. Mirrors _reset_carry() in pure JAX.

        Why random pose selection works in JIT/vmap:
            _init_poses_jnp is a (4, nq) constant. Indexing with a traced
            integer (jax.random.randint output) is a valid JAX dynamic index
            — it compiles to an XLA gather op. Each of the 2048 parallel
            envs gets an independently sampled pose_idx from its own subkey.
        """
        # ── Split carry.key for reproducible, independent randomness ─────
        key, k_pose, k_noise = jax.random.split(carry.key, 3)
        carry = carry.replace(key=key)

        # ── Pick random initial pose ─────────────────────────────────────
        pose_idx = jax.random.randint(k_pose, shape=(), minval=0, maxval=self._n_init_poses)
        data = data.replace(qpos=self._init_poses_jnp[pose_idx])

        # Apply polynomial equality constraints (auxiliary joints)
        data = self._apply_eq_constraints_jax(data)

        # ── Forward velocity ─────────────────────────────────────────────
        qvel = data.qvel.at[self._qvel_pelvis_tx].set(self._init_velocity)
        data = data.replace(qvel=qvel)

        # ── Heel height correction ───────────────────────────────────────
        data = mjx.forward(model, data)
        heel_z = data.xpos[self._body_calcn_r, 2]
        qpos = data.qpos.at[self._qpos_pelvis_ty].set(
            data.qpos[self._qpos_pelvis_ty] - heel_z
        )
        data = data.replace(qpos=qpos)

        # ── Gaussian noise on real DOFs ──────────────────────────────────
        noise = jax.random.normal(k_noise, shape=(model.nq,)) * self._init_noise_std_jnp
        data = data.replace(qpos=data.qpos + noise)

        # Final forward pass
        data = mjx.forward(model, data)

        # Parent resets reward state, domain randomizer, terminal handler, etc.
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
