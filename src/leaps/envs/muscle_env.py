"""Thin MuJoCo wrapper for the Gait10dof18musc muscle humanoid.

This does NOT depend on LocoMuJoCo — it uses raw mujoco for simplicity.
LocoMuJoCo integration (datasets, rewards, RL factory) comes later.

For now this provides:
  - Load the XML model
  - Step the simulation with 18-dim muscle commands
  - Render frames (for local visualization)
  - Record trajectories to video
"""

from pathlib import Path

import mujoco
import numpy as np

# Default path to the humanoid model XML
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "humanoid"
DEFAULT_XML = _MODEL_DIR / "gait10dof18musc.xml"


class MuscleHumanoidEnv:
    """Minimal simulation wrapper for the 18-muscle humanoid.

    Usage:
        env = MuscleHumanoidEnv()
        env.reset()
        for t in range(1000):
            env.step(action)          # action: (18,) in [0, 1]
            frame = env.render()      # (H, W, 3) uint8 array
        env.close()
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        width: int = 640,
        height: int = 480,
    ):
        xml_path = Path(xml_path) if xml_path else DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.width = width
        self.height = height

        # Renderer for offscreen frames
        self._renderer = None

    def reset(self) -> np.ndarray:
        """Reset to the default keyframe pose."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply muscle activations and step the simulation.

        Args:
            action: (18,) muscle commands in [0, 1].

        Returns:
            Observation vector (joint positions + velocities).
        """
        self.data.ctrl[:] = np.clip(action, 0.0, 1.0)
        mujoco.mj_step(self.model, self.data)
        return self._get_obs()

    def render(self) -> np.ndarray:
        """Render current frame as (H, W, 3) uint8 array."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        """Clean up renderer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_obs(self) -> np.ndarray:
        """Simple observation: joint positions + velocities."""
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    @property
    def n_actuators(self) -> int:
        return self.model.nu

    @property
    def dt(self) -> float:
        return self.model.opt.timestep
