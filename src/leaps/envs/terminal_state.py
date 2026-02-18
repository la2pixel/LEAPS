"""Custom terminal state handler for the 2D sagittal-plane gait10dof18musc model.

The default HeightBasedTerminalStateHandler assumes a free joint (reads qpos[2] for z-height).
Our model uses separate slide joints (pelvis_tx, pelvis_ty, pelvis_tilt), so we check
pelvis_ty directly for the height-based terminal condition.
"""

from types import ModuleType
from typing import Any, Dict, Tuple, Union

import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from loco_mujoco.core.utils import mj_jntname2qposid
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class HeightJointTerminalStateHandler(TerminalStateHandler):
    """Terminal state handler that checks pelvis_ty joint for height-based termination.

    Used for 2D sagittal-plane models where the root is defined by separate
    slide/hinge joints rather than a free joint.
    """

    def __init__(self, env: Any, **handler_config: Dict[str, Any]):
        super().__init__(env, **handler_config)
        self.root_height_range = self._info_props["root_height_healthy_range"]
        self.pelvis_ty_qpos_id = int(
            np.array(mj_jntname2qposid("pelvis_ty", env._model)).flat[0]
        )

    def reset(
        self,
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[Union[MjData, Data], Any]:
        assert_backend_is_supported(backend)
        return data, carry

    def is_absorbing(
        self,
        env: Any,
        obs: np.ndarray,
        info: Dict[str, Any],
        data: MjData,
        carry: Any,
    ) -> Union[bool, Any]:
        return self._is_absorbing_compat(env, obs, info, data, carry, backend=np)

    def mjx_is_absorbing(
        self,
        env: Any,
        obs: jnp.ndarray,
        info: Dict[str, Any],
        data: Data,
        carry: Any,
    ) -> Union[bool, Any]:
        return self._is_absorbing_compat(env, obs, info, data, carry, backend=jnp)

    def _is_absorbing_compat(
        self,
        env: Any,
        obs: Union[np.ndarray, jnp.ndarray],
        info: Dict[str, Any],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Union[bool, Any]:
        pelvis_height = data.qpos[self.pelvis_ty_qpos_id]
        height_cond = backend.logical_or(
            backend.less(pelvis_height, self.root_height_range[0]),
            backend.greater(pelvis_height, self.root_height_range[1]),
        )
        return height_cond, carry
