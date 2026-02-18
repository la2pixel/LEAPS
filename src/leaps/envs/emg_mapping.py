"""Mapping from 11-channel EMG activations to 18-muscle humanoid actuator commands.

The Camargo dataset records 11 muscles (right leg only).
The Gait10dof18musc model has 9 muscles per leg (18 total, bilateral).

This module handles:
  1. Mapping EMG channels → model actuators (some are 1:1, some are combined)
  2. Mirroring right-leg EMG to left leg (assuming symmetric gait)
  3. Setting default activations for muscles without EMG data (iliopsoas)
"""

import numpy as np

# The 11 EMG channels in the Camargo dataset (order matches preprocessing)
EMG_CHANNELS = [
    "gastrocmed",          # 0
    "tibialisanterior",    # 1
    "soleus",              # 2
    "vastusmedialis",      # 3
    "vastuslateralis",     # 4
    "rectusfemoris",       # 5
    "bicepsfemoris",       # 6
    "semitendinosus",      # 7
    "gracilis",            # 8
    "gluteusmedius",       # 9
    "rightexternaloblique",# 10
]

# The 18 actuators in gait10dof18musc.xml (order matches XML)
MODEL_ACTUATORS = [
    "hamstrings_r",   # 0
    "bifemsh_r",      # 1
    "glut_max_r",     # 2
    "iliopsoas_r",    # 3
    "rect_fem_r",     # 4
    "vasti_r",        # 5
    "gastroc_r",      # 6
    "soleus_r",       # 7
    "tib_ant_r",      # 8
    "hamstrings_l",   # 9
    "bifemsh_l",      # 10
    "glut_max_l",     # 11
    "iliopsoas_l",    # 12
    "rect_fem_l",     # 13
    "vasti_l",        # 14
    "gastroc_l",      # 15
    "soleus_l",       # 16
    "tib_ant_l",      # 17
]


class EMGToMuscleMapper:
    """Maps 11-channel EMG activations to 18 model muscle commands.

    Mapping rules:
        gastrocmed         → gastroc_r/l
        tibialisanterior   → tib_ant_r/l
        soleus             → soleus_r/l
        vastusmed + vastlat → vasti_r/l  (average)
        rectusfemoris      → rect_fem_r/l
        bicepsfemoris      → hamstrings_r/l, bifemsh_r/l (split equally)
        semitendinosus     → hamstrings_r/l (added to bicepsfemoris contribution)
        gluteusmedius      → glut_max_r/l
        gracilis           → dropped (no match)
        externaloblique    → dropped (no match)
        iliopsoas_r/l      → set to default_activation

    The right-leg EMG is mirrored to the left leg with an optional
    phase_offset for gait cycle shifting (left leg is ~50% offset).
    """

    def __init__(self, default_activation: float = 0.05):
        self.default_activation = default_activation

    def map_emg_to_muscles(self, emg: np.ndarray) -> np.ndarray:
        """Map a single 11-dim EMG vector to 18-dim actuator command.

        Args:
            emg: Shape (11,) EMG activations, values in [0, ~1].

        Returns:
            Shape (18,) muscle commands, clipped to [0, 1].
        """
        act = np.full(18, self.default_activation)

        # Right leg
        hamstring_combined = (emg[6] + emg[7]) / 2  # bicepsfemoris + semitendinosus
        act[0] = hamstring_combined                   # hamstrings_r
        act[1] = emg[6] * 0.5                        # bifemsh_r (biceps femoris short head)
        act[2] = emg[9]                               # glut_max_r ← gluteusmedius
        # act[3] = default                            # iliopsoas_r (no EMG)
        act[4] = emg[5]                               # rect_fem_r ← rectusfemoris
        act[5] = (emg[3] + emg[4]) / 2               # vasti_r ← avg(vastusmed, vastuslat)
        act[6] = emg[0]                               # gastroc_r ← gastrocmed
        act[7] = emg[2]                               # soleus_r
        act[8] = emg[1]                               # tib_ant_r ← tibialisanterior

        # Left leg — mirror right leg
        act[9:18] = act[0:9]

        return np.clip(act, 0.0, 1.0)

    def map_batch(self, emg_batch: np.ndarray) -> np.ndarray:
        """Map a batch of EMG vectors.

        Args:
            emg_batch: Shape (N, 11).

        Returns:
            Shape (N, 18).
        """
        return np.array([self.map_emg_to_muscles(e) for e in emg_batch])

    def map_stride_with_phase_offset(
        self, emg_stride: np.ndarray, offset_pct: int = 50,
    ) -> np.ndarray:
        """Map a full stride (101, 11) with phase-shifted left leg.

        During walking, the left leg is ~50% of gait cycle behind the right.
        This rolls the left-leg activations by offset_pct timepoints.

        Args:
            emg_stride: Shape (101, 11) — one gait cycle of EMG.
            offset_pct: Phase offset in % gait cycle (default 50).

        Returns:
            Shape (101, 18) — full bilateral muscle commands.
        """
        stride_len = emg_stride.shape[0]
        act = np.zeros((stride_len, 18))

        for t in range(stride_len):
            right_emg = emg_stride[t]
            # Left leg uses the same EMG but shifted in time
            t_left = (t + offset_pct) % stride_len
            left_emg = emg_stride[t_left]

            # Map right leg
            act[t, :9] = self.map_emg_to_muscles(right_emg)[:9]
            # Map left leg from shifted EMG
            act[t, 9:] = self.map_emg_to_muscles(left_emg)[:9]

        return np.clip(act, 0.0, 1.0)
