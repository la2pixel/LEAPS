"""Mapping from 11-channel EMG activations to 18-muscle humanoid actuator commands.

The Camargo dataset records 11 muscles (right leg only).
The Gait10dof18musc model has 9 muscles per leg (18 total, bilateral).

This module handles:
  1. Mapping EMG channels → model actuators (some are 1:1, some are combined)
  2. Mirroring right-leg EMG to left leg (assuming symmetric gait)
  3. Setting default activations for muscles without EMG data (iliopsoas)

Design notes
------------
gluteusmedius (EMG ch 9) is a hip *abductor* (frontal-plane stabiliser during stance).
The model is 2D sagittal-plane only — there is no hip-abduction DOF. glut_max is the
primary hip *extensor* and is a completely different muscle. Driving glut_max at full
strength with the abductor signal creates a large, wrong hip-extension torque every step.
We therefore apply a small partial-proxy weight (0.3) to acknowledge that gluteal muscles
co-activate during stance, while preventing runaway hip extension.

iliopsoas has no EMG channel in the Camargo dataset but is critical for the swing phase —
it is the primary hip *flexor* and fires at ~15–25% MVC throughout the gait cycle. Setting
its default to near-zero (0.05) means the leg cannot swing forward at all. A default of
0.20 is biomechanically motivated as a conservative resting level.
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
        gastrocmed         → gastroc_r/l           (1:1)
        tibialisanterior   → tib_ant_r/l           (1:1)
        soleus             → soleus_r/l             (1:1)
        vastusmed + vastlat → vasti_r/l             (average of two heads)
        rectusfemoris      → rect_fem_r/l           (1:1, biarticular quad)
        bicepsfemoris      → hamstrings_r/l         (combined with semitendinosus)
        semitendinosus     → hamstrings_r/l         (combined with bicepsfemoris)
        bicepsfemoris      → bifemsh_r/l * 0.5      (BF short head, monoarticular)
        gluteusmedius      → glut_max_r/l * 0.3     (partial proxy only — see module note)
        gracilis           → dropped (no match in 18-muscle model)
        externaloblique    → dropped (no match in 18-muscle model)
        iliopsoas_r/l      → default_activation     (no EMG; default=0.05, minimal baseline)

    The right-leg EMG is mirrored to the left leg with an optional
    phase_offset for gait cycle shifting (left leg is ~50% offset).
    """

    def __init__(self, default_activation: float = 0.05):
        # Iliopsoas fires primarily during swing (~50–80% gait cycle).
        # During stance it is largely silent. A constant 0.20 default creates
        # permanent hip-flexion torque (433 N) that overwhelms the under-driven
        # glut_max proxy (≤150 N), keeping the hip perpetually forward-flexed
        # and preventing knee extension. 0.05 is a minimal baseline.
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
        # Hamstrings group: average of biarticular biceps femoris long head + semitendinosus.
        # Both muscles act together as hip extensors + knee flexors during loading response.
        hamstring_combined = (emg[6] + emg[7]) / 2
        act[0] = hamstring_combined                   # hamstrings_r (biarticular)

        # Biceps femoris short head: monoarticular knee flexor only. It co-activates with
        # the long head but at a lower level — 50% of BF signal is a reasonable proxy.
        act[1] = emg[6] * 0.5                        # bifemsh_r

        # Glut max: primary hip extensor. No direct EMG. Gluteusmedius (ch 9) is a hip
        # ABDUCTOR, not an extensor, but co-activates with glut_max during loading
        # response (0–20% gait). We combine a constant minimum floor (0.25 — motivated
        # by glut_max firing at ~25–40% MVC during early stance in level-ground walking)
        # with a small contribution from gluteusmedius to capture the burst timing.
        # This replaces the previous proxy-only (0.3×) approach that gave ≤0.09
        # activation — insufficient to overcome the default iliopsoas hip-flexion torque.
        act[2] = np.clip(0.25 + emg[9] * 0.15, 0.0, 1.0)  # glut_max_r

        # Iliopsoas: primary hip flexor, essential for swing phase. No EMG channel exists.
        # Remains at default_activation (0.20) — see __init__ comment.
        # act[3] = default_activation                 # iliopsoas_r

        act[4] = emg[5]                               # rect_fem_r ← rectusfemoris (1:1)
        act[5] = (emg[3] + emg[4]) / 2               # vasti_r ← avg(vastusmed, vastuslat)
        act[6] = emg[0]                               # gastroc_r ← gastrocmed (1:1)
        act[7] = emg[2]                               # soleus_r (1:1)
        act[8] = emg[1]                               # tib_ant_r ← tibialisanterior (1:1)

        # Left leg — mirror right leg
        act[9:18] = act[0:9]

        return np.clip(act, 0.0, 1.0)

    def to_loco_action(self, muscle_act: np.ndarray) -> np.ndarray:
        """Convert muscle activations [0, 1] to LocoMuJoCo action range [-1, 1].

        LocoMuJoCo's DefaultControl maps actions in [-1, 1] to ctrl in [0, 1]
        via: ctrl = 0.5*(action + 1). Muscle activations from EMG are in [0, 1].
        This function inverts DefaultControl's mapping so that EMG activations
        produce the correct ctrl values when fed through DefaultControl.

            ctrl = 0.5*(action+1)  →  action = 2*ctrl - 1

        Args:
            muscle_act: Shape (..., 18), muscle activations in [0, 1].

        Returns:
            Same shape, actions in [-1, 1] for use with env.step().
        """
        return 2.0 * muscle_act - 1.0

    def map_batch(self, emg_batch: np.ndarray) -> np.ndarray:
        """Map a batch of EMG vectors.

        Args:
            emg_batch: Shape (N, 11).

        Returns:
            Shape (N, 18).
        """
        return np.array([self.map_emg_to_muscles(e) for e in emg_batch])

    def map_stride_with_phase_offset(
        self, emg_stride: np.ndarray, offset_pct: float = 50.0,
    ) -> np.ndarray:
        """Map a full stride (T, 11) with phase-shifted left leg.

        During walking the left leg is ~50% of the gait cycle behind the right.
        This function time-shifts the left-leg activations accordingly.

        Args:
            emg_stride: Shape (T, 11) — one gait cycle of right-leg EMG.
                        Typically T=101 (0–100% of gait cycle, inclusive).
            offset_pct: Phase offset as a true percentage of gait cycle (default 50.0).
                        50% means left leg is half a stride behind right leg.

        Returns:
            Shape (T, 18) — full bilateral muscle commands, clipped to [0, 1].
        """
        stride_len = emg_stride.shape[0]
        # Convert percentage to timepoint offset (rounds to nearest sample).
        # For stride_len=101, offset_pct=50 → offset_steps=50 (exactly half).
        offset_steps = int(round(offset_pct / 100.0 * stride_len))
        act = np.zeros((stride_len, 18))

        for t in range(stride_len):
            right_emg = emg_stride[t]
            # Left leg reads from the same EMG array, shifted forward in time.
            # The modulo wraps around at the end of the cycle (assumes periodicity).
            t_left = (t + offset_steps) % stride_len
            left_emg = emg_stride[t_left]

            # Map right leg (indices 0–8) and left leg (indices 9–17)
            act[t, :9] = self.map_emg_to_muscles(right_emg)[:9]
            act[t, 9:] = self.map_emg_to_muscles(left_emg)[:9]

        return np.clip(act, 0.0, 1.0)
