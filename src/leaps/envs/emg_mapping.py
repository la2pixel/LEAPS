"""Mapping from the 11 EMG channels to SCONE/Hyfydy muscle actuators.

H0918: 9 muscles/leg, H1622: 11, H2190: 45.

A model muscle gets EMG only if it is the recorded muscle, or a lumped
muscle whose recorded parts are averaged (vasti_r = mean of vastus medialis
and lateralis). Signals are not borrowed from other muscles. Everything else
is 0. Only the right leg is recorded; mirror_left=True copies each "_r" row
onto its "_l" counterpart.
"""

import numpy as np

EMG_CHANNELS = [
    "gastrocmed",
    "tibialisanterior",
    "soleus",
    "vastusmedialis",
    "vastuslateralis",
    "rectusfemoris",
    "bicepsfemoris",
    "semitendinosus",
    "gracilis",
    "gluteusmedius",
    "rightexternaloblique",
]

# excluded: gracilis, gluteusmedius, rightexternaloblique -- no matching muscle
# missing: iliopsoas_r, glut_max_r, bifemsh_r -- no EMG channel is a real substitute
H0918_MAP = {
    "gastrocmed": "gastroc_r",
    "tibialisanterior": "tib_ant_r",
    "soleus": "soleus_r",
    "rectusfemoris": "rect_fem_r",
    "vastusmedialis": "vasti_r",       # vasti_r = avg(vastusmed, vastuslat)
    "vastuslateralis": "vasti_r",
    "bicepsfemoris": "hamstrings_r",   # hamstrings_r = avg(bicepsfemoris, semiten)
    "semitendinosus": "hamstrings_r",
}


H1622_MAP = {
    **H0918_MAP,
    "gluteusmedius": "glut_med_r",
}
# missing: iliopsoas_r, glut_max_r, bifemsh_r, add_mag_r 

H2190_MAP = {
    "gastrocmed": "gas_med_r",
    "tibialisanterior": "tib_ant_r",
    "soleus": "soleus_r",
    "vastusmedialis": "vas_med_r",
    "vastuslateralis": "vas_lat_r",
    "rectusfemoris": "rect_fem_r",
    "bicepsfemoris": "bifemlh_r",       # long head -- the head surface EMG sits over
    "semitendinosus": "semiten_r",
    "gracilis": "grac_r",
    "rightexternaloblique": "ext_obl_r",
    "gluteusmedius": "glut_med1_r",     # proximal compartment -- closest to standard surface-EMG electrode placement
}
# missing: bifemsh_r, semimem_r, vas_int_r, glut_max1/2/3_r, add_*_r,
# iliacus_r, psoas_r, int_obl_r -- no EMG channel for any of these

MODEL_MAPS = {
    "h0918": H0918_MAP,
    "h1622": H1622_MAP,
    "h2190": H2190_MAP,
}


class EMGToMuscleMapper:
    """(n_actuators, 11) weight matrix; a row is zero, one-hot, or an even
    split over the channels that make up a lumped muscle."""

    def __init__(
        self, model_name: str, actuator_names: list[str],
        mirror_left: bool = False, null_prior: bool = False,
    ):
        channel_to_muscle = MODEL_MAPS[model_name]
        self.actuator_names = actuator_names
        self.weights = np.zeros((len(actuator_names), len(EMG_CHANNELS)))

        # null prior: weights stay zero, so a_hat = 0
        if null_prior:
            return

        muscle_to_channels: dict[str, list[str]] = {}
        for channel, muscle in channel_to_muscle.items():
            muscle_to_channels.setdefault(muscle, []).append(channel)

        for muscle, channels in muscle_to_channels.items():
            if muscle not in actuator_names:
                continue
            row = actuator_names.index(muscle)
            for channel in channels:
                self.weights[row, EMG_CHANNELS.index(channel)] = 1.0 / len(channels)

        # left leg gets the same row as the right, with no phase shift
        if mirror_left:
            for i, muscle in enumerate(actuator_names):
                if not muscle.endswith("_r") or not self.weights[i].any():
                    continue
                left_muscle = muscle[:-2] + "_l"
                if left_muscle not in actuator_names:
                    continue
                self.weights[actuator_names.index(left_muscle)] = self.weights[i]

    def map_emg_to_muscles(self, emg: np.ndarray) -> np.ndarray:
        """(11,) EMG -> (n_actuators,) activations; unmapped actuators are 0."""
        return self.weights @ emg
