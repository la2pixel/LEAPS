"""
Mapping from 11-channel EMG activations to SCONE/Hyfydy muscle actuators.

1. H0918 -- 9 muscles/leg
2. H1622 -- 11 muscles/leg
3. H2190 -- 45 muscles/leg

A model muscle gets EMG only if it's a single real muscle (1:1), or a lumped composite whose EMG-recorded constituents we average (e.g. vasti_r = avg(vastusmed,
vastuslat), since both are parts of the vasti group). We don't borrow a different muscle's signal (e.g. bicepsfemoris, which
records the long head, is not used for bifemsh_r, the short head). Everything else is 0; the frozen-decoder is expected to fill it in.

Right leg only -> left leg is left at 0 by default (policy handles this with
other missing muscles), unless `mirror_left=True` is passed to
`EMGToMuscleMapper`, in which case each mapped "_r" muscle's "_l" counterpart
gets an identical (synchronous, no phase-shift) copy of the same weights.
"""

from typing import Optional

import numpy as np

# in the order of processing
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

# H2190 coverage-vs-constraint-scope confound (2026-08-01): mapped_residual_weight
# (see LatentActionPriorWrapper) is applied to exactly the actuators with a real
# EMG channel, so on h2190 (22% coverage) the tight constraint binds only a small
# minority of the action space -- the free residual has 78% of it, including the
# biomechanically dominant proximal muscles, to route around whatever the
# constrained minority does. That's a different regime than h0918/h1622, where
# coverage (and therefore constraint scope) is the *majority* of actuators. This
# conflates "does real EMG content transfer at low coverage" with "does the
# constraint mechanism have any grip once scoped this narrowly" -- two different
# questions the h2190 hard-constraint runs (RUNS_INDEX.md) can't currently tell
# apart.
#
# H2190_SCOPE_PROXY extends the tight-constraint *scope* (not the EMG-content
# coverage) to a fraction matched to h0918/h1622 (~65-69%), by giving each proxy
# muscle a content target copied from the real-EMG-mapped muscle it's the
# strongest anatomical synergist of. This is the same "copy a channel onto an
# anatomically related actuator" mechanism mirror_left already uses across
# left/right -- applied here within a leg, muscle-to-muscle, instead. No
# moment-arm accessor exists on this SCONE build to derive synergist grouping
# automatically (see reference_sconepy_api note), so this is hand-built from
# muscle name/anatomy, same as H0918_MAP/H1622_MAP/H2190_MAP themselves.
#
# Every donor below is a directly EMG-mapped muscle (never another proxy), so
# proxy weights are well-defined regardless of dict iteration order. Each entry
# only included where the anatomical relationship is a genuinely strong one:
#   - same muscle, different head/compartment (vas_int, gas_lat, bifemsh,
#     semimem, glut_med2/3) -- the strongest possible justification, these
#     co-activate as one functional unit in essentially all synergy literature.
#   - classic single-joint-function synergist pairs documented in gait EMG
#     literature: iliopsoas~rectus femoris (swing-phase hip flexion),
#     gluteus maximus~hamstrings (stance-phase hip extension), peroneals~soleus
#     (plantarflexion), extensor digitorum/peroneus tertius~tibialis anterior
#     (dorsiflexion), adductor group~gracilis (hip adduction), obliques~rectus
#     abdominis (abdominal wall).
# Deliberately left OUT of scope (no defensible single donor): the deep hip
# rotators (quad_fem, gem, piri), gluteus minimus compartments, tfl, sartorius,
# pectineus, and the remaining trunk muscles (erector spinae, quadratus
# lumborum) -- forcing these into the tight constraint via a fabricated pairing
# would reintroduce the exact "constraining actuators with no real content"
# problem the scope-isolation control (RUNS_INDEX.md) already showed hurts.
H2190_SCOPE_PROXY = {
    "vas_int_r": "vas_med_r",
    "gas_lat_r": "gas_med_r",
    "semimem_r": "semiten_r",
    "bifemsh_r": "bifemlh_r",
    "glut_med2_r": "glut_med1_r",
    "glut_med3_r": "glut_med1_r",
    "iliacus_r": "rect_fem_r",
    "psoas_r": "rect_fem_r",
    "glut_max1_r": "semiten_r",
    "glut_max2_r": "semiten_r",
    "glut_max3_r": "semiten_r",
    "tib_post_r": "soleus_r",
    "per_brev_r": "soleus_r",
    "per_long_r": "soleus_r",
    "per_tert_r": "tib_ant_r",
    "ext_dig_r": "tib_ant_r",
    "int_obl_r": "ext_obl_r",
    "add_long_r": "grac_r",
    "add_brev_r": "grac_r",
    "add_mag1_r": "grac_r",
}
# 11 directly-mapped + 20 proxy-extended = 31/45 muscles/leg in scope (68.9%),
# matching h0918 (67%) and h1622 (64%) -- the number this design deliberately
# targets, not a coincidence.

MODEL_MAPS = {
    "h0918": H0918_MAP,
    "h1622": H1622_MAP,
    "h2190": H2190_MAP,
}


class EMGToMuscleMapper:
    """Maps 11-channel EMG onto chosen model's muscle actuators.

    Builds a (n_actuators, 11) weight matrix from the exact-match table for `model_name`: 
    - each row is 0 (no match) 
    - one-hot (single substitute) or
    - an even split across the channels that are literal constituents of that muscle (ex. vasti_r gets 0.5/0.5 on vastusmed/vastuslat).
    Mapping is then just `weights @ emg`.
    """

    def __init__(
        self, model_name: str, actuator_names: list[str],
        mirror_left: bool = False, null_prior: bool = False,
        proxy_map: Optional[dict[str, str]] = None,
    ):
        channel_to_muscle = MODEL_MAPS[model_name]
        self.actuator_names = actuator_names
        self.weights = np.zeros((len(actuator_names), len(EMG_CHANNELS)))

        # null_prior: dimensionality-matched control for isolating whether an
        # underperforming latent-prior run is caused by the EMG signal itself
        # or just by the larger (dim_latent + n_actuators) action space vs.
        # plain MPO's (n_actuators)-only space. Leaving self.weights at its
        # zero-initialized default (skipping the mapping logic entirely,
        # rather than computing real weights and zeroing them after) means
        # a_hat is provably 0 for every actuator every step -- no code path
        # here can leave a stray nonzero entry.
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

        # proxy_map (e.g. H2190_SCOPE_PROXY): extends real content onto
        # anatomically-related actuators that have no EMG channel of their
        # own, by copying their donor's already-computed row verbatim. Every
        # donor must be a directly-mapped muscle (never another proxy), so
        # this is a single pass with no ordering dependency -- copying from
        # self.weights (not muscle_to_channels) means it's agnostic to
        # whether the donor's row came from the loop above.
        if proxy_map:
            for muscle, donor in proxy_map.items():
                if muscle not in actuator_names or donor not in actuator_names:
                    continue
                self.weights[actuator_names.index(muscle)] = self.weights[actuator_names.index(donor)]

        # Mirroring gives each populated "_r" actuator's "_l" counterpart the
        # exact same weight row -- real EMG-mapped or proxy-extended alike --
        # so a_hat_l == a_hat_r every step (no gait-phase offset -- real
        # contralateral limbs are ~50% out of phase, this is the naive
        # synchronous version to test first). Scans populated rows directly
        # rather than muscle_to_channels so proxy rows get mirrored too.
        if mirror_left:
            for i, muscle in enumerate(actuator_names):
                if not muscle.endswith("_r") or not self.weights[i].any():
                    continue
                left_muscle = muscle[:-2] + "_l"
                if left_muscle not in actuator_names:
                    continue
                self.weights[actuator_names.index(left_muscle)] = self.weights[i]

    def map_emg_to_muscles(self, emg: np.ndarray) -> np.ndarray:
        """Map an (11,) EMG vector to a (n_actuators,) activation vector.

        Args:
            emg: Shape (11,) EMG activations, values in [0, ~1].

        Returns:
            Shape (n_actuators,) muscle commands. Unmapped actuators are 0.
        """
        return self.weights @ emg

    def map_batch(self, emg_batch: np.ndarray) -> np.ndarray:
        """Map a batch of EMG vectors.

        Args:
            emg_batch: Shape (N, 11).

        Returns:
            Shape (N, n_actuators). Unmapped actuators are 0.
        """
        return emg_batch @ self.weights.T

    # def to_loco_action(self, muscle_act: np.ndarray) -> np.ndarray:
    #     """Convert muscle activations [0, 1] to simulator action range [-1, 1].

    #     locomujoco expects actions in [-1, 1] and map to ctrl in [0, 1]
    #     via: ctrl = 0.5*(action + 1). Muscle activations from EMG are in [0, 1].
    #     This function inverts DefaultControl's mapping so that EMG activations
    #     produce the correct ctrl values when fed through DefaultControl.

    #         ctrl = 0.5*(action+1)  ->  action = 2*ctrl - 1

    #     Args:
    #         muscle_act: Shape (..., N), muscle activations in [0, 1].

    #     Returns:
    #         Same shape, actions in [-1, 1] for use with env.step().
    #     """
    #     return 2.0 * muscle_act - 1.0
