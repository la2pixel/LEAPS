"""Camargo et al. (2021) dataset metadata and experiment constants.

Reference: Camargo et al., "A comprehensive, open-source dataset of lower limb
biomechanics in multiple conditions of stairs, ramps, and level-ground
ambulation and transitions." Journal of Biomechanics, 2021.

Contains per-subject demographics, experiment protocol parameters, sensor
specifications, and signal processing constants.
"""

from dataclasses import dataclass

# ── Dataset root path (cluster) ──────────────────────────────────────

CAMARGO_DATA_ROOT: str = "/fast/lsivakumar/datasets/camargo"

# ── Subject demographics ──────────────────────────────────────────────

@dataclass(frozen=True)
class SubjectInfo:
    age: int
    gender: str  # "M" or "F"
    height: float  # meters
    mass: float  # kg


SUBJECTS: dict[str, SubjectInfo] = {
    "AB06": SubjectInfo(age=20, gender="M", height=1.80, mass=74.8),
    "AB07": SubjectInfo(age=20, gender="M", height=1.65, mass=55.3),
    "AB08": SubjectInfo(age=21, gender="M", height=1.74, mass=72.6),
    "AB09": SubjectInfo(age=21, gender="F", height=1.63, mass=63.5),
    "AB10": SubjectInfo(age=22, gender="M", height=1.75, mass=83.9),
    "AB11": SubjectInfo(age=21, gender="M", height=1.75, mass=77.1),
    "AB12": SubjectInfo(age=24, gender="M", height=1.74, mass=86.2),
    "AB13": SubjectInfo(age=19, gender="M", height=1.73, mass=59.0),
    "AB14": SubjectInfo(age=22, gender="F", height=1.52, mass=58.4),
    "AB15": SubjectInfo(age=21, gender="M", height=1.78, mass=96.2),
    "AB16": SubjectInfo(age=20, gender="F", height=1.65, mass=55.8),
    "AB17": SubjectInfo(age=19, gender="M", height=1.68, mass=61.2),
    "AB18": SubjectInfo(age=19, gender="F", height=1.80, mass=60.1),
    "AB19": SubjectInfo(age=19, gender="M", height=1.70, mass=68.0),
    "AB20": SubjectInfo(age=21, gender="F", height=1.71, mass=68.0),
    "AB21": SubjectInfo(age=20, gender="F", height=1.57, mass=58.1),
    "AB23": SubjectInfo(age=20, gender="M", height=1.80, mass=76.8),
    "AB24": SubjectInfo(age=21, gender="F", height=1.73, mass=72.6),
    "AB25": SubjectInfo(age=20, gender="F", height=1.63, mass=52.2),
    "AB27": SubjectInfo(age=21, gender="M", height=1.70, mass=68.0),
    "AB28": SubjectInfo(age=33, gender="F", height=1.69, mass=62.1),
    "AB30": SubjectInfo(age=31, gender="M", height=1.77, mass=77.0),
}

ALL_SUBJECT_IDS: list[str] = sorted(SUBJECTS.keys())
N_SUBJECTS: int = len(SUBJECTS)

# Group-level stats: age 21 ± 3.4 yr, height 1.70 ± 0.07 m, mass 68.3 ± 10.83 kg

# ── Locomotion modes ──────────────────────────────────────────────────

ALL_MODES: list[str] = ["treadmill", "levelground", "ramp", "stair"]

# Treadmill: 28 speeds, 0.5–1.85 m/s in 0.05 m/s increments
# 7 trials x 4 speeds per trial block, 30s steady-state per speed
TREADMILL_SPEEDS: list[float] = [round(0.5 + i * 0.05, 2) for i in range(28)]

# Level-ground: 3 self-selected speeds, 10 circuits each (30 total)
LEVELGROUND_SPEEDS: dict[str, float] = {
    "slow": 0.88,    # ± 0.19 m/s
    "normal": 1.17,  # ± 0.21 m/s
    "fast": 1.45,    # ± 0.27 m/s
}

# Stairs: 6-step staircase, 5 trials x 2 starting legs x 4 heights = 40 trials
STAIR_HEIGHTS_MM: list[int] = [102, 127, 152, 178]  # mm (4, 5, 6, 7 inches)

# Ramps: 5m long, 5 trials x 2 starting legs x 6 angles = 60 trials
RAMP_ANGLES_DEG: list[float] = [5.2, 7.8, 9.2, 11.0, 12.4, 18.0]

# ── Sensor specifications ─────────────────────────────────────────────

# 11 right-side lower-limb EMG channels (column names as they appear in .mat files)
EMG_CHANNELS: list[str] = [
    "gastrocmed",           # gastrocnemius medialis
    "tibialisanterior",     # tibialis anterior
    "soleus",               # soleus
    "vastusmedialis",       # vastus medialis
    "vastuslateralis",      # vastus lateralis
    "rectusfemoris",        # rectus femoris
    "bicepsfemoris",        # biceps femoris
    "semitendinosus",       # semitendinosus
    "gracilis",             # gracilis
    "gluteusmedius",        # gluteus medius
    "rightexternaloblique", # right external oblique
]
N_EMG_CHANNELS: int = len(EMG_CHANNELS)

# Human-readable names for display/plotting
EMG_CHANNEL_LABELS: dict[str, str] = {
    "gastrocmed": "Gastrocnemius Med.",
    "tibialisanterior": "Tibialis Anterior",
    "soleus": "Soleus",
    "vastusmedialis": "Vastus Medialis",
    "vastuslateralis": "Vastus Lateralis",
    "rectusfemoris": "Rectus Femoris",
    "bicepsfemoris": "Biceps Femoris",
    "semitendinosus": "Semitendinosus",
    "gracilis": "Gracilis",
    "gluteusmedius": "Gluteus Medius",
    "rightexternaloblique": "Ext. Oblique (R)",
}

# Goniometer channels (column names in .mat files)
GON_CHANNELS: list[str] = [
    "ankle_sagittal", "ankle_frontal",
    "knee_sagittal",
    "hip_sagittal", "hip_frontal",
]

# IMU channels (column names in .mat files)
# 4 placements x (3 Accel + 3 Gyro) = 24 channels
IMU_PLACEMENTS: list[str] = ["foot", "shank", "thigh", "trunk"]
IMU_AXES: list[str] = ["X", "Y", "Z"]
IMU_CHANNEL_TYPES: list[str] = ["Accel", "Gyro"]

# Gait cycle columns (gcLeft / gcRight)
GC_CHANNELS: list[str] = ["HeelStrike", "ToeOff"]

# Conditions: treadmill has "Speed" column; other modes have string-only label
# columns (e.g. "Label") that are NOT extracted by the numeric .mat parser.
# Label-based stride classification requires a separate string extraction step.
CONDITIONS_NUMERIC_CHANNELS: dict[str, list[str]] = {
    "treadmill": ["Speed"],
    "levelground": [],
    "ramp": [],
    "stair": [],
}

# Bilateral motion capture (84 numeric channels + Header)
N_MOCAP_MARKERS: int = 32

# Force plates per ambulation mode
N_FORCE_PLATES: int = 5

# ── Sample rates (Hz) ─────────────────────────────────────────────────

SAMPLE_RATES: dict[str, int] = {
    "emg": 1000,
    "gon": 1000,
    "fp": 1000,
    "conditions": 1000,
    "imu": 200,
    "markers": 200,
    "gcLeft": 200,
    "gcRight": 200,
    "id": 200,
    "ik": 200,
    "jp": 200,
}

# ── Signal processing parameters ─────────────────────────────────────
# These match the actual code in rectify.m / STRIDES.m, which differs
# slightly from what the paper describes.

@dataclass(frozen=True)
class FilterSpec:
    sample_rate: int    # Hz
    filter_type: str    # "highpass", "bandpass", or "lowpass"
    cutoff: tuple[float, ...] | float  # Hz (tuple for bandpass, scalar otherwise)
    order: int          # filter order
    method: str = "butter"  # "butter" (IIR) or "fir"


# EMG rectification chain from rectify.m:
#   1. highpass IIR 10 Hz (Butterworth order 4, applied with filtfilt)
#   2. bandpass FIR 10–450 Hz (order 20, applied with filtfilt)
#   3. abs (full-wave rectification)
#   4. lowpass IIR 6 Hz (Butterworth order 4, applied with filtfilt) → envelope
#   5. abs
EMG_HIGHPASS = FilterSpec(
    sample_rate=1000, filter_type="highpass", cutoff=10.0, order=4, method="butter",
)
EMG_BANDPASS = FilterSpec(
    sample_rate=1000, filter_type="bandpass", cutoff=(10.0, 450.0), order=20, method="fir",
)
EMG_ENVELOPE_LOWPASS = FilterSpec(
    sample_rate=1000, filter_type="lowpass", cutoff=6.0, order=4, method="butter",
)

# IMU and GON acquisition filters (from paper)
IMU_FILTER = FilterSpec(sample_rate=200, filter_type="lowpass", cutoff=100.0, order=6)
GON_FILTER = FilterSpec(sample_rate=1000, filter_type="lowpass", cutoff=20.0, order=4)

# ── EMG normalization ─────────────────────────────────────────────────

# EMG is normalized to the average rectified amplitude during treadmill walking
# at this reference speed, per subject. Chosen to reduce inter-subject variability
# (near normal walking speed for the age group).
EMG_NORMALIZATION_SPEED: float = 1.35  # m/s

# ── Gait cycle ────────────────────────────────────────────────────────

# Gait phase: 0–100% from heel strike to heel strike (right leg)
# Heel strike detected from MoCap as zero linear velocity of heel marker
# Phase computed via linear interpolation between heel strikes
GAIT_CYCLE_PERCENT: tuple[float, float] = (0.0, 100.0)
