"""Camargo dataset metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Server-local paths — override by setting environment variables:
#   CAMARGO_DATA   path to raw Camargo dataset root
#   LEAPS_EMG_H5   path to processed emg_activations_v2.h5
CAMARGO_DATA_ROOT: str = os.environ.get("CAMARGO_DATA", "")
LEAPS_H5_PATH: str = os.environ.get("LEAPS_EMG_H5", "")

# demographics

@dataclass(frozen=True)
class SubjectInfo:
    age: int
    gender: str  # "M" or "F"
    height: float  # metres
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

ALL_MODES: list[str] = ["treadmill", "levelground", "ramp", "stair"]

# 11 right-side lower-limb EMG channels (column names in the .mat files)
EMG_CHANNELS: list[str] = [
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

# sample rates for each signal (Hz)
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
