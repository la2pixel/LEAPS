"""EMG rectification (from Camargo's rectify.m) 
refer to leaps.data.metadata
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, firwin


def rectify_emg(emg: np.ndarray, fs: int = 1000) -> np.ndarray:
    """Apply rectify.m filter chain: highpass -> bandpass FIR -> abs -> lowpass -> abs."""
    x = emg.astype(np.float64)
    #step 1: highpass 10 Hz, butterworth order 4 (EMG_HIGHPASS)
    b, a = butter(4, 10.0, btype="high", fs=fs)
    x = filtfilt(b, a, x, axis=0)
    #step 2: bandpass FIR 10-450 Hz, order 20 (EMG_BANDPASS)
    b_fir = firwin(21, [10.0, 450.0], pass_zero=False, fs=fs)
    x = filtfilt(b_fir, [1.0], x, axis=0)
    #step 3: rectify full wave
    x = np.abs(x)
    #step 4: lowpass envelope 6 Hz, butterworth order 4 (EMG_ENVELOPE_LOWPASS)
    b, a = butter(4, 6.0, btype="low", fs=fs)
    x = filtfilt(b, a, x, axis=0)
    return np.abs(x)
