import h5py
import numpy as np
from collections import Counter

with h5py.File('/fast/lsivakumar/data/processed/emg_activations_v2.h5', 'r') as f:
    subj = 'AB09'
    modes = np.array(f[subj]['modes']).astype(str)
    conds = np.array(f[subj]['conditions']).astype(str)
    speeds = np.array(f[subj]['speeds'])

    for mode in np.unique(modes):
        mask = modes == mode
        c = Counter(conds[mask])
        print(f'{mode} ({mask.sum()} strides): {dict(sorted(c.items()))}')
        if mode == 'treadmill':
            sp = speeds[mask]
            print(f'  speed range: {sp[np.isfinite(sp)].min():.2f} - {sp[np.isfinite(sp)].max():.2f} m/s')
