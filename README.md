# LEAPS: Learning Humanoid Locomotion from EMG-Based Latent Action Priors and Muscle Synergies

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Research code for ICLR 2025 submission.

## Overview

LEAPS learns muscle-actuated humanoid locomotion by extracting low-dimensional action representations from human EMG data using:
- **Latent Action Priors**: Autoencoder-based latent space
- **Muscle Synergies**: NMF-based muscle groupings

## Installation
```bash
pip install -e .
```

## Quick Start
```python
from leaps.data import EMGPreprocessor
from leaps.models import SynergyExtractor, LatentActionPrior

# Preprocess EMG
preprocessor = EMGPreprocessor()
data = preprocessor.process_trial(emg, heel_strikes)

# Extract synergies
synergies = SynergyExtractor(n_synergies=5)
synergies.fit(data)

# Train latent prior
prior = LatentActionPrior(latent_dim=8)
prior.fit(data)
```

## Citation
```bibtex
@inproceedings{sivakumar2025leaps,
  title={Learning Humanoid Locomotion from EMG-Based Latent Action Priors and Muscle Synergies},
  author={Sivakumar, Lalitha and Badie, Nadine and Schmitt, Syn},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

## License

MIT License - see LICENSE file.
