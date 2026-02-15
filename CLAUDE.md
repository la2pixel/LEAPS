# LEAPS — Project Guide for Claude Code

## Project Overview

LEAPS (Learning Humanoid Locomotion from EMG-Based Latent Action Priors and Muscle Synergies) is a research project that learns muscle-actuated humanoid locomotion by extracting low-dimensional action representations from human EMG data. Two core techniques:

- **Latent Action Priors**: Autoencoder-based latent space representation of actions
- **Muscle Synergies**: Non-Negative Matrix Factorization (NMF) for identifying muscle activation groupings

Early-stage research project (potential ICLR submission). Author: Lalitha Sivakumar.

## Tech Stack

- **Python** >=3.9 (target 3.9–3.11)
- **ML/DL**: PyTorch (primary), JAX/Flax/Brax (secondary for RL environments)
- **Robotics/RL**: MuJoCo, Gymnasium, Stable-Baselines3, LocoMuJoCo
- **Data**: NumPy, SciPy, pandas, h5py, scikit-learn
- **Config**: Hydra + OmegaConf, PyYAML
- **Logging**: TensorBoard, Weights & Biases
- **Visualization**: Matplotlib, imageio (with FFmpeg)
- **Build**: setuptools via pyproject.toml (PEP 517/518)

## Project Structure

```
src/leaps/
├── __init__.py          # Public API: EMGPreprocessor, load_camargo_dataset, LatentActionPrior, SynergyExtractor
├── data/                # EMG data preprocessing and dataset loading (Camargo dataset)
├── models/              # Core ML models (LatentActionPrior, SynergyExtractor)
├── training/            # Training loops and utilities
├── evaluation/          # Evaluation metrics and analysis
├── envs/                # Gymnasium environment wrappers (MuJoCo-based)
├── controllers/         # Control policies for humanoid locomotion
├── scripts/             # CLI entry points (train, eval, preprocess, render)
└── utils/               # Shared helper utilities
```

Data directories (gitignored): `data/raw/`, `data/processed/`, `experiments/`, `checkpoints/`, `logs/`, `wandb/`, `results/`, `renders/`

## Data & Environment

- **Raw data**: Camargo dataset, located on the institute's cluster `/fast` drive (not on local machine)
- **Local repo**: Cloned to `C:\Users\lsivakumar\Documents\LEAPS` (Windows, local development)
- **Reference scripts**: MATLAB `.m` files from the Camargo study are in a `scripts/` folder on the cluster — these are the reference for how to correctly parse/process the raw data
- Data processing scripts in this repo (Python) should replicate the logic from those `.m` files

## Current Focus

The immediate priority is the **data pipeline**:
1. Parse the raw Camargo dataset correctly (guided by the reference `.m` scripts)
2. Write Python preprocessing scripts that faithfully replicate the MATLAB processing
3. Validate that the Python output matches the MATLAB reference

## Commands

### Installation
```bash
pip install -e .            # Standard install
pip install -e ".[dev]"     # With dev tools (pytest, black, isort)
pip install -e ".[all]"     # All optional dependencies
```

### CLI Entry Points
```bash
leaps-train                 # Training (leaps.scripts.train:main)
leaps-eval                  # Evaluation (leaps.scripts.evaluate:main)
leaps-preprocess            # Data preprocessing (leaps.scripts.preprocess_data:main)
leaps-render                # Rendering (leaps.scripts.render:main)
```

### Code Quality
```bash
black src/ tests/ --line-length 100        # Format code
isort src/ tests/ --profile black          # Sort imports
pytest                                      # Run tests with coverage
pytest tests/test_specific.py -k "name"    # Run a single test
```

## Code Style & Conventions

- **Formatter**: black, line length 100
- **Import sorting**: isort with black profile, line length 100
- **Naming**: snake_case for modules/functions, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Imports**: Use absolute imports within the package (e.g., `from leaps.data.preprocessor import EMGPreprocessor`)
- **Public API**: Define `__all__` in every `__init__.py` to control exports
- **Docstrings**: Use triple-quoted docstrings; include module-level docstrings in `__init__.py`
- **Type hints**: Use Python 3.9+ syntax (e.g., `list[int]` not `List[int]`)
- **Tests**: Place in `tests/` directory, pytest with `--cov=src/leaps`

## Architecture Notes

- Models follow a scikit-learn-like interface: `fit()`, `transform()`, `predict()`
- Environments are Gymnasium-compatible wrappers around MuJoCo humanoid models
- Experiment configuration is managed via Hydra/OmegaConf YAML configs
- The project uses LocoMuJoCo for humanoid locomotion environments
- EMG data pipeline: raw EMG → preprocessing (filtering, normalization, gait segmentation) → synergy extraction / latent encoding → policy training
- Synergy extraction uses scikit-learn's NMF; latent action priors use PyTorch autoencoders

## Key Domain Concepts

- **EMG (Electromyography)**: Electrical signals from muscles during movement
- **Muscle Synergies**: Low-dimensional representations of coordinated muscle activations, extracted via NMF
- **Latent Action Priors**: Learned compressed action spaces that capture natural movement patterns
- **Camargo Dataset**: Public human locomotion EMG dataset used for training
- **Gait Cycle**: One complete walking stride, segmented by heel strikes
- **MuJoCo**: Physics simulator for articulated body dynamics

## Important Patterns

- Keep ML model code (in `models/`) separate from training logic (in `training/`)
- Environment wrappers should be thin — delegate physics to MuJoCo/LocoMuJoCo
- Data loading functions should support both raw file paths and preprocessed HDF5 files
- Use TensorBoard for local experiment tracking, W&B for remote/shared tracking
- Checkpoint saving/loading should use PyTorch's `state_dict` pattern

## Cluster Environment (MPI IS Tübingen)

### Access
- **Login**: `ssh lsivakumar@login.cluster.is.localnet` (campus network / registered WiFi required)
- **SSH alias**: `ssh cluster` (configured in `~/.ssh/config`)
- **Cluster home**: auto-created at first login, separate from regular `$HOME`
- **File access from workstation**: `sftp://lsivakumar@login.cluster.is.localnet`
- **Support**: IT service desk (Cluster related) or zwe-sc@tuebingen.mpg.de

### Filesystem Layout
- **`/fast/lsivakumar/`** — primary workspace (data + code + experiments). Fast I/O, NOT backed up
- **Cluster home** (`~`) — small, use only for dotfiles/config. Has quota
- **`/tmp`** — 1GB quota on login nodes. Set `TMPDIR=~/tmp` for large pip installs

### Project Paths on Cluster
```
/fast/lsivakumar/
├── leaps/                  # Git repo clone
├── data/camargo/           # Raw Camargo dataset (adjust path as needed)
└── ...
```

### Job Scheduler: HTCondor
- **All computation must go through HTCondor** — login nodes are for light tasks only
- Default allocation: 1 CPU core, 2 GB RAM
- CPU limits are hard-enforced; RAM is soft (guaranteed amount, may swap under contention)
- Jobs auto-removed after 1 month

**Submit a job:**
```bash
condor_submit_bid 10 job.sub
```

**Interactive session (for debugging/development):**
```bash
condor_submit_bid 10 -i
condor_submit_bid 10 -i -append request_cpus=2 -append request_memory=4096
condor_submit_bid 10 -i -append request_gpus=1 -append request_memory=8192
```

**Monitor / manage jobs:**
```bash
condor_q lsivakumar          # Check your jobs (I=idle, R=running, H=held)
condor_q -run lsivakumar     # Running jobs only
condor_rm <job_id>            # Remove a job
condor_ssh_to_job <job_id>    # SSH into a running job (light inspection only)
```

### GPU & Deep Learning Setup
```bash
# In submit file or -append:
request_gpus = 1

# Inside the job, load CUDA + cuDNN (required for PyTorch GPU):
source /etc/profile.d/modules.sh   # if module cmd not available
module load cuda                    # default version
module load cudnn                   # required for PyTorch
module avail cuda                   # list available CUDA versions
module avail cudnn                  # list available cuDNN versions
```
- PyTorch GPU requires both `cuda` and `cudnn` modules loaded
- If compiling PyTorch from source with CUDA, must use an interactive session WITH a GPU
- GPU devices are isolated per job (appear as id=0,1,... regardless of physical card)
- Some GPU nodes require minimum bids
- Check actual available versions with `module avail` once on the cluster (docs reference old versions)

### Networking
- **Compute nodes have NO direct internet** — only HTTP/HTTPS via proxy
- `http_proxy` and `https_proxy` env vars are set automatically
- No `ping`, no raw sockets, no non-HTTP protocols on compute nodes

### Jupyter on Cluster
```bash
# 1. On cluster: get an interactive session
condor_submit_bid 10 -i
# Note which node you land on (e.g. e018)

# 2. On that node: start Jupyter
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# 3. On local machine: SSH tunnel (replace <node> with actual node name)
ssh -N -L 8888:<node>:8888 cluster

# 4. Open http://localhost:8888/ in local browser
```
If port 8888 is taken, pick another (1024–65535) and use that in both commands.

### Containers (Apptainer, formerly Singularity)
The cluster uses Apptainer (NOT Docker — Docker is deprecated). Containers give reproducible
environments and avoid module-loading issues. Build containers on a compute node, not login nodes.

**Building a container (one-time):**
```bash
# 1. Get an interactive session with enough resources
condor_submit_bid 25 -i -append 'request_memory=81920' -append 'request_cpus=10' \
  -append 'request_disk=100G' -append 'request_gpus=1'

# 2. Create sandbox from a Docker base image
apptainer build --sandbox /tmp/leaps_sandbox docker://continuumio/miniconda3
mkdir /tmp/home

# 3. Enter sandbox (isolated from shared filesystems)
apptainer shell --writable --nv -f -c --pwd=/tmp/home -H /tmp/home /tmp/leaps_sandbox/

# 4. Inside sandbox: install dependencies
#    If /tmp write errors: chmod 1777 /tmp
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install gymnasium mujoco stable-baselines3 scipy h5py pandas scikit-learn
mkdir /code && cp -r /fast/lsivakumar/leaps /code/leaps  # or pip install from source
exit

# 5. Test (re-enter with shared filesystems visible, but no persistent writes)
apptainer shell --writable-tmpfs --nv --pwd=/tmp/home -H /tmp/home /tmp/leaps_sandbox/
python -c "import torch; print(torch.cuda.is_available())"
exit

# 6. Freeze sandbox into a portable .sif image
apptainer build /fast/lsivakumar/images/leaps.sif /tmp/leaps_sandbox/
exit  # leave interactive session
```

**Key flags:**
- `--nv` — GPU passthrough
- `--writable-tmpfs` — temp writes allowed, not persisted to image
- `--pwd=/tmp/home -H /tmp/home` — isolate home dir to avoid polluting shared FS
- `-f -c` — use only during build (isolates from shared FS entirely)
- Shared folders `/fast`, `/is`, `/ps` are auto-mounted at runtime

**HTCondor with Apptainer** (see `cluster/job_container.sub`):
```
executable = /usr/bin/singularity
arguments = "exec --writable-tmpfs --nv --pwd=/tmp/home -H /tmp/home leaps.sif python /code/leaps/train.py"
should_transfer_files = yes
transfer_input_files = /fast/lsivakumar/images/leaps.sif
request_gpus = 1
```

**NVidia NGC images** (optimized, requires free account + API key at ngc.nvidia.com):
```bash
SINGULARITY_DOCKER_USERNAME='$oauthtoken' \
SINGULARITY_DOCKER_PASSWORD="<API_KEY>" \
apptainer build image.sif docker://nvcr.io/nvidia/pytorch:<tag>
```

### TMPDIR Workaround
Login node `/tmp` has 1GB quota. For pip installs:
```bash
mkdir -p ~/tmp
export TMPDIR=~/tmp
pip install -e ".[dev]"
```
The cluster uses its own storage system, which is configured and heavily optimized for the special workload of cluster jobs. Nevertheless, all department and group shares, like /is, /ei or /ps, are also available on all cluster nodes on the same path as on your workstation.

Department shares are normally not configured for cluster use. Therefore some shares may be restricted on the cluster. If you have questions about accessing these shares on the cluster, please contact your IT administrator.
This storage can be accessed from outside the cluster, but with a lower performance. Furthermore, it is divided in several areas with different aims and policies.

The access to the cluster shares from outside the cluster is done through SMBv3. This has some implications, most notably when dealing with symbolic links. In short, symbolic links won't be visible as such from outside the cluster. Under some conditions they will still be partially usable, though, with the following limitations:

Symlinks work as usual from within the cluster
Symlinks can only be created from within the cluster
Symlinks pointing to the same area will appear as different files, although internally they will still be a single file. This means that they will be copied as separate, independent files, but that modifications (including creating files in the case of directories) will affect both
Symlinks pointing to other areas will not be visible, nor usable (trying to create files with the same name will fail)
In some places there are quotas set (primarily to avoid backing up huge volumes of unnecessary data and also to stop programs from filling up the storage space unintentionally).

Please also note that the quotas are set as default, but can can be adapted to individual needs. If you have a justified need for more space, please don't hesitate and let us know, so we can adapt them to your situation.

Storage Areas
There are several storage areas connected to the cluster. Each area has its own purpose. This table shows all available storage areas in a quick overview. Please see the following section for a detailed description.

Home	/home/<username>	/is/cluster/<username>	configuration, code, post-processed results, multiple small files	Yes	Yes, almost daily	SSD
Fast	/fast/<username>	/is/cluster/fast/<username>	working data, big files, small files	Yes	No	SSD
To access the external path, the SMB protocol is used. It relies on kerberos. To access the path, you need a valid kerberos ticket. By default the kerberos ticket is handled by the operation system, when you are log in locally. If you are accessing the shares through a remote ssh session, you have to verify your kerberos ticket. Please refer to the Kerberos Guide for further information.

Data retention on cluster filesystems

All cluster filesystems are meant to be used only for the computational purpose and not as a group or collaboration share. The /home directory is backed up (almost) daily to prevent data loss in case of a filesystem failure. The /fast directory designed to handle the workload of the cluster and provide enough capacity for the working data for all users. Due to its size and design, it is not possible to back it up in a reasonable amount of time. Therefore you should copy all important data as fast as possible to a reasonable group share. In general you should NOT store data on the cluster filesystems, which you cannot reproduce somehow.
