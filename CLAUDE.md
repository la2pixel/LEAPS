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

Data pipeline is complete. Current focus is **representation learning + RL integration**.

### Existing Models (implemented)

**Snapshot-level** (`models/pytorch_models.py`, trained via `scripts/train.py`):
- AE: 11 → 2×latent → latent → 2×latent → 11 (Tanh activations)
- VAE: Same + reparameterization, β=0.1 KL loss
- MAE: AE with 50% random channel masking
- Loss: MSE + lnorm_weight × Lnorm(z) (soft penalty outside [-1.2, 1.2])
- **Known issue**: No sigmoid on decoder — outputs can exceed [0,1]

**Stride-level** (`models/stride_models.py`, trained via `scripts/train_strides.py`):
- Conv1D encoder: Conv(11→32, k=7) + ReLU → Conv(32→64, k=5) + ReLU → FC → latent
- Conv1D decoder: FC → ConvTranspose layers → sigmoid output [0,1]
- StrideVAE, StrideMAE variants (block masking: 15-point contiguous)

**Sklearn** (`models/sklearn_models.py`): PCA, NMF baselines

### AE Variant Strategies (to try all, pick winner empirically)

Priority order based on theoretical fit for EMG latent action priors:

1. **WAE-MMD** (Wasserstein AE with Maximum Mean Discrepancy)
   - Replace KL divergence with MMD penalty against N(0,I)
   - Optimal transport ≈ Hausdorff distance preservation at distribution level
   - Sharper reconstructions than VAE, smooth latent space for RL
   - ~20 lines change from existing AE
   - Ref: Tolstikhin et al., 2018

2. **Beta-VAE** (beta=0.01–0.1) — already have VAE, just tune beta lower
   - Safe baseline, smooth latent, risk of blurring gait transitions at high beta

3. **AE + soft-DTW reconstruction loss** (differentiable Dynamic Time Warping)
   - Penalizes temporal shape mismatches, not just pointwise MSE
   - Preserves peak timing and gait profile shape directly
   - Packages: `tslearn` or `soft-dtw-cuda`
   - Ref: Cuturi & Blondel, 2017

4. **WAE-MMD + soft-DTW** — combine #1 and #3 for best distance + shape preservation

5. **VQ-VAE** (Vector Quantized VAE)
   - Discrete codebook → learned gait phase prototypes
   - Great for interpretability, but discrete-continuous mismatch for RL
   - Use as analytical tool alongside main model

6. **Contractive AE** — Jacobian penalty for robustness to EMG noise
   - Good as regularizer on top of another variant

### Key Decisions
- **Prefer stride-level models** (Conv1D) over snapshot for RL — they preserve temporal structure
- **TCN backbone** is fastest for ~100 timestep gait cycles
- **Sigmoid decoder** required for all models (muscle activations must be [0,1])
- **Latent dim**: 4–8 (9 muscles, want compression but not too lossy)
- **Evaluation**: Use gait-aware metrics (peak_timing_error, gait_profile_correlation, correlation_matrix_distance, latent_smoothness) — not just MSE/R²

### Relevant Literature
- PLAS (Zhou et al., 2020, CoRL) — VAE latent action space for offline RL
- OPAL (Ajay et al., 2021, ICLR) — offline primitive discovery via VAE action priors
- WAE (Tolstikhin et al., 2018, ICLR) — optimal transport for AE latent spaces
- Soft-DTW (Cuturi & Blondel, 2017, ICML) — differentiable time series distance
- MyoSuite (Vittorio et al., 2022) — musculoskeletal simulation benchmark

### Reference Paper
Hausdörfer et al. (2024) "Latent Action Priors for Locomotion with DRL" (arXiv:2410.03246)
- Simple deterministic AE (1 hidden layer, tanh, MSE + Lnorm), latent_dim = action_dim / 2
- Policy outputs z → frozen decoder → action, blended with residual: a = (1-w)*decode(z) + w*a_res
- PPO with style reward (joint position matching) + task reward (velocity tracking)
- LEAPS twist: EMG data replaces RL expert demos as the AE training source

### Available Models (all registered, ready to train)

**Snapshot-level** (train.py): PCA, NMF, AE, VAE, MAE, WAE
**Stride-level** (train_strides.py): StridePCA, StrideNMF, StrideAE, StrideVAE, StrideMAE, StrideWAE

### Training Commands
```bash
# Stride-level full sweep (recommended, ~10-20min on GPU)
python -m leaps.scripts.train_strides \
    --data /fast/lsivakumar/data/processed/emg_activations.h5 \
    --wandb --wandb-project leaps \
    --output-dir experiments/stride_sweep \
    --epochs 150 --batch-size 128 --beta 0.05 \
    --latent-dims 4 6 8 \
    --models StridePCA StrideNMF StrideAE StrideVAE StrideMAE StrideWAE

# Snapshot-level full sweep (~5min on GPU)
python -m leaps.scripts.train \
    --data /fast/lsivakumar/data/processed/emg_activations.h5 \
    --wandb --wandb-project leaps \
    --output-dir experiments/sweep \
    --epochs 200 --batch-size 4096 --beta 0.05 \
    --latent-dims 4 6 \
    --models PCA NMF AE VAE MAE WAE

# Plot reconstructions after training
python -m leaps.scripts.plot_reconstructions \
    --data /fast/lsivakumar/data/processed/emg_activations.h5 \
    --checkpoint-dir experiments/stride_sweep/checkpoints \
    --output-dir experiments/stride_sweep/plots \
    --latent-dim 4
```

### Pipeline Flow
```
Raw EMG (11ch) → Rectify → BPF → Stride Segment (heel strike)
→ Time Normalize (101 pts) → Min-Max Normalize → Clip 99th %ile → [0,1]
→ Train Model (snapshot or stride) → Latent z
→ EMGToMuscleMapper (11→18 muscles, bilateral) → MuJoCo muscle commands
→ MuscleHumanoidEnv (gait10dof18musc) → RL training
```

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

## LocoMuJoCo Integration

### Task IDs
- **Target**: `HumanoidMuscle.walk` (muscle-actuated humanoid walking)
- Task ID format: `<environment>.<task>.<dataset_type>` (e.g., `HumanoidMuscle.walk.real`)
- Defaults: missing task → "walk", missing dataset_type → "real"
- Multi-age variant: `HumanoidMuscle4Ages.walk.<1-4>.real` (1=smallest, 4=adult)
- List all: `LocoEnv.get_all_task_names()`
- **DEPRECATION**: `HumanoidMuscle` is a deprecated wrapper for `SkeletonMuscle`. Use `SkeletonMuscle` class directly for code, but task ID `HumanoidMuscle.walk` still works.

### Creating Environments
```python
# Direct LocoMuJoCo API
from loco_mujoco import LocoEnv
env = LocoEnv("HumanoidMuscle.walk.real")

# Gymnasium API
import loco_mujoco  # registers envs
import gymnasium as gym
env = gym.make("LocoMujoco", env_name="HumanoidMuscle.walk.real", render_mode="human")
```

### HumanoidMuscle / SkeletonMuscle Details
- **92 muscle actuators** on lower limbs + **14 torque motors** (upper body, prefixed `mot_`)
- Total action dim = 106 (92 muscles + 14 motors)
- Action space in LocoMuJoCo: **[-1, 1]**, auto-scaled to MuJoCo's [0, 1] for muscles
- Observation space: dim=36 (default, some obs disabled by default)
- Walking task: target speed **1.25 m/s**; Running: **2.5 m/s**
- Terminal state: robot falls (checked via orientation, CoM height, back joint)
- XML: `loco_mujoco/models/skeleton/skeleton_muscle.xml`

### 92 Lower-Limb Muscles (per leg, _r and _l suffixes)
```
glut_med1, glut_med2, glut_med3, glut_min1, glut_min2, glut_min3,
semimem, semiten, bifemlh, bifemsh, sar, add_long, add_brev,
add_mag1, add_mag2, add_mag3, tfl, pect, grac, glut_max1, glut_max2,
glut_max3, iliacus, psoas, quad_fem, gem, peri, rect_fem, vas_med,
vas_int, vas_lat, med_gas, lat_gas, soleus, tib_post, flex_dig,
flex_hal, tib_ant, per_brev, per_long, per_tert, ext_dig, ext_hal
```
Plus trunk: `ercspn_r/l, intobl_r/l, extobl_r/l`

### EMG → 92-Muscle Mapping (needs update from 18-muscle version)
Our EMG has 11 channels (right leg). The 92-muscle model has much finer granularity:
- gastrocmed → med_gas_r/l (+ lat_gas_r/l?)
- tibialisanterior → tib_ant_r/l
- soleus → soleus_r/l
- vastusmedialis → vas_med_r/l; vastuslateralis → vas_lat_r/l (+ vas_int?)
- rectusfemoris → rect_fem_r/l
- bicepsfemoris → bifemlh_r/l + bifemsh_r/l
- semitendinosus → semiten_r/l (+ semimem?)
- gluteusmedius → glut_med1/2/3_r/l
- gracilis → grac_r/l
- rightexternaloblique → extobl_r/l
- Unmapped muscles (no EMG): iliacus, psoas, add_long, add_brev, add_mag1/2/3, tfl, pect, glut_max1/2/3, glut_min1/2/3, sar, quad_fem, gem, peri, tib_post, flex_dig, flex_hal, per_brev, per_long, per_tert, ext_dig, ext_hal, ercspn, intobl, lat_gas
- Upper body motors (14): set to 0 or default

### Dataset Types
- **real**: Motion capture data (no actions, may have mismatches like floating feet). Available for all envs.
- **perfect**: Generated by best-performing baseline policy (has actions, no mismatches). Only some envs.
- **preference**: Ranked trajectory sets from suboptimal policies (has actions). Only some envs.

### Key APIs
```python
env.create_dataset()                              # Get expert dataset for imitation learning
env.play_trajectory(n_steps_per_episode=500)      # Replay dataset positions (no dynamics)
env.play_trajectory_from_velocity(n_episodes=30)  # Replay from velocities (verify consistency)
env.action_space.shape[0]                         # Action dimensionality
```

### Important Notes
- Datasets for imitation learning; rewards only used for evaluation (not training)
- Each env has a default reward function, but custom rewards can be provided
- Domain randomization available during training
- Arms NOT included in observation space by default (perfect/preference datasets only for default settings)
- LocoMuJoCo installed editable at `/home/lsivakumar/GitHub/Thesis/loco-mujoco/`

### Custom Env for gait10dof18musc — Implementation Details

**Decision**: Subclass `LocoEnv` directly, NOT `BaseSkeleton`.
- `BaseSkeleton` assumes full 92-muscle skeleton with arms, box feet options, subtalar/mtp joints, hip_adduction/rotation, 3-DOF lumbar, etc.
- Our model is a **2D sagittal-plane** humanoid: no free joint, no arms, 10 DOFs, 18 muscles
- Root is 3 separate joints (pelvis_tx slide, pelvis_ty slide, pelvis_tilt hinge) — NOT a `freejoint`

**Class hierarchy**: `Mujoco → Mjx → LocoEnv → Gait10dof18Musc` (in `src/leaps/envs/gait10dof_env.py`)

**Key LocoMuJoCo base class files** (already explored, do NOT re-explore):
- `loco_mujoco/core/mujoco_base.py` — `Mujoco` base: `load_mujoco()`, `step()`, `reset()`, `registered_envs`, `.register()`
- `loco_mujoco/environments/base.py` — `LocoEnv`: trajectory support, `__init__(n_substeps, timestep, spec, actuation_spec, observation_spec, **core_params)`
- `loco_mujoco/environments/humanoids/base_skeleton.py` — `BaseSkeleton`: reference for obs/action spec patterns, `@info_property` usage
- `loco_mujoco/core/observations/` — `ObservationType.JointPos()`, `ObservationType.JointVel()`, `ObservationType.FreeJointPosNoXY()`, etc.
- `loco_mujoco/core/control_functions/default.py` — `DefaultControl._unnormalize_action()`: maps [-1,1] → actuator ctrlrange. For muscles [0,1]: `ctrl = 0.5*(action+1)`
- `loco_mujoco/core/terminal_state/` — `TerminalStateHandler` base class with `.register()` and `.registered` dict
- `loco_mujoco/task_factories/rl_factory.py` — `RLFactory.make(env_name, terminal_state_type, goal_type, reward_type, **kwargs)`
- `loco_mujoco/core/utils/` — `info_property` decorator, `mj_jntname2qposid()`

**Constructor pattern** (from BaseSkeleton, adapt for our case):
```python
class Gait10dof18Musc(LocoEnv):
    def __init__(self, spec=None, observation_spec=None, actuation_spec=None, **kwargs):
        if spec is None:
            spec = self.get_default_xml_file_path()
        spec = mujoco.MjSpec.from_file(spec) if isinstance(spec, str) else spec
        if observation_spec is None:
            observation_spec = self._get_observation_specification(spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)
        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)
```

**Observation spec** (18 dims total):
- Joint positions (8): pelvis_tilt, hip_flexion_r, knee_angle_r, ankle_angle_r, hip_flexion_l, knee_angle_l, ankle_angle_l, lumbar_extension
  - pelvis_tx excluded (translation invariance), pelvis_ty included via velocity
  - Constraint joints excluded (knee_r_translation1/2, muscle wrapping points)
- Joint velocities (10): all 8 above + pelvis_tx, pelvis_ty
- Use `ObservationType.JointPos("q_X", xml_name="X")` and `ObservationType.JointVel("dq_X", xml_name="X")`
- Do NOT use `FreeJointPosNoXY` — no free joint exists

**Action spec** (18 muscles, exact XML order):
```python
["hamstrings_r", "bifemsh_r", "glut_max_r", "iliopsoas_r", "rect_fem_r", "vasti_r",
 "gastroc_r", "soleus_r", "tib_ant_r",
 "hamstrings_l", "bifemsh_l", "glut_max_l", "iliopsoas_l", "rect_fem_l", "vasti_l",
 "gastroc_l", "soleus_l", "tib_ant_l"]
```

**Critical overrides needed** (no free joint in 2D model):
- `free_jnt_qpos_id` property → return `np.zeros((0, 7), dtype=int)` (empty, no free joint)
- `free_jnt_qvel_id` property → return `np.zeros((0, 6), dtype=int)` (empty)
- These prevent crashes in base class code that concatenates free joint arrays

**@info_property values**:
- `root_body_name` → `"pelvis"`
- `upper_body_xml_name` → `"torso"`
- `root_height_healthy_range` → `(0.5, 1.3)` (pelvis_ty default ~0.95)

**Custom terminal state handler** (`src/leaps/envs/terminal_state.py`):
- `HeightJointTerminalStateHandler(TerminalStateHandler)` — checks `pelvis_ty` qpos against healthy range
- Default `HeightBasedTerminalStateHandler` assumes free joint → crashes on our model
- Must call `.register()` to add to LocoMuJoCo's handler registry

**generate() classmethod**:
```python
@classmethod
def generate(cls, task=None, **kwargs):
    kwargs.setdefault("terminal_state_type", "HeightJointTerminalStateHandler")
    kwargs.setdefault("goal_type", "NoGoal")
    kwargs.setdefault("reward_type", "NoReward")
    return cls(**kwargs)
```

**Action normalization for EMG data**:
- LocoMuJoCo expects actions in [-1, 1], auto-maps to [0, 1] for muscles
- EMG mapper outputs [0, 1] → must convert: `loco_action = 2 * emg_action - 1`

**Registration** (in `src/leaps/envs/__init__.py`):
```python
from leaps.envs.gait10dof_env import Gait10dof18Musc
from leaps.envs.terminal_state import HeightJointTerminalStateHandler
HeightJointTerminalStateHandler.register()
Gait10dof18Musc.register()
```

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
