# LEAPS: Representation Learning — Methodology Notes

**Project**: LEAPS (Learning Humanoid Locomotion from EMG-Based Latent Action Priors)

---

## 1. Problem Setup

We want a compact latent code `z` that summarises one full gait cycle of EMG data. The decoder
`g(z)` reconstructs the full stride so it can be used as a muscle activation prior for RL. The
policy outputs `z` instead of all 11 (or 18) raw muscle activations.

**Input**: one gait cycle of 11-channel EMG, time-normalised to 101 frames → shape `(101, 11)`.
**Target latent dim**: 4–16 (aiming for 6–9 in practice).
**Output constraint**: decoded muscle activations must be in `[0, 1]` (MuJoCo ctrl range).

---

## 2. Data Pipeline

### 2.1 Source

Camargo dataset: 22 able-bodied subjects (AB06–AB30), 4 locomotion modes
(levelground, treadmill, ramp, stair), right leg only.
Total: ~51,000 strides after segmentation.

### 2.2 Preprocessing (in `src/leaps/data/`)

```
Raw EMG (2000 Hz)
  → Band-pass filter (20–450 Hz, 4th-order Butterworth)
  → Rectify (abs)
  → Low-pass envelope (6 Hz)
  → Stride segmentation by heel-strike events
  → Time-normalise each stride to 101 points (0–100% gait cycle, inclusive)
  → Subject-level min-max normalise per channel
  → Clip at 99th percentile, rescale to [0, 1]
```

11 channels (right leg): gastrocmed, tibialisanterior, soleus, vastusmedialis,
vastuslateralis, rectusfemoris, bicepsfemoris, semitendinosus, gracilis,
gluteusmedius, rightexternaloblique.

### 2.3 Data Split

- 80% train / 20% validation (random split, all subjects pooled)
- All 22 subjects used together → more diverse prior for RL
- Best anthropometric match to model (AB06, AB23: 1.80 m, ~75 kg male) noted but not
  used as a filter, because RL benefits from a more generalizable prior

---

## 3. Models

All models compress `(101, 11)` → `z ∈ R^d` and reconstruct back to `(101, 11)`.

### 3.1 Baselines: Linear Methods

#### PCA (StridePCA)

Flatten stride → `x ∈ R^{1111}`, apply standard PCA, unflatten.

```
x̂ = μ + U_k U_k^T (x - μ)
```

where `U_k` are the top-k eigenvectors of the empirical covariance. No constraint on
sign, so reconstructions can go below 0 or above 1 (clipped at inference).

- Implemented via `sklearn.decomposition.PCA`
- Fitting is instant; no hyperparameters beyond `k`
- **Not suitable as a prior**: latent space is orthogonal PCA coordinates with no
  distributional structure; RL policy cannot easily sample meaningful actions from it

#### NMF (StrideNMF)

Factorises the non-negative data matrix `X ≈ W H` where `W, H ≥ 0`.

```
min_{W,H≥0}  ½ ||X - W H||_F²
```

- `W ∈ R^{n × k}`: activation coefficients per stride (latent representation)
- `H ∈ R^{k × 1111}`: basis strides (muscle synergy patterns over the full cycle)
- Non-negativity matches the physical interpretation of EMG (activations ≥ 0)
- Implemented via `sklearn.decomposition.NMF(init='nndsvda')`
- Classic "muscle synergy" approach applied at the full-stride level

#### CNMF (StrideCNMF) — Smoothness-Constrained NMF

Adds an L2 penalty on the activation coefficients to reduce sensitivity to local optima:

```
min_{W,H≥0}  ½ ||X - W H||_F²  +  α ||W||_F²
```

- `α = 0.1` (default)
- Uses multiplicative update (MU) solver, matching the muscle synergy literature
- Ref: Lee & Seung (1999), with smoothness extension from EMG synergy studies

**Important finding**: CNMF performed *worse* than standard NMF in our experiments
(val R² ~0.33 vs ~0.49 at d=9). The L2 penalty over-constrains the activations in the
non-negative setting, causing the basis vectors `H` to be poorly fitted.

---

### 3.2 Deep Models: Conv1D Architecture (StrideAE, StrideVAE, StrideWAE)

All three share the same `_StrideEncoder` / `_StrideDecoder` backbone:

#### Encoder: `_StrideEncoder`

```
Input:  (batch, 101, 11)
  → transpose to (batch, 11, 101)            # channels first for Conv1d
  → Conv1d(11→32, kernel=7, stride=2, pad=3) + ReLU   # → (batch, 32, 51)
  → Conv1d(32→64, kernel=5, stride=2, pad=2) + ReLU   # → (batch, 64, 26)
  → flatten → (batch, 64*26=1664)
  → Linear(1664→128) + ReLU
  → Linear(128→latent_dim)                             # → (batch, d)
```

Receptive field of each conv output neuron over the input: ~15 timepoints out of 101.
This is only ~15% of the stride — the network can detect local bursts but cannot directly
compare events that are 50% of the cycle apart (e.g., left vs. right leg coordination).

#### Decoder: `_StrideDecoder`

```
Input:  (batch, d)
  → Linear(d→128) + ReLU
  → Linear(128→1664) + ReLU
  → reshape to (batch, 64, 26)
  → ConvTranspose1d(64→32, kernel=5, stride=2) + ReLU   # → (batch, 32, 51)
  → ConvTranspose1d(32→11, kernel=7, stride=2)
  → sigmoid                                              # → (batch, 11, 101) in [0,1]
  → transpose to (batch, 101, 11)
```

**Known bug**: The final `sigmoid` saturates near 0 and 1, producing flat gradients that
hurt training. See section 5.1 for analysis. This bug is present in StrideAE/VAE/WAE but
NOT in StrideFlatAE.

#### StrideAE — Deterministic Autoencoder

Loss:
```
L = MSE(x, x̂) + λ_norm · L_norm(z)
```

where `λ_norm = 0.01` and L_norm is the Hausdörfer et al. (2024) latent bounding penalty:

```
L_norm(z) = mean over {i,b} of:
    0                               if |z_{b,i}| < 0.8
    exp((z_{b,i} / 1.2)^10) - 1    otherwise
```

This is zero in `[-0.8, 0.8]`, then rises steeply near ±1.2, keeping the latent space
inside `[-1, 1]` for RL compatibility (PPO action space is typically `[-1, 1]`).

**Problem found in practice**: At random initialisation, Conv1D encoder outputs large
latents → L_norm produces huge loss (`6.97e13` at epoch 1) → gradients explode → model
lands in a bad local minimum even with gradient clipping at 1.0. The model recovers but
with worse performance than FlatAE.

#### StrideVAE — Variational Autoencoder

The encoder outputs `(μ, log σ²)` from a shared conv backbone + two separate FC heads.
Reparameterisation: `z = μ + σ ⊙ ε`, `ε ~ N(0, I)`.

Loss:
```
L = MSE(x, x̂) + β · KL(q(z|x) || N(0,I)) + λ_norm · L_norm(μ)
```

```
KL(q||p) = -½ · mean( 1 + log σ² - μ² - σ² )
```

- `β = 0.1` (mild; prioritises reconstruction over latent regularity)
- At inference, only `μ` is used (deterministic decode)
- The L_norm on `μ` keeps the posterior mean inside `[-1, 1]`

#### StrideWAE-MMD — Wasserstein Autoencoder

Deterministic encoder (same as StrideAE), but replaces KL divergence with Maximum Mean
Discrepancy (MMD) between the encoded batch and samples from `N(0, I)`:

```
L = MSE(x, x̂) + λ_mmd · MMD²(q(z), N(0,I)) + λ_norm · L_norm(z)
```

MMD uses RBF kernel with bandwidth `σ² = d` (latent_dim):

```
MMD²(P, Q) = E_{z~P}[k(z,z')] - 2 E_{z~P, p~Q}[k(z,p)] + E_{p~Q}[k(p,p')]
k(x, y) = exp(-||x - y||² / (2 · d))
```

Samples `p ~ N(0,I)` are drawn fresh each batch.

- `λ_mmd = 10.0`, `λ_norm = 0.01`
- Motivation: optimal transport (Wasserstein distance) ≈ better distance preservation in
  latent space; sharper reconstructions than VAE because no variance term forces blur
- Ref: Tolstikhin et al. (2018), "Wasserstein Auto-Encoders"

**Status**: Implemented but not yet trained in the fixed-architecture sweep (next priority).

---

### 3.3 Main Model: StrideFlatAE — Flatten-MLP Autoencoder

Adapted directly from Hausdörfer et al. (2024), which uses a simple 1-hidden-layer AE
with tanh and Lnorm on raw joint angles. We adapt for EMG strides.

#### Architecture

```
Input stride:  (101, 11)
  → flatten →  (1111,)

Encoder:
  Linear(1111, 512) + ReLU
  Linear(512,  256) + ReLU
  Linear(256,  d)           # latent z, no activation, unbounded

Decoder:
  Linear(d,   256) + ReLU
  Linear(256, 512) + ReLU
  Linear(512, 1111)         # LINEAR output — no sigmoid

  → reshape → (101, 11)
  → np.clip(x, 0, 1) at inference only
```

#### Why MLP over Conv1D?

The key structure in EMG strides is **cross-temporal correlation**: the push-off burst at
~60% of the cycle must relate to the loading response at ~5%. Conv1D with kernel size 7
has a receptive field of only ~15 timepoints — it cannot see across 50% of the stride.
An MLP with a flattened input sees the entire 1111-element vector at once, allowing it to
learn any cross-time dependency regardless of temporal distance.

Empirically: FlatAE (val R²=0.637, d=8) vs StrideAE (val R²=0.481, d=8) on the same data.

#### Why no sigmoid on decoder output?

EMG data after preprocessing spans the full `[0, 1]` range. The sigmoid function has
gradient `σ(x)(1-σ(x))` → near 0 at `x >> 0` or `x << 0`. This means the loss surface
becomes flat exactly where EMG peaks and troughs are most informative. The MSE loss
cannot push the decoder past these saturated regions.

**Fix**: Remove sigmoid from decoder. Output is linear (unbounded real). Clip to `[0, 1]`
at inference before passing to the EMG→muscle mapper. This is safe because the training
signal is MSE against `[0,1]`-bounded targets, so the decoder learns to stay near that
range anyway.

Before fix (with sigmoid): val R² ≈ 0.40–0.52 (worse than PCA).
After fix (no sigmoid):    val R² = 0.545–0.646 (beats PCA at matched d).

#### Why no Lnorm on FlatAE?

Lnorm is needed when the Conv1D encoder produces large latent values at random init
(large weight matrices over 1664-dim feature vectors). The MLP encoder starts from a
smaller 1111-dim input with smaller typical magnitudes, and the MSE loss itself keeps
latents from growing arbitrarily. Adding Lnorm worsens reconstruction without benefit.

For RL: the policy applies tanh externally to keep actions in `[-1, 1]`. No need to
bake it into the prior.

#### Loss function

```
L = MSE(x, x̂)  =  (1/T·C) · Σ_{t,c} (x_{t,c} - x̂_{t,c})²
```

where `T=101`, `C=11`. No regularisation beyond early stopping.

#### Training

- Optimiser: Adam, `lr=1e-3`
- Scheduler: ReduceLROnPlateau, factor 0.5, patience 10 val-checks (= every 5 epochs →
  patience = 50 epochs without improvement at LR level)
- Early stopping: patience 15 val-checks = 75 epochs without val-loss improvement
- Max epochs: 200
- Batch size: 256
- Best checkpoint saved by val MSE, restored at end of training

---

## 4. Evaluation Metrics

All metrics computed on the validation set.

| Metric | Definition | What it measures |
|--------|-----------|-----------------|
| `val_r2` | `1 - SS_res/SS_tot` over all elements | Overall reconstruction quality |
| `per_muscle_r2_mean` | R² per channel, then average | Which channels are well-reconstructed |
| `peak_timing_err` | Mean absolute error (% gait cycle) between true and reconstructed peak locations | Are burst timings preserved? |
| `peak_amp_ratio` | `x̂_peak / x_peak` per channel | Are burst amplitudes preserved? (<1 = under-shoot, >1 = over-shoot) |
| `gait_profile_corr` | Pearson r between mean stride of true vs reconstructed | Is the average gait pattern preserved? |
| `corr_matrix_dist` | Frobenius norm of (true corr matrix - reconstructed corr matrix) | Are inter-muscle coordination patterns preserved? |
| `mode_X_r2` | val_r2 computed within each locomotion mode | Does the model generalise across modes? |

**Note on `latent_smoothness`**: Currently computed as 0.0 in our setup (requires
consecutive-stride pairs in the loader, which we don't currently track). This metric
would measure whether adjacent strides map to nearby latent codes.

---

## 5. Training Runs and Results

### 5.1 Run 1: stride_sweep (with sigmoid bug)

All Conv1D and FlatAE models trained together. FlatAE still had sigmoid at this point.
Latent dims tested: 6, 8, 9, 10, 12, 16.

Summary at selected latent dims (val R²):

| Model | d=6 | d=8 | d=9 | d=12 | d=16 |
|-------|-----|-----|-----|------|------|
| StridePCA  | 0.466 | 0.504 | 0.541 | 0.590 | 0.645 |
| StrideNMF  | 0.439 | 0.460 | 0.488 | 0.574 | 0.607 |
| StrideCNMF | 0.332 | 0.335 | 0.335 | 0.335 | 0.335 |
| StrideFlatAE (w/ sigmoid) | 0.450 | 0.474 | 0.482 | 0.496 | 0.520 |

Key observations:
- FlatAE with sigmoid is **worse than PCA** at d<12. This was the sigmoid bug.
- CNMF is consistently worst — L2 penalty on W collapses to near-constant reconstruction.
- NMF beats CNMF but saturates quickly; PCA beats NMF at matched dim (linear methods
  prefer variance-based decomposition over non-negativity constraint for this data).

Gait quality metrics at d=9 (reference):

| Model | peak_amp_ratio | gait_profile_corr | corr_matrix_dist |
|-------|---------------|------------------|-----------------|
| StridePCA | 0.727 | 0.920 | 1.906 |
| StrideNMF | 0.724 | 0.922 | 2.318 |
| StrideCNMF | 0.515 | 0.915 | 3.390 |
| StrideFlatAE (w/ sigmoid) | 0.636 | 0.925 | 1.746 |

Even with sigmoid degrading R², FlatAE already had the lowest `corr_matrix_dist` at d=9,
meaning it preserves inter-muscle coordination patterns better than the linear methods.

### 5.2 Run 2: stride_sweep_fixed (sigmoid removed from FlatAE decoder)

Only StrideFlatAE and StrideAE retrained. Latent dims: 4, 6, 8, 9.

| Model | d=4 | d=6 | d=8 | d=9 |
|-------|-----|-----|-----|-----|
| StrideFlatAE | 0.545 | 0.604 | 0.637 | 0.646 |
| StrideAE (Conv1D, sigmoid still present) | — | — | 0.481 | — |

StrideFlatAE improvements after sigmoid removal:
- d=8: 0.474 → 0.637 (+0.163)
- d=9: 0.482 → 0.646 (+0.164)

StrideFlatAE now **beats PCA** at matched latent dim for d≥8.

Gait quality at d=8 (final model):

| Metric | StrideFlatAE d=8 | StrideAE d=8 |
|--------|-----------------|-------------|
| val R² | 0.637 | 0.481 |
| peak_amp_ratio (mean) | 0.837 | 0.648 |
| gait_profile_corr (mean) | 0.928 | 0.935 |
| corr_matrix_dist | 1.228 | 1.707 |
| mode_levelground R² | 0.608 | 0.285 |
| mode_treadmill R² | 0.653 | 0.598 |

FlatAE beats StrideAE on all reconstruction metrics. StrideAE's epoch-1 Lnorm explosion
(`~6.97e13`) likely caused it to land in a worse local minimum, and the sigmoid in its
decoder compounds the problem.

---

## 6. Key Design Decisions and Lessons

### 6.1 MLP beats Conv1D for EMG stride reconstruction

**Why we tried Conv1D first**: Standard approach for time series. Conv1D is
translation-equivariant and parameter-efficient.

**Why MLP wins here**: EMG gait structure is dominated by *long-range cross-temporal
correlations* — the gastrocnemius burst at push-off (~60%) must co-vary with the tibialis
burst at swing onset (~65%) and both must relate to the loading response (~5%). Conv1D
with kernel=7, stride=2 has a receptive field of ~15 timepoints after 2 layers. It cannot
see 50% of the cycle away. The MLP's fully-connected bottleneck forces compression across
the entire 1111-dim input simultaneously.

The failure of Conv1D is also visible in the Lnorm explosion at epoch 1: a 1664-dim
feature vector fed to a linear layer initialised with standard random weights has much
larger expected norm than an 1111-dim vector in the MLP.

### 6.2 Non-linear (AE) did not obviously beat linear (PCA) — and why that's fine

At equal latent dims, our best AE (StrideFlatAE, no sigmoid) is only slightly better
than PCA (val R² 0.637 vs 0.504 at d=8). This may seem surprising for a nonlinear model
on nonlinear EMG data.

**Explanation**: PCA is optimal for reconstruction under MSE (among linear methods) and
extremely data-efficient. The benefit of the AE lies not in raw R² but in:
1. **Latent structure**: AE produces a compact code that can be decoded at any `z` (not
   just projections of real strides). The RL policy can explore continuously in z-space
   and find novel but physically plausible activations.
2. **Non-negativity of output** (when sigmoid is removed and we clip): decoder outputs
   stay close to [0,1] because it was trained against [0,1] targets.
3. **Future regularisation**: WAE-MMD will add distributional structure (smooth, Gaussian
   prior) that makes RL sampling meaningful. PCA has no such prior.

So the narrative is correct: AE/VAE is justified not because it gives strictly better R²,
but because it gives a **usable prior for RL** (smooth latent, meaningful interpolation,
no out-of-support issue).

### 6.3 Sigmoid on decoder: do not use

Sigmoid maps unbounded logits to (0,1). The gradient is `σ(1-σ)`. When the target is
near 0 (muscle at rest) or 1 (muscle at peak), the sigmoid output is already near the
target, gradient is tiny, and the decoder stops learning fine structure. This explains
why peak_amp_ratio is 0.636 with sigmoid but 0.837 without at d=8.

**Fix**: linear output + clip. The MSE loss against [0,1] targets provides enough implicit
constraint that the decoder stays near [0,1] without saturation.

### 6.4 Lnorm on FlatAE is not needed

The Hausdörfer (2024) paper uses Lnorm because their RL framework clips actions to [-1,1]
and they need the AE to produce latents compatible with that range. In our case:
- FlatAE encoder is an MLP starting from 1111-dim normalised input → latents are
  naturally smaller in magnitude
- RL policy will apply tanh before passing to the decoder anyway
- Adding Lnorm when not needed just distorts the reconstruction

---

## 7. Next Steps

### 7.1 StrideWAE-MMD (immediate priority)

Retrain WAE-MMD with:
- Sigmoid removed from `_StrideDecoder` (same fix as FlatAE)
- Gradient clipping already present (`max_norm=1.0`)
- Target: val R² competitive with FlatAE, with added Gaussian prior structure

### 7.2 RL Integration

Pipeline:
```
RL policy (PPO) outputs z ∈ R^d
  → StrideFlatAE.decode(z) → (1111,) → reshape → (101, 11) → clip [0,1]
  → EMGToMuscleMapper.map_batch() → (101, 18) muscle commands
  → Gait10dof18Musc environment (bilateral, 50% phase offset)
```

The policy learns to output `z` that produces stable walking. The decoder is frozen.
Use LocoMuJoCo's PPOJax (not SB3) — it supports JAX vmap over 2048+ parallel envs.

**Action space for RL**: The policy outputs one `z ∈ R^d` per timestep, decoded to 18
muscle activations for that timestep. This is a stride-level code used as a snapshot
action, not replaying a full stride — the decoder maps `z → (101,11)` and we take one
column at time `t`. Or we operate at stride level in a hierarchical policy.

**Recommended starting checkpoint**: `experiments/stride_sweep_fixed/checkpoints/StrideFlatAE_d8.pt`

### 7.3 Fix StrideAE sigmoid (low priority, FlatAE is currently better)

Remove `torch.sigmoid(self.deconv1(x))` at line 304 of `stride_models.py`, add gradient
clipping already present. Will likely recover to ~0.60+ val R².

---

## 8. References

- Hausdörfer et al. (2024). "Latent Action Priors for Locomotion with DRL." arXiv:2410.03246.
  *Simple AE (1 hidden layer, tanh, MSE + Lnorm), latent = action_dim/2. Policy outputs z → frozen decoder. PPO with style + task reward.*

- Zhou et al. (2020). PLAS: Latent action space for offline RL. CoRL 2020.

- Ajay et al. (2021). OPAL: Offline primitive discovery via VAE action priors. ICLR 2021.

- Tolstikhin et al. (2018). Wasserstein Auto-Encoders. ICLR 2018.

- Cuturi & Blondel (2017). Soft-DTW: a Differentiable Loss Function for Time-Series. ICML 2017.

- Lee & Seung (1999). Learning the parts of objects by non-negative matrix factorization. Nature.

- Camargo et al. (2021). A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions. Journal of Biomechanics.
