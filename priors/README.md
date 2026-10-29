This folder contains the frozen EMG decoders used as action priors in the experiments, and the diagnostics of how they were fitted.

The EMG comes from [Camargo et al. 2021](https://doi.org/10.1016/j.jbiomech.2021.110320), subject AB06 (1.80 m, 74.8 kg, the closest match to the SCONE models by height and mass), 11 right-leg channels, all four locomotion modes (level ground, ramp, stair, treadmill). `leaps preprocess` turns the raw recordings into 1984 time-normalized strides; `docs/eda/` shows each preprocessing stage.

Every folder is the output of

```bash
python -m leaps.scripts.train_single_subject_decoder --subject AB06 --k <k> \
  --threshold 0.6 --scale 0.9 --depth 1 --seed 0
```

Rerunning it reproduces `decoder.pt` and `norm.npz` bit for bit, and overwrites the folder.

| folder | latent dim | held-out R² | used for |
|---|---|---|---|
| `decoder_k2_AB06` | 2 | 0.63 | latent dimension sweep |
| `decoder_k4_AB06` | 4 | 0.84 | latent dimension sweep |
| `decoder_k6_AB06` | 6 | 0.94 | all other experiments |
| `decoder_k8_AB06` | 8 | 0.97 | latent dimension sweep |
| `decoder_k11_AB06` | 11 | 0.98 | latent dimension sweep |

Files in each folder:

| file | content |
|---|---|
| `decoder.pt` | decoder state dict: `Linear(k, 2k)`, tanh, `Linear(2k, 11)`, sigmoid. This is the prior loaded by `leaps.envs.LatentActionPriorWrapper`. |
| `encoder.pt` | encoder state dict: `Linear(11, 2k)`, tanh, `Linear(2k, k)`. Not used in RL, only to project EMG into the latent space. |
| `norm.npz` | `p99`: per-channel divisor applied to the EMG before training (mean of the per-stride 99th percentile, training split); `p01`: zeros. `z_lo`, `z_hi`: per-dimension range of the training latents, onto which the policy's latent action in [-1, 1] is rescaled. `whiten_A`, `whiten_mu`: mean and 3σ principal axes of the training latents, not used by any thesis run. |
| `results.yaml` | hyperparameters, stride and frame counts, fit-set and held-out MSE / MAE / R², cumulative PCA variance. |
| `ae_training.pdf` | training loss (reconstruction + latent-norm penalty) and held-out reconstruction loss per epoch. |
| `pca.pdf` | cumulative explained variance of the training frames against the number of principal components; dashed line at k. |

The held-out set is a random 20% of AB06's strides (396 strides), so it measures fit on the same subject, not generalization to other subjects. Channel order is `EMG_CHANNELS` in `src/leaps/envs/emg_mapping.py`, which also holds the channel-to-muscle maps for H0918, H1622 and H2190.
