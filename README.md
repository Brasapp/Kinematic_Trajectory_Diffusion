# Kinematic-Guided Trajectory Diffusion Model

> A score-based diffusion model that generates physically plausible **AUTO-mode** flight trajectories for a high-speed, highly maneuvering fixed-wing UAV, learned from ArduPilot (ArduPlane) DataFlash logs.

The model generates time series of **roll, pitch and altitude**. A 2D/3D flight path is then rebuilt from roll with a coordinated-turn model (dead reckoning at the fleet's mean speed). During sampling, an optional **kinematic guidance** term pushes the samples towards physically consistent motion.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Data Pipeline](#data-pipeline)
3. [Mathematical Theory](#mathematical-theory)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Configuration Reference](#configuration-reference)
8. [Output Format](#output-format)
9. [Known Limitations](#known-limitations)
10. [References](#references)

---

## Pipeline Overview

```
.bin logs ──► data_parser ──► AUTO [roll, pitch, alt] @ 10 Hz ──► sliding windows (128 × 3)
                                                                   │
                                     altitude relative to window start, z-score
                                     normalisation, random left/right mirroring
                                                                   ▼
                                     train.py: 1D U-Net learns ε_θ(x_t, t)
                                                                   │
                         sample.py: DDIM (50 steps) + kinematic guidance
                                                                   │
                     denormalise ──► dead-reckon N/E/yaw from roll ──► CSV
                                                                   │
                                       evaluate.py: metrics + plots
```

---

## Data Pipeline

`src/data_parser.py` turns raw ArduPlane logs into clean, uniformly sampled flight segments.

### Sources

| Data | Message | Notes |
|------|---------|-------|
| Position (N, E, alt) | `XKF1` core 0: `PN`, `PE`, `-PD` | EKF3 (ArduPilot 4.x). `NKF1` (EKF2) is used for older firmware. |
| Velocity (vN, vE, vD) | `XKF1` core 0: `VN`, `VE`, `VD` | Same timeline as position, so the two stay consistent. |
| Attitude (roll, pitch, yaw) | `XKF1` core 0 | Degrees → radians; roll and yaw are wrapped to (−π, π]. |
| Airspeed | `CTUN.As` | Extra column, not used by the model. |
| Flight mode | `MODE.ModeNum` | ArduPlane mode numbers (AUTO = 10, STABILIZE = 2, …). |
| GPS fix quality | `GPS` (receiver with the most 3D fixes) | Only used to reject stretches where the EKF had no GPS. |

If a log has no EKF messages, the parser falls back to `GPS` for position/velocity and `ATT` (or `AHR2`) for attitude. Position sources are **never mixed**.

### Processing

1. **Resample** every stream onto a uniform grid (`resample_hz`, default 10 Hz → dt = 0.1 s). Angles are interpolated through sin/cos, so ±180° wraps (inverted flight, north crossings) stay correct.
2. **Filter**: keep only samples that are
   - in one of the selected flight `modes`,
   - flying (horizontal speed ≥ `min_speed_ms`),
   - densely logged (source sample spacing ≤ `max_gap_s`),
   - covered by a GPS 3D fix in the last `gps_timeout_s`,
   - free of EKF resets (position step within `max_pos_jump_m` of what velocity predicts).
3. **Segment**: contiguous kept runs of at least `min_mission_samples` samples become segments.
4. **Deduplicate**: byte-identical log files (the same flight saved under two names) are parsed once.
5. **Window**: `TrajectoryDataset` cuts sliding windows (`sequence_length`, `stride`). With `relative_alt`, altitude becomes the change since the window start. Each channel is then z-score normalised.
6. **Mirror** (training only): with `mirror_roll`, roll is negated for a random half of every batch, since the airframe can fly the same manoeuvre to the left or right. Roll is normalised symmetrically (mean 0), so a mirrored sample is exactly `-roll` in normalised space.

Each segment carries all columns in `FULL_COLUMNS` (`N, E, alt, vN, vE, vD, roll, pitch, yaw, airspeed`); `feature_names` picks the model channels.

### Current dataset (73 logs)

| | |
|---|---|
| Firmware | ArduPlane 4.5.4 – 4.6.3, all with `XKF1` |
| Unique logs | 70 (3 duplicate pairs skipped) |
| AUTO flight time (all lengths) | 24 min in 78 segments from 67 logs |
| Used for training | 43 AUTO segments ≥ 12.8 s → 203 windows (×2 with mirroring) |
| Mean ground speed in AUTO | 43 m/s |

AUTO legs are short (median about 20 s), which is why windows are 12.8 s (`sequence_length: 128`) with a stride of 32. With 25.6 s windows only 27 windows would remain. Setting `modes` to all flying modes gives 82 min (most of it pilot-flown STABILIZE), but this project models AUTO flight only.

---

## Mathematical Theory

### 1. Forward (Noising) Process

Given a clean window $\mathbf{x}_0 \in \mathbb{R}^{F \times L}$
(F = 3 channels: roll, pitch, altitude change; L = 128 samples), the forward process
corrupts it with Gaussian noise over $T$ timesteps:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\,
  \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\;
  (1 - \bar{\alpha}_t)\,\mathbf{I}\right)$$

which can be sampled in closed form:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0
               + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon},
\quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### 2. Cosine Noise Schedule

The **cosine variance schedule** (Nichol & Dhariwal, 2021) keeps $\beta_t$ from
being negligibly small near $t = 0$:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)},
\qquad
f(t) = \cos^2\!\!\left(\frac{t/T + s}{1 + s}\cdot\frac{\pi}{2}\right)$$

$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}},
\qquad \beta_t \in [10^{-5},\; 0.9999]$$

where $s = 0.008$.

### 3. Score Network & Training Objective

A 1D U-Net $\epsilon_\theta(\mathbf{x}_t, t)$ (≈1.8 M parameters) predicts the
injected noise. It is conditioned on $t$ through sinusoidal time embeddings and
**FiLM** (per-channel scale and shift after GroupNorm):

$$\text{emb}(t)_{2i}   = \sin\!\left(\frac{t}{10000^{2i/d}}\right),
\quad
\text{emb}(t)_{2i+1} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

The training loss is the simplified DDPM loss plus a smoothness penalty on the
predicted clean roll and pitch (second differences):

$$\mathcal{L} =
  \underbrace{\|\boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{x}_t, t)\|^2}_{\mathcal{L}_{\text{simple}}}
  + \lambda_{\text{smooth}}\,\bar\alpha_t \sum_{c \in \{\phi,\theta\}}
    \overline{\left(\hat{x}_{0,c}[i{+}1] - 2\hat{x}_{0,c}[i] + \hat{x}_{0,c}[i{-}1]\right)^2}$$

where $\hat{\mathbf{x}}_0 = \left(\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta\right)/\sqrt{\bar\alpha_t}$. The $\bar\alpha_t$ weight keeps the penalty to low-noise steps; without it, the $1/\sqrt{\bar\alpha_t}$ factor inflated the gradient about 10,000× at high noise and gave the sampler a strong one-sided roll bias.
The validation loss reports $\mathcal{L}_{\text{simple}}$ only.

### 4. DDIM Reverse Process (Sampling)

**DDIM** (Song et al., 2020) samples in $S \ll T$ steps:

$$\mathbf{x}_{\tau_{i-1}}
= \sqrt{\bar{\alpha}_{\tau_{i-1}}}\,\hat{\mathbf{x}}_0^{\text{guided}}
+ \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} - \sigma_{\tau_i}^2}\cdot
  \hat{\boldsymbol{\epsilon}}_\theta
+ \sigma_{\tau_i}\,\mathbf{z}$$

$$\sigma_{\tau_i} = \eta
  \sqrt{\frac{1 - \bar{\alpha}_{\tau_{i-1}}}{1 - \bar{\alpha}_{\tau_i}}}
  \sqrt{1 - \frac{\bar{\alpha}_{\tau_i}}{\bar{\alpha}_{\tau_{i-1}}}}$$

| Symbol | Meaning |
|--------|---------|
| $\tau_i$ | $i$-th step in the DDIM subsequence |
| $\sigma_{\tau_i}$ | DDIM stochasticity coefficient |
| $\eta$ | Global noise level ($\eta=0$ → deterministic) |

### 5. Kinematic Guidance

At every DDIM step the predicted clean window is denormalised to physical units
($\phi$ roll, $\theta$ pitch, $z$ altitude), and the gradient of a kinematic
cost $J$ is subtracted:

$$\hat{\mathbf{x}}_0^{\text{guided}}
= \hat{\mathbf{x}}_0 - w \cdot \operatorname{clip}\!\left(\nabla_{\hat{\mathbf{x}}_0} J,\,-1,\,1\right)$$

**Cost terms** (each averaged over the batch and time):

| Term | Formula | Meaning |
|------|---------|---------|
| Roll smoothness | $J_\phi = \left(\frac{\phi_{i+1}-\phi_i}{\Delta t}\right)^2$ | Penalises roll rate |
| Pitch–altitude coupling | $J_\theta = \left(\frac{z_{i+1}-z_i}{\Delta t} - V\sin\theta_i\right)^2$ | Climb rate should match pitch at nominal speed $V$ |
| Attitude limits | $J_b = \lambda_\phi\,\max(0,\lvert\phi\rvert-\phi_{\max})^2 + \lambda_\theta\left[\max(0,\theta_{\min}-\theta)^2 + \max(0,\theta-\theta_{\max})^2\right]$ | $\phi_{\max}=85°$, $\theta \in [-15°, 25°]$ |

$$J = \lambda_\phi J_\phi + \lambda_\theta J_\theta + J_b$$

$\Delta t$ and $V$ come from the dataset (0.1 s and the mean ground speed).
The gradient is computed with `torch.autograd.grad`; no retraining is needed to change $w$ or the weights.

### 6. Flight-Path Reconstruction

The model only generates roll, pitch and altitude. N, E and yaw are integrated
with a coordinated-turn model at constant speed $V$:

$$\psi_{i+1} = \psi_i + \frac{g\tan\phi_i}{V}\,\Delta t,\qquad
N_{i+1} = N_i + V\cos\psi_i\,\Delta t,\qquad
E_{i+1} = E_i + V\sin\psi_i\,\Delta t$$

---

## Repository Structure

```
Kinematic_Trajectory_Diffusion/
├── configs/
│   └── default.yaml          ← All settings
├── data/                     ← .bin logs (git-ignored)
├── checkpoints_v4/           ← Model weights + loss_history.csv (git-ignored)
├── outputs_v4/               ← Generated CSV + evaluation plots (git-ignored)
├── src/
│   ├── data_parser.py        ← .bin → filtered 10 Hz segments → TrajectoryDataset
│   ├── diffusion_math.py     ← CosineNoiseScheduler + KinematicGuidance
│   ├── unet_1d.py            ← 1D U-Net score network
│   ├── train.py              ← Training loop
│   ├── sample.py             ← DDIM sampler, guidance, flight-path reconstruction
│   └── evaluate.py           ← Metrics (MMD, yaw rate) and plots
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Brasapp/Kinematic_Trajectory_Diffusion.git
cd Kinematic_Trajectory_Diffusion

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

---

## Usage

### 1. Add logs and check them

Put the ArduPlane `.bin` logs in `data/`, then print per-log diagnostics:

```bash
python -m src.data_parser --config configs/default.yaml
```

Example output:

```
file                             src  fly_s  use_s sparse badnav segs    spd    as |roll|99        pitch           alt
2025-10-10 09-11-03.bin         XKF1     60     60      0      0    1   45.0  42.7       85   -30..39        1..65
2026-05-23 12-39-41.bin         XKF1     52     23      0     30    0
2025-04-18 09-58-36.bin       duplicate of 17 18-04-2025 10-12-26.bin
...
Total: 64 segments, 49433 samples (82.4 min), mean speed 30.2 m/s, segments >= 256 samples.
```

| Column | Meaning |
|--------|---------|
| `fly_s` | Seconds above `min_speed_ms` |
| `use_s` | Seconds that pass every filter |
| `sparse` / `badnav` | Flying seconds dropped for sparse logging / no GPS fix or EKF reset |
| `segs` | Segments long enough to keep |
| `spd`, `as` | Mean ground speed and airspeed (m/s) |
| `\|roll\|99` | 99th percentile of absolute roll (°) |

### 2. Train

```bash
python -m src.train --config configs/default.yaml
```

Losses are appended to `checkpoints_v4/loss_history.csv`; a checkpoint is saved every `checkpoint_every` epochs.
Training **resumes from the newest checkpoint** in `checkpoint_dir`. Use a new folder to start from scratch.

### 3. Generate trajectories

```bash
python -m src.sample --config configs/default.yaml --checkpoint checkpoints_v4/epoch_0800.pt
```

### 4. Evaluate

```bash
python -m src.evaluate --config configs/default.yaml --generated outputs_v4/generated_trajectories.csv
```

Prints a report (yaw-rate statistics, MMD² between real and generated windows) and saves to `output_dir`:
`eval_2d_trajectories.png`, `eval_3d_trajectories.png`, `eval_feature_histograms.png`, `eval_yaw_series.png`, `eval_loss_curve.png`.

### Architecture sanity check

```bash
python src/unet_1d.py
```

---

## Configuration Reference

### `data`

| Key | Default | Description |
|-----|---------|-------------|
| `data_dir` | `data/` | Folder with `.bin` logs |
| `output_dir` | `outputs_v4/` | Plots and generated CSV |
| `checkpoint_dir` | `checkpoints_v4/` | Checkpoints and loss log |
| `sequence_length` | 128 | Window length in samples (12.8 s at 10 Hz) |
| `features` | 3 | Number of model channels |
| `feature_names` | `[roll, pitch, alt]` | Model channels, from `FULL_COLUMNS` |
| `resample_hz` | 10.0 | Uniform grid rate |
| `modes` | `[AUTO]` | ArduPlane modes to keep |
| `min_speed_ms` | 10.0 | Minimum horizontal speed to count as flying |
| `max_gap_s` | 0.3 | Maximum source sample spacing |
| `gps_timeout_s` | 1.0 | Maximum age of the last GPS 3D fix |
| `max_pos_jump_m` | 3.0 | EKF-reset threshold (position step vs velocity) |
| `min_mission_samples` | 128 | Minimum segment length (keep ≥ `sequence_length`) |
| `stride` | 32 | Sliding-window stride |
| `relative_alt` | true | Altitude as change since the window start |
| `mirror_roll` | true | Randomly mirror roll during training |

### `diffusion`

| Key | Default | Description |
|-----|---------|-------------|
| `T` | 1000 | Diffusion timesteps |
| `s` | 0.008 | Cosine schedule offset |
| `guidance_scale` | 0.0 | Guidance strength $w$ (0 = off; see limitations) |
| `lambda_roll` | 0.5 | Weight of roll smoothness and roll limit |
| `lambda_pitch` | 1.0 | Weight of pitch–altitude coupling and pitch limits |
| `lambda_vel` | 0.1 | Not used by the current 3-channel model |

### `model`

| Key | Default | Description |
|-----|---------|-------------|
| `in_channels` | 3 | Must match `data.features` |
| `base_channels` | 64 | U-Net base width |
| `depth` | 3 | Encoder/decoder stages |
| `time_emb_dim` | 128 | Time-embedding size |

### `training`

| Key | Default | Description |
|-----|---------|-------------|
| `batch_size` | 32 | Batch size |
| `lr` | 1e-4 | AdamW learning rate |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `epochs` | 800 | Total epochs (about 6 batches each) |
| `checkpoint_every` | 50 | Checkpoint interval (epochs) |
| `num_workers` | 0 | DataLoader workers (0 is fastest for this small dataset on Windows) |
| `seed` | 42 | Random seed (also fixes the train/val split) |
| `grad_clip` | 1.0 | Gradient-norm clip |
| `lambda_smooth` | 0.1 | Weight of the roll/pitch smoothness loss |

### `sampling`

| Key | Default | Description |
|-----|---------|-------------|
| `num_samples` | 64 | Trajectories per run |
| `ddim_steps` | 50 | DDIM steps |
| `eta` | 0.0 | DDIM stochasticity (0 = deterministic) |
| `output_csv` | `outputs_v4/generated_trajectories.csv` | Output file |

---

## Output Format

`generated_trajectories.csv` has one row per time step (dt = 0.1 s):

| Column | Unit | Description |
|--------|------|-------------|
| `sample_id` | — | Trajectory index (0 … `num_samples` − 1) |
| `step` | — | Time-step index |
| `N_m` | m | North, reconstructed from roll (starts at 0) |
| `E_m` | m | East, reconstructed from roll (starts at 0) |
| `alt_m` | m | Altitude change since the start of the window (generated) |
| `yaw_rad` | rad | Heading, reconstructed from roll (starts at 0) |
| `roll_rad` | rad | Roll (generated) |
| `pitch_rad` | rad | Pitch (generated) |

---

## Known Limitations

- **The evaluation's "real" flight paths are reconstructed too.** Real N/E/yaw in the plots are rebuilt from real roll with the same model as the generated ones, not taken from GPS. The roll, pitch and altitude histograms are the direct comparison.
- **Guidance is switched off** (`guidance_scale: 0`). With w = 2 and the gradient clipped to ±1, each DDIM step could move a value by up to 2 standard deviations, which turned altitude into a ±50 m zig-zag. The roll-rate penalty and 85° roll limit also work against the aircraft's real manoeuvres (up to about 400°/s roll).
- **Pitch–altitude coupling ignores angle of attack.** Level flight at a few degrees nose-up reads as a climb.
- **The train/val split is random over overlapping windows**, so the validation loss is optimistic. Split by log file for a fair check.
- **Normalisation statistics are not stored in the checkpoint.** `sample.py` and `evaluate.py` re-parse `data/` to recover them, so the logs must not change between training and sampling.
- **Roll is wrapped to ±180°.** Inverted flight creates jumps between +180° and −180° in the model input.

---

## References

1. **Ho et al. (2020)** — Denoising Diffusion Probabilistic Models. *NeurIPS 2020*. https://arxiv.org/abs/2006.11239
2. **Nichol & Dhariwal (2021)** — Improved Denoising Diffusion Probabilistic Models. *ICML 2021*. https://arxiv.org/abs/2102.09672
3. **Song et al. (2020)** — Denoising Diffusion Implicit Models (DDIM). *ICLR 2021*. https://arxiv.org/abs/2010.02502
4. **Perez et al. (2018)** — FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI 2018*. https://arxiv.org/abs/1709.07871
5. **ArduPilot DataFlash log messages (ArduPlane)** — https://ardupilot.org/plane/docs/logmessages.html
