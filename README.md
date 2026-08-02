# Kinematic-Guided Trajectory Diffusion Model

> A Score-Based Diffusion Model for generating physically plausible UAV / drone flight trajectories from ArduPilot DataFlash logs.

---

## Table of Contents

1. [Mathematical Theory](#mathematical-theory)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration Reference](#configuration-reference)
6. [Output Format](#output-format)
7. [References](#references)

---

## Mathematical Theory

### 1. Forward (Noising) Process

Given a clean trajectory $\mathbf{x}_0 \in \mathbb{R}^{F \times L}$
(F = 7 features, L = sequence length), the forward process gradually
corrupts it with Gaussian noise over $T$ timesteps:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\,
  \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\;
  (1 - \bar{\alpha}_t)\,\mathbf{I}\right)$$

This can be sampled **in closed form** (no iterative application needed):

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0
               + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon},
\quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### 2. Cosine Noise Schedule

We use the **cosine variance schedule** (Nichol & Dhariwal, 2021) which
prevents $\beta_t$ from being negligibly small near $t = 0$:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)},
\qquad
f(t) = \cos^2\!\!\left(\frac{t/T + s}{1 + s}\cdot\frac{\pi}{2}\right)$$

$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}},
\qquad \beta_t \in [10^{-5},\; 0.9999]$$

where $s = 0.008$ is a small offset.

### 3. Score Network & Training Objective

A 1D U-Net $\epsilon_\theta(\mathbf{x}_t, t)$ is trained to predict the noise
injected at each timestep using the **simplified DDPM loss**:

$$\mathcal{L}_{\text{simple}} =
  \mathbb{E}_{t,\,\mathbf{x}_0,\,\boldsymbol{\epsilon}}\!\left[
  \|\boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{x}_t, t)\|^2
  \right]$$

The network is conditioned on $t$ via **sinusoidal time embeddings** and
**FiLM (Feature-wise Linear Modulation)**:

$$\text{emb}(t)_{2i}   = \sin\!\left(\frac{t}{10000^{2i/d}}\right),
\quad
\text{emb}(t)_{2i+1} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

### 4. DDIM Reverse Process (Sampling)

**DDIM** (Song et al., 2020) generates samples deterministically (or with
controlled stochasticity $\eta$) in $S \ll T$ steps:

| Symbol | Meaning |
|--------|---------|
| $\tau_i$ | $i$-th step in the DDIM subsequence |
| $\bar{\alpha}_{\tau_i}$ | Cumulative signal power at step $\tau_i$ |
| $\sigma_{\tau_i}$ | DDIM stochasticity coefficient |
| $\eta$ | Global noise level ($\eta=0$ → deterministic) |

**DDIM update rule:**

$$\mathbf{x}_{\tau_{i-1}}
= \sqrt{\bar{\alpha}_{\tau_{i-1}}}\,\hat{\mathbf{x}}_0^{\text{guided}}
+ \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} - \sigma_{\tau_i}^2}\cdot
  \hat{\boldsymbol{\epsilon}}_\theta
+ \sigma_{\tau_i}\,\mathbf{z}$$

where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and

$$\sigma_{\tau_i} = \eta
  \sqrt{\frac{1 - \bar{\alpha}_{\tau_{i-1}}}{1 - \bar{\alpha}_{\tau_i}}}
  \sqrt{1 - \frac{\bar{\alpha}_{\tau_i}}{\bar{\alpha}_{\tau_{i-1}}}}$$

### 5. Kinematic Guidance

During each DDIM step, we steer the predicted clean trajectory towards
physically plausible UAV motion by subtracting the gradient of a **kinematic
cost function** $J$:

**Cost function:**

$$J(\mathbf{s}) = \underbrace{\sum_{i=1}^{L-1}(\psi_{i+1} - \psi_i)^2}_{J_\psi:\;\text{yaw-rate penalty}}
+ \lambda\underbrace{\sum_{i=1}^{L-1}\|\mathbf{v}_{i+1} - \mathbf{v}_i\|^2}_{J_v:\;\text{velocity-change penalty}}$$

**Guided prediction:**

$$\hat{\mathbf{x}}_0^{\text{guided}}
= \hat{\mathbf{x}}_0 - w \cdot \nabla_{\hat{\mathbf{x}}_0} J(\hat{\mathbf{x}}_0)$$

The gradient $\nabla J$ is computed analytically via `torch.autograd.grad` —
no retraining is required.

---

## Repository Structure

```
Kinematic_Trajectory_Diffusion/
├── configs/
│   └── default.yaml          ← All hyperparameters
├── data/                     ← Put your .bin log files here (git-ignored)
├── outputs/                  ← Generated trajectory CSVs
├── checkpoints/              ← Saved model weights
├── src/
│   ├── __init__.py
│   ├── data_parser.py        ← ArduPilot .bin → NED+velocity tensor
│   ├── diffusion_math.py     ← CosineNoiseScheduler + KinematicGuidance
│   ├── unet_1d.py            ← 1D U-Net score network
│   ├── train.py              ← Training loop
│   └── sample.py             ← DDIM sampler with kinematic guidance
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd Kinematic_Trajectory_Diffusion

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Prepare Data

Place your ArduPilot `.bin` DataFlash logs in the `data/` directory:

```
data/
├── mission_01.bin
├── mission_02.bin
└── ...
```

The parser automatically:
- Detects **AUTO mode** segments (ArduCopter mode 3, ArduPlane mode 10)
- Discards segments shorter than `min_mission_samples`
- Extracts GPS → NED, AHR2 → yaw, NKF1 → velocity

### Step 2 — Train

```bash
python -m src.train --config configs/default.yaml
```

Training logs are saved to `checkpoints/loss_history.csv`.
Model checkpoints are saved every `checkpoint_every` epochs.

### Step 3 — Generate Trajectories

```bash
python -m src.sample \
    --config configs/default.yaml \
    --checkpoint checkpoints/epoch_0200.pt
```

Generated trajectories are exported to `outputs/generated_trajectories.csv`.

### Quick Architecture Sanity Check

```bash
python src/unet_1d.py
```

Expected output:
```
Input : torch.Size([4, 7, 256])
Output: torch.Size([4, 7, 256])
Params: 1,234,567
UNet1D sanity check PASSED ✓
```

---

## Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `data.sequence_length` | 256 | Fixed window length (samples) |
| `data.features` | 7 | `[N, E, D, vN, vE, vD, yaw]` |
| `data.min_mission_samples` | 100 | Min AUTO samples per segment |
| `data.stride` | 128 | Sliding window stride |
| `diffusion.T` | 1000 | Total diffusion timesteps |
| `diffusion.s` | 0.008 | Cosine schedule offset |
| `diffusion.guidance_scale` | 0.5 | Kinematic guidance strength $w$ |
| `diffusion.lambda_vel` | 0.1 | Velocity penalty weight $\lambda$ |
| `model.base_channels` | 64 | U-Net base width |
| `model.depth` | 3 | Encoder/decoder stages |
| `model.time_emb_dim` | 128 | Time embedding dimension |
| `training.batch_size` | 32 | Training batch size |
| `training.lr` | 1e-4 | AdamW learning rate |
| `training.epochs` | 200 | Total training epochs |
| `sampling.ddim_steps` | 50 | DDIM denoising steps |
| `sampling.eta` | 0.0 | DDIM stochasticity (0 = deterministic) |
| `sampling.num_samples` | 8 | Trajectories per generation run |

### Tuning Kinematic Guidance

| `guidance_scale` | Effect |
|------------------|--------|
| `0.0` | No guidance — raw diffusion output |
| `0.1 – 0.5` | Mild smoothing (recommended starting point) |
| `1.0+` | Strong smoothing; may reduce trajectory diversity |

---

## Output Format

The output CSV contains one row per trajectory step:

| Column | Unit | Description |
|--------|------|-------------|
| `sample_id` | — | Trajectory index (0 to num_samples-1) |
| `step` | — | Time step index |
| `N_m` | metres | North displacement from launch point |
| `E_m` | metres | East displacement from launch point |
| `D_m` | metres | Down displacement (negative = above origin) |
| `vN_ms` | m/s | North velocity |
| `vE_ms` | m/s | East velocity |
| `vD_ms` | m/s | Down velocity |
| `yaw_rad` | radians | Heading angle |

---

## References

1. **Ho et al. (2020)** — Denoising Diffusion Probabilistic Models.
   *NeurIPS 2020*. https://arxiv.org/abs/2006.11239

2. **Nichol & Dhariwal (2021)** — Improved Denoising Diffusion Probabilistic Models.
   *ICML 2021*. https://arxiv.org/abs/2102.09672

3. **Song et al. (2020)** — Denoising Diffusion Implicit Models (DDIM).
   *ICLR 2021*. https://arxiv.org/abs/2010.02502

4. **Dhariwal & Nichol (2021)** — Diffusion Models Beat GANs on Image Synthesis.
   *NeurIPS 2021*. https://arxiv.org/abs/2105.05233

5. **ArduPilot DataFlash Logging** — https://ardupilot.org/copter/docs/logmessages.html
