"""
sample.py
=========
Reverse diffusion (DDIM) sampler with Kinematic Guidance.

DDIM Algorithm (Song et al. 2020, Algorithm 1)
-----------------------------------------------
DDIM accelerates sampling by evaluating only a subsequence
:math:`\\{\\tau_1, \\tau_2, \\ldots, \\tau_S\\}` of the full :math:`T` timesteps,
where :math:`S \\ll T` (e.g. 50 steps vs 1000).

For each step :math:`i` going from :math:`\\tau_S` down to :math:`\\tau_1`:

**Step 1 — Predict noise:**

.. math::

    \\hat{\\boldsymbol{\\epsilon}}_\\theta
        = \\epsilon_\\theta(\\mathbf{x}_{\\tau_i},\\, \\tau_i)

**Step 2 — Reconstruct predicted clean trajectory:**

.. math::

    \\hat{\\mathbf{x}}_0
        = \\frac{\\mathbf{x}_{\\tau_i}
                 - \\sqrt{1 - \\bar{\\alpha}_{\\tau_i}}\\,\\hat{\\boldsymbol{\\epsilon}}_\\theta}
                {\\sqrt{\\bar{\\alpha}_{\\tau_i}}}

**Step 3 — Kinematic guidance (applied to** :math:`\\hat{\\mathbf{x}}_0` **):**

.. math::

    \\hat{\\mathbf{x}}_0^{\\text{guided}}
        = \\hat{\\mathbf{x}}_0
        - w \\cdot \\nabla_{\\hat{\\mathbf{x}}_0} J(\\hat{\\mathbf{x}}_0)

where :math:`J` is the kinematic cost function (yaw-rate + velocity-change
penalty) and :math:`w` is the guidance scale.

**Step 4 — DDIM update:**

.. math::

    \\mathbf{x}_{\\tau_{i-1}}
        = \\sqrt{\\bar{\\alpha}_{\\tau_{i-1}}}\\,\\hat{\\mathbf{x}}_0^{\\text{guided}}
        + \\underbrace{\\sqrt{1 - \\bar{\\alpha}_{\\tau_{i-1}} - \\sigma_{\\tau_i}^2}
                       \\cdot \\hat{\\boldsymbol{\\epsilon}}_\\theta}_{\\text{direction towards }x_{\\tau_i}}
        + \\sigma_{\\tau_i}\\,\\boldsymbol{z}

**DDIM sigma:**

.. math::

    \\sigma_{\\tau_i}
        = \\eta \\sqrt{
            \\frac{1 - \\bar{\\alpha}_{\\tau_{i-1}}}
                 {1 - \\bar{\\alpha}_{\\tau_i}}
        } \\cdot
        \\sqrt{1 - \\frac{\\bar{\\alpha}_{\\tau_i}}{\\bar{\\alpha}_{\\tau_{i-1}}}}

Setting :math:`\\eta = 0` makes sampling **fully deterministic** (DDIM);
setting :math:`\\eta = 1` recovers stochastic DDPM-like behaviour.

Usage
-----
    python -m src.sample \\
        --config configs/default.yaml \\
        --checkpoint checkpoints/epoch_0200.pt
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.data_parser import load_split, select_features, TrajectoryDataset
from src.diffusion_math import CosineNoiseScheduler, KinematicGuidance
from src.unet_1d import build_model, core_module

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

FEATURE_NAMES = ["N_m", "E_m", "alt_m", "yaw_rad", "roll_rad", "pitch_rad"]


# ──────────────────────────────────────────────────────────────────────────────
# DDIM SAMPLER
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    scheduler: CosineNoiseScheduler,
    guidance: KinematicGuidance,
    *,
    num_samples: int,
    seq_len: int,
    features: int,
    ddim_steps: int,
    eta: float,
    device: torch.device,
) -> torch.Tensor:
    r"""Generate synthetic trajectories via kinematic-guided DDIM sampling.

    Parameters
    ----------
    model       : trained UNet1D score network
    scheduler   : CosineNoiseScheduler (pre-computed schedule)
    guidance    : KinematicGuidance instance
    num_samples : number of trajectories to generate (batch size B)
    seq_len     : sequence length L
    features    : number of feature channels F (9)
    ddim_steps  : number of DDIM denoising steps S (e.g. 50)
    eta         : DDIM stochasticity coefficient (0 = deterministic)
    device      : torch.device

    Returns
    -------
    x_generated : Tensor[B, F, L]  — generated normalised trajectories
    """
    model.eval()
    T = scheduler.T

    # ── DDIM subsequence: S evenly spaced indices in [0, T-1] ─────────────
    # tau[0] = T-1 (most noisy), tau[-1] = 0 (clean)
    tau = torch.linspace(0, T - 1, ddim_steps + 1).long()
    tau = tau.flip(0)  # [S+1]  e.g. [999, 979, ..., 20, 0]

    # ── Start from pure Gaussian noise: x_{T} ~ N(0, I) ──────────────────
    x = torch.randn(num_samples, features, seq_len, device=device)

    logger.info(
        f"DDIM sampling: {ddim_steps} steps | "
        f"eta={eta} | guidance_scale={guidance.w}"
    )

    # ── Reverse diffusion loop: τ_S → τ_{S-1} → ... → τ_1 ───────────────
    for i in tqdm(range(len(tau) - 1), desc="Sampling", unit="step"):
        t_curr = tau[i].item()    # e.g. 999
        t_prev = tau[i + 1].item()  # e.g. 979

        t_batch = torch.full((num_samples,), t_curr, device=device, dtype=torch.long)

        # ── Step 1: Predict noise ─────────────────────────────────────────
        # ε̂_θ = model(x_{τ_i}, τ_i)
        eps_pred = model(x, t_batch)                  # [B, F, L]

        # ── Step 2: Reconstruct x̂_0 from x_{τ_i} and ε̂_θ ────────────────
        # x̂_0 = (x_{τ_i} - sqrt(1 - ᾱ_{τ_i}) * ε̂) / sqrt(ᾱ_{τ_i})
        ab_t = scheduler.alpha_bars[t_curr]           # scalar

        sqrt_ab_t        = ab_t.sqrt()
        sqrt_one_minus_t = (1.0 - ab_t).sqrt()

        x0_pred = (x - sqrt_one_minus_t * eps_pred) / sqrt_ab_t
        x0_pred = x0_pred.clamp(-5.0, 5.0)           # soft clamp for stability

        # ── Step 3: Apply kinematic guidance to x̂_0 ─────────────────────
        # x̂_0^guided = x̂_0 - w * ∇_{x̂_0} J(x̂_0)
        x0_pred = guidance.apply(x0_pred, alpha_bar=ab_t)  # [B, F, L]

        # ── Step 4: DDIM update ───────────────────────────────────────────
        ab_prev = scheduler.alpha_bars[t_prev]        # scalar

        # DDIM sigma_τ:
        #   σ_τ = η * sqrt( (1 - ᾱ_{t-1}) / (1 - ᾱ_t) ) * sqrt(1 - ᾱ_t / ᾱ_{t-1})
        ratio     = (1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8)
        ab_ratio  = (ab_t / ab_prev.clamp(min=1e-8)).clamp(max=1.0)
        sigma_tau = eta * ratio.sqrt() * (1.0 - ab_ratio).sqrt()

        # Direction towards x_{τ_i}:
        #   dir = sqrt(1 - ᾱ_{t-1} - σ²) * ε̂_θ
        direction_coef = (1.0 - ab_prev - sigma_tau ** 2).clamp(min=0.0).sqrt()

        # Noise term: z ~ N(0, I)  (zero at the final step t_prev == 0)
        z = torch.randn_like(x) if (eta > 0.0 and t_prev > 0) else torch.zeros_like(x)

        # x_{t-1} = sqrt(ᾱ_{t-1}) * x̂_0^guided
        #          + sqrt(1 - ᾱ_{t-1} - σ²) * ε̂_θ
        #          + σ * z
        x = ab_prev.sqrt() * x0_pred + direction_coef * eps_pred + sigma_tau * z

    return x  # [B, F, L]  — still normalised


# ──────────────────────────────────────────────────────────────────────────────
# MAIN SAMPLING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_trajectory(roll: np.ndarray, pitch: np.ndarray, alt: np.ndarray,
                           v_nominal: float, dt: float, max_bank_deg: float = 85.0) -> np.ndarray:
    """
    Reconstruct full trajectory using forward integration (dead-reckoning).
    
    Parameters
    ----------
    roll, pitch, alt : ndarray[L]
    v_nominal : float
    dt : float
    
    Returns
    -------
    traj : ndarray[L, 6] -> [N_m, E_m, alt_m, yaw_rad, roll_rad, pitch_rad]
    """
    L = len(roll)
    N = np.zeros(L)
    E = np.zeros(L)
    yaw = np.zeros(L)
    g = 9.81
    max_bank = np.radians(max_bank_deg)

    for i in range(L - 1):
        # Coordinated turn. Bank is capped for the turn rate: tan() explodes near
        # 90 deg, while the real aircraft tops out around 100-130 deg/s.
        bank = np.clip(roll[i], -max_bank, max_bank)
        yaw[i+1] = yaw[i] + (g * np.tan(bank) / v_nominal) * dt
        N[i+1] = N[i] + v_nominal * np.cos(yaw[i]) * dt
        E[i+1] = E[i] + v_nominal * np.sin(yaw[i]) * dt
        
    return np.stack([N, E, alt, yaw, roll, pitch], axis=1)

def calibrate_guidance(dataset, alt_ch: int = 2, pitch_ch: int = 1, roll_ch: int = 0) -> dict:
    """Fit the guidance constants on the real (training) windows.

    Returns climb_gain g and climb_bias b of  climb ≈ g·sin(pitch) + b
    (least squares), the residual RMS of that fit, and the RMS of the roll
    second difference.
    """
    x = dataset.denormalise(dataset.data).numpy()        # [N, F, L] physical
    dt = dataset.mean_dt_s
    climb = np.diff(x[:, alt_ch], axis=1) / dt
    sp = np.sin(x[:, pitch_ch, :-1])
    A = np.stack([sp.ravel(), np.ones(sp.size)], 1)
    (g, b), *_ = np.linalg.lstsq(A, climb.ravel(), rcond=None)
    resid = climb - (g * sp + b)
    d2 = np.diff(x[:, roll_ch], 2, axis=1)
    return {
        "climb_gain": float(g),
        "climb_bias": float(b),
        "climb_resid_std": float(np.sqrt((resid ** 2).mean())),
        "roll_d2_std": float(np.sqrt((d2 ** 2).mean())),
    }


def build_guidance(config: dict, stats: dict, calibration: dict) -> KinematicGuidance:
    """Create the KinematicGuidance object from config, normalisation stats
    (``TrajectoryDataset.stats``) and ``calibrate_guidance`` output."""
    d = config["diffusion"]
    cal = calibration
    logger.info(
        "Guidance calibration: climb = %.1f*sin(pitch) %+.2f m/s (resid %.2f m/s), "
        "roll d2 rms %.2f deg" % (cal["climb_gain"], cal["climb_bias"],
                                  cal["climb_resid_std"], np.degrees(cal["roll_d2_std"])))
    return KinematicGuidance(
        guidance_scale=d.get("guidance_scale", 0.0),
        lambda_roll=d.get("lambda_roll", 1.0),
        lambda_pitch=d.get("lambda_pitch", 1.0),
        lambda_bound=d.get("lambda_bound", 1.0),
        roll_channel=0, pitch_channel=1, alt_channel=2,
        dt=stats["mean_dt_s"],
        max_roll_deg=d.get("max_roll_deg", 90.0),
        min_pitch_deg=d.get("min_pitch_deg", -20.0),
        max_pitch_deg=d.get("max_pitch_deg", 40.0),
        anchor_alt=d.get("anchor_alt", True),
        pitch_drives_alt=d.get("pitch_drives_alt", True),
        iters=d.get("guidance_iters", 1),
        means=torch.tensor(stats["mean"], dtype=torch.float32),
        stds=torch.tensor(stats["std"], dtype=torch.float32),
        **cal,
    )


def load_checkpoint_for_sampling(config: dict, checkpoint_path: str, device) -> tuple:
    """Load model + the metadata stored at training time.

    Returns (model, stats, calibration, mean_speed_ms).  Checkpoints from before
    metadata was stored fall back to re-parsing the training logs.
    """
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    core_module(model).load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}  (epoch {ckpt['epoch']+1}, "
                f"{type(model).__name__})")
    if "norm" in ckpt:
        return model, ckpt["norm"], ckpt["guidance_calibration"], ckpt["mean_speed_ms"]
    logger.warning("Checkpoint has no stored statistics; re-parsing the training logs.")
    split = load_split(config)
    ds = TrajectoryDataset.from_config(
        select_features(split["train"], split["feat_idx"]), config["data"], split["dt_s"])
    return model, ds.stats, calibrate_guidance(ds), split["mean_speed_ms"]


def denormalise(x: torch.Tensor, stats: dict) -> torch.Tensor:
    """Normalised [B, F, L] -> physical units."""
    mean = torch.tensor(stats["mean"], device=x.device, dtype=x.dtype).view(1, -1, 1)
    std = torch.tensor(stats["std"], device=x.device, dtype=x.dtype).view(1, -1, 1)
    return x * std + mean


def generate(config: dict, model, stats: dict, calibration: dict, device,
             num_samples: int = None, seed: int = None) -> np.ndarray:
    """Run guided DDIM and return physical [B, F, L] (roll, pitch, alt)."""
    scheduler = CosineNoiseScheduler(
        T=config["diffusion"]["T"], s=config["diffusion"]["s"], device=str(device))
    guidance = build_guidance(config, stats, calibration)
    seed = config["sampling"].get("seed") if seed is None else seed
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        x_norm = ddim_sample(
            model, scheduler, guidance,
            num_samples=num_samples or config["sampling"]["num_samples"],
            seq_len=config["data"]["sequence_length"],
            features=config["data"]["features"],
            ddim_steps=config["sampling"]["ddim_steps"],
            eta=config["sampling"]["eta"],
            device=device,
        )
    return denormalise(x_norm, stats).cpu().numpy()


def sample(config: dict, checkpoint_path: str):
    """Load a trained model and generate synthetic UAV trajectories.

    Parameters
    ----------
    config          : parsed YAML config dict
    checkpoint_path : path to a ``.pt`` checkpoint file
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, stats, calibration, mean_speed_ms = load_checkpoint_for_sampling(
        config, checkpoint_path, device)
    x_phys = generate(config, model, stats, calibration, device)   # [B, 3, L]

    # ── Export to CSV ─────────────────────────────────────────────────────────
    output_path = Path(config["sampling"]["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    B, F, L = x_phys.shape
    rows = []
    for sample_idx in range(B):
        roll, pitch, alt = x_phys[sample_idx]
        traj_6d = reconstruct_trajectory(roll, pitch, alt, mean_speed_ms, stats["mean_dt_s"])
        for step_idx, features_row in enumerate(traj_6d):
            row = {"sample_id": sample_idx, "step": step_idx}
            for feat_name, val in zip(FEATURE_NAMES, features_row):
                row[feat_name] = float(val)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {B} trajectories ({L} steps each) → {output_path}")
    logger.info(f"Preview:\n{df[df['sample_id'] == 0].head(5).to_string(index=False)}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic UAV trajectories via kinematic-guided DDIM"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sample(cfg, args.checkpoint)
