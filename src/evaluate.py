"""
evaluate.py
===========
Quantitative and visual evaluation of the Kinematic-Guided Trajectory
Diffusion Model.

Three evaluation pillars
------------------------

1. Kinematic Metrics  (physics-based, per-trajectory)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   - **Mean Yaw Rate** :math:`\\bar{|\\Delta\\psi|}` — average absolute turn rate per step (rad/step).
     Lower = smoother turns.
   - **Max Yaw Rate** :math:`\\max|\\Delta\\psi|` — worst-case single-step heading change.
   - **Mean Speed** :math:`\\bar{\\|\\mathbf{v}\\|_2}` — average 3D flight speed (m/s).
   - **Velocity Smoothness** :math:`\\bar{\\|\\Delta\\mathbf{v}\\|_2}` — average step-to-step speed
     change (acceleration proxy, m/s per step).
   - **Kinematic Cost** :math:`J(s) = \\sum(\\Delta\\psi)^2 + \\lambda\\sum\\|\\Delta\\mathbf{v}\\|^2`
     — the exact cost minimised by guidance.  Lower = better.

2. Distribution Metrics  (real vs generated)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   **Maximum Mean Discrepancy (MMD)** with an RBF kernel:

   .. math::

       \\text{MMD}^2(P, Q)
           = \\mathbb{E}_{x,x' \\sim P}[k(x,x')]
           - 2\\,\\mathbb{E}_{x \\sim P,\\, y \\sim Q}[k(x,y)]
           + \\mathbb{E}_{y,y' \\sim Q}[k(y,y')]

   where :math:`k(x,y) = \\exp\\!\\left(-\\|x-y\\|^2 / (2\\sigma^2)\\right)`.

   MMD ≈ 0 means the two distributions are statistically indistinguishable.
   MMD >> 0 means the generated distribution is systematically different.

   Per-trajectory summary statistics (mean + std of each feature channel)
   are used as the feature vector so that the metric is tractable.

3. Visual Outputs  (saved to outputs/)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   - **3D trajectory plot** — real (blue) vs generated (orange) flight paths
   - **Feature histograms** — per-channel distribution comparison
   - **Loss curve** — train vs val loss from ``checkpoints/loss_history.csv``
   - **Yaw time-series** — real vs generated yaw angle over time

Usage
-----
    python -m src.evaluate \\
        --config configs/default.yaml \\
        --generated outputs/generated_trajectories.csv
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import yaml

from src.data_parser import parse_directory, TrajectoryDataset
from src.diffusion_math import KinematicGuidance

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

FEATURE_NAMES = ["N_m", "E_m", "D_m", "vN_ms", "vE_ms", "vD_ms", "yaw_rad", "roll_rad", "pitch_rad"]
FEATURE_LABELS = ["N (m)", "E (m)", "D (m)", "vN (m/s)", "vE (m/s)", "vD (m/s)", "Yaw (rad)", "Roll (rad)", "Pitch (rad)"]


# ──────────────────────────────────────────────────────────────────────────────
# 1.  KINEMATIC METRICS
# ──────────────────────────────────────────────────────────────────────────────

def kinematic_metrics(trajs: np.ndarray, diff_cfg: dict, mean_dt_s: float) -> dict:
    r"""Compute per-trajectory kinematic statistics and return their means.

    Parameters
    ----------
    trajs      : ndarray[N, 9, L]  — trajectories in physical units, channel-first
    diff_cfg   : dict              — diffusion config containing lambda weights
    mean_dt_s  : float             — mean timestep for coordinated turn constraint

    Returns
    -------
    dict with keys:
        mean_yaw_rate, max_yaw_rate, mean_speed,
        velocity_smoothness, kinematic_cost
    """
    N, F, L = trajs.shape

    # ── Yaw (channel 6) ──────────────────────────────────────────────────────
    yaw       = trajs[:, 6, :]              # [N, L]
    delta_psi = np.diff(yaw, axis=-1)      # [N, L-1]

    mean_yaw_rate = np.abs(delta_psi).mean(axis=-1).mean()   # scalar
    max_yaw_rate  = np.abs(delta_psi).max(axis=-1).mean()    # scalar (avg of per-traj max)

    # ── Velocity (channels 3,4,5) ─────────────────────────────────────────────
    vel   = trajs[:, 3:6, :]               # [N, 3, L]
    speed = np.linalg.norm(vel, axis=1)    # [N, L]   3D speed

    mean_speed = speed.mean(axis=-1).mean()  # scalar

    # ── Velocity smoothness: avg ||Δv||₂ ─────────────────────────────────────
    delta_v   = np.diff(vel, axis=-1)      # [N, 3, L-1]
    vel_smooth = np.linalg.norm(delta_v, axis=1).mean(axis=-1).mean()  # scalar

    # ── Roll and Pitch ───────────────────────────────────────────────────────
    roll  = trajs[:, 7, :]
    pitch = trajs[:, 8, :]
    delta_roll  = np.diff(roll,  axis=-1)
    delta_pitch = np.diff(pitch, axis=-1)

    # ── Kinematic cost J(s) ──────────────────────────────────────────────────
    J_vel   = (np.linalg.norm(delta_v, axis=1) ** 2).mean(axis=-1).mean()
    J_roll  = (delta_roll ** 2).mean(axis=-1).mean()
    J_pitch = (delta_pitch ** 2).mean(axis=-1).mean()

    # Coordinated turn constraint error: Δψ - (g·dt/V)·tan(φ)
    speed_i = speed[:, :-1] + 1e-4
    roll_i  = roll[:, :-1]
    coord_err = delta_psi - (9.81 * mean_dt_s / speed_i) * np.tan(roll_i)
    J_coord = (coord_err ** 2).mean(axis=-1).mean()

    lam_v = diff_cfg.get("lambda_vel", 0.1)
    lam_r = diff_cfg.get("lambda_roll", 1.0)
    lam_p = diff_cfg.get("lambda_pitch", 0.5)
    lam_c = diff_cfg.get("lambda_coord", 2.0)

    J_cost = lam_v * J_vel + lam_r * J_roll + lam_p * J_pitch + lam_c * J_coord

    return {
        "mean_yaw_rate_rad":   float(mean_yaw_rate),
        "max_yaw_rate_rad":    float(max_yaw_rate),
        "mean_speed_ms":       float(mean_speed),
        "velocity_smoothness": float(vel_smooth),
        "kinematic_cost_J":    float(J_cost),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DISTRIBUTION METRIC: MMD
# ──────────────────────────────────────────────────────────────────────────────

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    r"""Compute the RBF kernel matrix between rows of X and Y.

    .. math::

        K_{ij} = \exp\!\left(-\frac{\|X_i - Y_j\|^2}{2\sigma^2}\right)

    Parameters
    ----------
    X : ndarray[n, d]
    Y : ndarray[m, d]
    sigma : float — bandwidth

    Returns
    -------
    K : ndarray[n, m]
    """
    # ||X_i - Y_j||² via broadcasting
    diff = X[:, None, :] - Y[None, :, :]      # [n, m, d]
    sq   = (diff ** 2).sum(axis=-1)            # [n, m]
    return np.exp(-sq / (2.0 * sigma ** 2))


def compute_mmd(real: np.ndarray, generated: np.ndarray,
                sigma: float = None) -> float:
    r"""Compute the squared Maximum Mean Discrepancy (MMD²) with an RBF kernel.

    .. math::

        \text{MMD}^2(P, Q)
            = \mathbb{E}[k(x,x')]
            - 2\,\mathbb{E}[k(x,y)]
            + \mathbb{E}[k(y,y')]

    where :math:`x, x' \sim P` (real) and :math:`y, y' \sim Q` (generated).

    Feature representation: per-trajectory (mean, std) of each channel →
    concatenated to a 14-dim vector per trajectory.

    Parameters
    ----------
    real      : ndarray[N_r, 7, L]
    generated : ndarray[N_g, 7, L]
    sigma     : RBF bandwidth; if None, uses the median heuristic

    Returns
    -------
    mmd2 : float  (MMD², non-negative; 0 = identical distributions)
    """
    def featurise(trajs):
        # Per-trajectory (mean, std) across the time axis → [N, 7*2=14]
        mu  = trajs.mean(axis=-1)   # [N, 7]
        std = trajs.std(axis=-1)    # [N, 7]
        return np.concatenate([mu, std], axis=-1)   # [N, 14]

    X = featurise(real).astype(np.float64)       # [N_r, 14]
    Y = featurise(generated).astype(np.float64)  # [N_g, 14]

    # Median heuristic for sigma
    if sigma is None:
        all_pts = np.concatenate([X, Y], axis=0)
        diff    = all_pts[:, None, :] - all_pts[None, :, :]
        sq      = (diff ** 2).sum(axis=-1)
        sigma   = np.sqrt(np.median(sq) / 2.0 + 1e-8)

    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)

    n, m = len(X), len(Y)

    # Unbiased estimator: zero diagonal for same-sample terms
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (Kxx.sum() / (n * (n - 1))
            - 2.0 * Kxy.mean()
            + Kyy.sum() / (m * (m - 1)))

    return float(mmd2)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────

def plot_3d_trajectories(real: np.ndarray, generated: np.ndarray,
                         out_path: Path, n_show: int = 5):
    """3D NED scatter plot: real (blue) vs generated (orange) trajectories."""
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection="3d")

    n_real = min(n_show, len(real))
    n_gen  = min(n_show, len(generated))

    for i in range(n_real):
        N, E, D = real[i, 0, :], real[i, 1, :], real[i, 2, :]
        ax.plot(E, N, -D, alpha=0.7, linewidth=1.5,
                color=cm.Blues(0.4 + 0.5 * i / n_real),
                label="Real" if i == 0 else None)

    for i in range(n_gen):
        N, E, D = generated[i, 0, :], generated[i, 1, :], generated[i, 2, :]
        ax.plot(E, N, -D, alpha=0.7, linewidth=1.5, linestyle="--",
                color=cm.Oranges(0.4 + 0.5 * i / n_gen),
                label="Generated" if i == 0 else None)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("Real vs Generated Trajectories (NED)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved 3D plot → {out_path}")


def plot_feature_histograms(real: np.ndarray, generated: np.ndarray,
                            out_path: Path):
    """Per-feature histogram comparing real and generated distributions."""
    n_feats = real.shape[1]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(n_feats):
        ax  = axes[i]
        r_v = real[:, i, :].flatten()
        g_v = generated[:, i, :].flatten()

        # Clip to 1st–99th percentile of real data for readable axes
        lo, hi = np.percentile(r_v, 1), np.percentile(r_v, 99)
        bins   = np.linspace(lo, hi, 60)

        ax.hist(r_v, bins=bins, alpha=0.6, density=True,
                color="#1f77b4", label="Real")
        ax.hist(g_v, bins=bins, alpha=0.6, density=True,
                color="#ff7f0e", label="Generated")
        ax.set_title(FEATURE_LABELS[i])
        ax.legend(fontsize=8)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    fig.suptitle("Feature Distribution: Real vs Generated", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved histogram plot → {out_path}")


def plot_loss_curve(loss_csv: Path, out_path: Path):
    """Plot train vs val loss from the training log CSV."""
    if not loss_csv.exists():
        logger.warning(f"Loss CSV not found: {loss_csv} — skipping loss plot.")
        return

    df = pd.read_csv(loss_csv)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train loss", color="#1f77b4")
    ax.plot(df["epoch"], df["val_loss"],   label="Val loss",   color="#ff7f0e",
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Convergence")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved loss curve → {out_path}")


def plot_yaw_series(real: np.ndarray, generated: np.ndarray,
                    out_path: Path, n_show: int = 3):
    """Plot yaw angle over time for real and generated trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    n_real = min(n_show, len(real))
    n_gen  = min(n_show, len(generated))

    for i in range(n_real):
        axes[0].plot(real[i, 6, :], alpha=0.8, label=f"Real {i+1}")
    axes[0].set_title("Real Yaw Angle Over Time")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Yaw (rad)")
    axes[0].legend(fontsize=8)

    for i in range(n_gen):
        axes[1].plot(generated[i, 6, :], alpha=0.8, linestyle="--",
                     label=f"Gen {i+1}")
    axes[1].set_title("Generated Yaw Angle Over Time")
    axes[1].set_xlabel("Step")
    axes[1].legend(fontsize=8)

    fig.suptitle("Yaw Angle Time Series Comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved yaw series → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  REPORT PRINTER
# ──────────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------------------

def _row(label, real_val, gen_val, fmt=".4f", unit=""):
    r = f"{real_val:{fmt}}"
    g = f"{gen_val:{fmt}}"
    diff_pct = abs(gen_val - real_val) / (abs(real_val) + 1e-8) * 100
    flag = "[!]" if diff_pct > 50 else "[ok]"
    return f"  {flag} {label:<30} Real: {r:>12}{unit}  |  Gen: {g:>12}{unit}  |  D={diff_pct:.1f}%"


def print_report(real_m: dict, gen_m: dict, mmd2: float):
    """Print a formatted evaluation report to stdout."""
    sep = "-" * 80
    print()
    print(sep)
    print("  EVALUATION REPORT -- Kinematic-Guided Trajectory Diffusion Model")
    print(sep)
    print()
    print("  KINEMATIC METRICS")
    print(_row("Mean yaw rate",        real_m["mean_yaw_rate_rad"],   gen_m["mean_yaw_rate_rad"],   unit=" rad/step"))
    print(_row("Max yaw rate",         real_m["max_yaw_rate_rad"],    gen_m["max_yaw_rate_rad"],    unit=" rad/step"))
    print(_row("Mean speed",           real_m["mean_speed_ms"],       gen_m["mean_speed_ms"],       unit=" m/s"))
    print(_row("Velocity smoothness",  real_m["velocity_smoothness"], gen_m["velocity_smoothness"], unit=" m/s²"))
    print(_row("Kinematic cost J(s)",  real_m["kinematic_cost_J"],    gen_m["kinematic_cost_J"],    unit=""))
    print()
    print("  DISTRIBUTION METRIC")
    interp = "~0 (distributions match)" if abs(mmd2) < 0.05 else "> 0 (distributions differ)"
    print(f"  {'MMD2 (RBF kernel)':<30} {mmd2:.6f}  {interp}")
    print()
    print(sep)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(config: dict, generated_csv: str):
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load real trajectories ────────────────────────────────────────────────
    logger.info("Loading real trajectories for evaluation …")
    trajectories, mean_dt_s = parse_directory(
        config["data"]["data_dir"],
        min_mission_samples=config["data"]["min_mission_samples"],
    )
    dataset = TrajectoryDataset(
        trajectories,
        sequence_length=config["data"]["sequence_length"],
        stride=config["data"]["stride"],
        mean_dt_s=mean_dt_s,
    )

    # Denormalise real data: dataset.data is [N, 7, L] in normalised form
    real_norm = dataset.data                         # [N, 7, L]
    real_phys = dataset.denormalise(real_norm)       # [N, 7, L]  physical units
    real_np   = real_phys.numpy()                    # [N, 7, L]

    # ── Load generated trajectories ───────────────────────────────────────────
    logger.info(f"Loading generated trajectories from: {generated_csv}")
    df = pd.read_csv(generated_csv)
    n_samples   = df["sample_id"].max() + 1
    seq_len     = config["data"]["sequence_length"]
    n_features  = config["data"]["features"]
    feat_cols   = FEATURE_NAMES[:n_features]

    gen_np = np.zeros((n_samples, n_features, seq_len), dtype=np.float32)
    for sid in range(n_samples):
        sub = df[df["sample_id"] == sid].sort_values("step")
        for f_idx, col in enumerate(feat_cols):
            gen_np[sid, f_idx, :] = sub[col].values[:seq_len]

    logger.info(f"Real: {real_np.shape}  |  Generated: {gen_np.shape}")

    # ── Kinematic metrics ─────────────────────────────────────────────────────
    diff_cfg = config["diffusion"]
    real_km = kinematic_metrics(real_np, diff_cfg, mean_dt_s)
    gen_km  = kinematic_metrics(gen_np,  diff_cfg, mean_dt_s)

    # ── MMD ───────────────────────────────────────────────────────────────────
    mmd2 = compute_mmd(real_np, gen_np)

    # ── Print report ──────────────────────────────────────────────────────────
    print_report(real_km, gen_km, mmd2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_3d_trajectories(real_np, gen_np,
                         output_dir / "eval_3d_trajectories.png")
    plot_feature_histograms(real_np, gen_np,
                            output_dir / "eval_feature_histograms.png")
    plot_loss_curve(
        Path(config["data"]["checkpoint_dir"]) / "loss_history.csv",
        output_dir / "eval_loss_curve.png",
    )
    plot_yaw_series(real_np, gen_np,
                    output_dir / "eval_yaw_series.png")

    logger.info("Evaluation complete. Plots saved to outputs/")
    return {"real": real_km, "generated": gen_km, "mmd2": mmd2}


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the trajectory diffusion model"
    )
    parser.add_argument("--config",    type=str,
                        default="configs/default.yaml")
    parser.add_argument("--generated", type=str,
                        default="outputs_9feat/generated_trajectories.csv",
                        help="Path to the generated trajectories CSV")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    evaluate(cfg, args.generated)
