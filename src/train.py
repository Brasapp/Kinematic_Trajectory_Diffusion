"""
train.py
========
Training loop for the Kinematic-Guided Trajectory Diffusion Model.

Algorithm (Ho et al. 2020, Algorithm 1 — simplified objective)
--------------------------------------------------------------
For each training step:

    1.  Sample a clean trajectory  x_0 ~ q(x_0)
    2.  Sample a random timestep   t   ~ Uniform{1, ..., T}
    3.  Sample Gaussian noise      ε   ~ N(0, I)
    4.  Compute noisy trajectory   x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε
    5.  Predict noise              ε̂_θ = model(x_t, t)
    6.  Optimise loss              L = ||ε - ε̂_θ||²

Usage
-----
    python -m src.train --config configs/default.yaml

or simply::

    python src/train.py
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml

from src.data_parser import load_split, select_features, TrajectoryDataset
from src.diffusion_math import CosineNoiseScheduler
from src.sample import calibrate_guidance
from src.unet_1d import build_model, core_module

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, epoch, loss, path, extra=None):
    """Save weights (of the bare UNet1D) plus ``extra`` metadata: normalisation
    stats, guidance calibration and the train/val log split."""
    torch.save(
        {
            "epoch":     epoch,
            "loss":      loss,
            "model":     core_module(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            **(extra or {}),
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    core_module(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["loss"]


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train(config: dict):
    # ── Reproducibility ──────────────────────────────────────────────────────
    seed = config["training"]["seed"]
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Paths ────────────────────────────────────────────────────────────────
    ckpt_dir = Path(config["data"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data: split by log file so no flight is in both train and val ────────
    logger.info(f"Loading trajectories from: {config['data']['data_dir']}")
    split = load_split(config)
    dataset = TrajectoryDataset.from_config(
        select_features(split["train"], split["feat_idx"]), config["data"], split["dt_s"])
    val_ds = TrajectoryDataset.from_config(
        select_features(split["val"], split["feat_idx"]), config["data"], split["dt_s"],
        stats=dataset.stats)
    logger.info(f"Validation logs: {', '.join(split['val_files'])}")

    train_loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Dataset: {len(dataset)} train | {len(val_ds)} val windows "
                f"({len(split['train_files'])} / {len(split['val_files'])} logs)")

    # Everything sampling needs, so it does not have to re-parse the logs.
    ckpt_extra = {
        "norm": dataset.stats,
        "guidance_calibration": calibrate_guidance(dataset),
        "mean_speed_ms": split["mean_speed_ms"],
        "train_files": split["train_files"],
        "val_files": split["val_files"],
        "feature_names": split["feature_names"],
    }

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{type(model).__name__} parameters: {n_params:,}")

    # ── Noise scheduler ──────────────────────────────────────────────────────
    scheduler = CosineNoiseScheduler(
        T=config["diffusion"]["T"],
        s=config["diffusion"]["s"],
        device=str(device),
    )

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    # ── Resume from checkpoint if available ──────────────────────────────────
    start_epoch = 0
    latest_ckpt = sorted(ckpt_dir.glob("epoch_*.pt"))
    if latest_ckpt:
        start_epoch, _ = load_checkpoint(model, optimizer, latest_ckpt[-1], device)
        start_epoch += 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # ── Loss log ─────────────────────────────────────────────────────────────
    loss_log_path = ckpt_dir / "loss_history.csv"
    if not loss_log_path.exists():
        loss_log_path.write_text("epoch,train_loss,val_loss,elapsed_s\n")

    # ── Training epochs ───────────────────────────────────────────────────────
    epochs        = config["training"]["epochs"]
    ckpt_every    = config["training"]["checkpoint_every"]
    grad_clip     = config["training"]["grad_clip"]
    lambda_smooth = config["training"].get("lambda_smooth", 0.1)
    T             = config["diffusion"]["T"]

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x0 = dataset.mirror_batch(batch.to(device))   # [B, F, L], random left/right mirror
            B  = x0.shape[0]

            # Sample random diffusion timestep t ~ Uniform{0, T-1}
            t = torch.randint(0, T, (B,), device=device)

            # Sample noise ε ~ N(0, I)
            noise = torch.randn_like(x0)

            # Forward process: x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε
            x_t, noise = scheduler.q_sample(x0, t, noise)

            # Predict noise:  ε̂_θ(x_t, t)
            eps_pred = model(x_t, t)

            # Simplified DDPM loss:  L = ||ε - ε̂_θ||²
            loss_mse = F.mse_loss(eps_pred, noise)
            
            # Smoothness constraint on predicted x0 (roll and pitch)
            # x0_pred has shape [B, 3, L] -> index 0: roll, 1: pitch, 2: alt
            x0_pred = scheduler.predict_x0_from_eps(x_t, t, eps_pred)

            # Compute 2nd derivatives (Laplacian)
            # grad2 = x[i+1] - 2x[i] + x[i-1]
            roll_grad2 = x0_pred[:, 0, 2:] - 2 * x0_pred[:, 0, 1:-1] + x0_pred[:, 0, :-2]
            pitch_grad2 = x0_pred[:, 1, 2:] - 2 * x0_pred[:, 1, 1:-1] + x0_pred[:, 1, :-2]

            # Weight each sample by alpha_bar_t: x0_pred is only meaningful at low
            # noise. Unweighted, the 1/sqrt(alpha_bar_t) factor in x0_pred blows the
            # gradient up ~1e4x at high t and biases the model there.
            w_t = scheduler.alpha_bars[t]                                   # [B]
            smooth_per_sample = (roll_grad2 ** 2).mean(dim=1) + (pitch_grad2 ** 2).mean(dim=1)
            loss_smooth = (w_t * smooth_per_sample).mean()
            loss = loss_mse + lambda_smooth * loss_smooth

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        # Fixed timesteps and noise every epoch (and 4 passes) so the curve is
        # comparable between epochs instead of dominated by sampling noise.
        model.eval()
        val_loss, n_val = 0.0, 0
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for _ in range(4):
                for batch in val_loader:
                    x0 = val_ds.mirror_batch(batch.to(device))
                    B  = x0.shape[0]
                    t  = torch.randint(0, T, (B,), generator=gen).to(device)
                    noise = torch.randn(x0.shape, generator=gen).to(device)
                    x_t, noise = scheduler.q_sample(x0, t, noise)
                    eps_pred   = model(x_t, t)
                    val_loss  += F.mse_loss(eps_pred, noise).item() * B
                    n_val     += B
        val_loss /= n_val

        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch+1:04d}/{epochs}] "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"({elapsed:.1f}s)"
        )

        # Append to CSV log
        with open(loss_log_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.8f},{val_loss:.8f},{elapsed:.2f}\n")

        # ── Save checkpoint ───────────────────────────────────────────────────
        if (epoch + 1) % ckpt_every == 0 or epoch == epochs - 1:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path, ckpt_extra)

    logger.info("Training complete.")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train trajectory diffusion model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
