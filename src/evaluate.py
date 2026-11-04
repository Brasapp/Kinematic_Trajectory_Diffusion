"""
evaluate.py
===========
Compare generated trajectories with real AUTO flight.

Real data
---------
The logs are split by file exactly as in training (``data_parser.load_split``):

- **real-train** — windows from the logs the model was trained on
- **real-val**   — windows from held-out logs the model never saw

The real flight paths are the **GPS/EKF positions** from the logs (N, E, ground
course), shifted to start at the origin and rotated so the initial course is
north, the same frame as the generated paths.  Generated paths are rebuilt
from generated roll by dead reckoning (``sample.reconstruct_trajectory``); the
"dead-reckoning check" applies the same reconstruction to real roll and
measures how far it ends up from the real GPS path, i.e. how much the
generated paths can be trusted.

With ``data.mirror_roll`` the model generates a left/right symmetric
distribution, so the real sets are compared together with their mirror image.

Metrics
-------
- Attitude/altitude statistics (ranges, step sizes, left-turn share)
- Turn rate and heading change over the window
- Pitch-climb residual (the guidance coupling term)
- MMD² on per-window summary features, with **real-train vs real-val** as the
  reference: a generated MMD near that value is as close as two real sets are
- Memorisation check: nearest-neighbour distance of each generated window to
  the training windows, compared with the same distance for real-val windows

Usage
-----
    python -m src.evaluate --config configs/default.yaml \\
        [--generated outputs_v6/generated_trajectories.csv] \\
        [--checkpoint checkpoints_v6/epoch_0800.pt]
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.data_parser import COL, load_split, select_features, TrajectoryDataset
from src.sample import calibrate_guidance, reconstruct_trajectory

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

D = np.degrees


# ──────────────────────────────────────────────────────────────────────────────
# 1.  WINDOW SETS
# ──────────────────────────────────────────────────────────────────────────────
# A "set" is a dict of arrays [N, L]: N, E, alt, course, roll, pitch.

def real_windows(segments, seq_len: int, stride: int) -> dict:
    """Cut real full-column segments into windows in the generated frame."""
    out = {k: [] for k in ("N", "E", "alt", "course", "roll", "pitch", "speed")}
    for s in segments:
        for a in range(0, len(s) - seq_len + 1, stride):
            w = s[a:a + seq_len]
            course = np.unwrap(np.arctan2(w[:, COL["vE"]], w[:, COL["vN"]]))
            c0 = course[0]
            n = w[:, COL["N"]] - w[0, COL["N"]]
            e = w[:, COL["E"]] - w[0, COL["E"]]
            # rotate so the initial course points north (+N)
            out["N"].append(np.cos(c0) * n + np.sin(c0) * e)
            out["E"].append(-np.sin(c0) * n + np.cos(c0) * e)
            out["alt"].append(w[:, COL["alt"]] - w[0, COL["alt"]])
            out["course"].append(course - c0)
            out["roll"].append(w[:, COL["roll"]])
            out["pitch"].append(w[:, COL["pitch"]])
            out["speed"].append(np.hypot(w[:, COL["vN"]], w[:, COL["vE"]]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def mirror(d: dict) -> dict:
    """Left/right mirror image of a window set."""
    m = dict(d)
    for k in ("E", "course", "roll"):
        m[k] = -d[k]
    return m


def concat(*sets) -> dict:
    return {k: np.concatenate([s[k] for s in sets]) for k in sets[0] if k in sets[-1]}


def dead_reckon(d: dict, speed: float, dt: float) -> dict:
    """Rebuild N/E/course from roll exactly as generated paths are built."""
    out = {k: d[k] for k in ("alt", "roll", "pitch")}
    rec = np.stack([reconstruct_trajectory(r, p, a, speed, dt)
                    for r, p, a in zip(d["roll"], d["pitch"], d["alt"])])
    out["N"], out["E"], out["course"] = rec[:, :, 0], rec[:, :, 1], rec[:, :, 3]
    return out


def load_generated(csv_path: str, seq_len: int) -> dict:
    df = pd.read_csv(csv_path).sort_values(["sample_id", "step"])
    n = df["sample_id"].nunique()
    col = lambda c: df[c].to_numpy(np.float64).reshape(n, -1)[:, :seq_len]
    return {"N": col("N_m"), "E": col("E_m"), "alt": col("alt_m"),
            "course": col("yaw_rad"), "roll": col("roll_rad"), "pitch": col("pitch_rad")}


# ──────────────────────────────────────────────────────────────────────────────
# 2.  METRICS
# ──────────────────────────────────────────────────────────────────────────────

METRICS = [
    # key, label, unit
    ("roll_std", "Roll std", "deg"),
    ("roll_abs_med", "Roll |median|", "deg"),
    ("roll_p99", "Roll |p99|", "deg"),
    ("bank80_pct", "Time at |roll| > 80 deg", "%"),
    ("roll_step_med", "Roll step median", "deg"),
    ("roll_step_p99", "Roll step p99", "deg"),
    ("left_pct", "Left-turn windows", "%"),
    ("pitch_mean", "Pitch mean", "deg"),
    ("pitch_std", "Pitch std", "deg"),
    ("pitch_step_med", "Pitch step median", "deg"),
    ("alt_std", "Alt change std", "m"),
    ("alt_end_std", "Alt change at end, std", "m"),
    ("alt_step_p99", "Alt step p99", "m"),
    ("climb_resid", "Pitch-climb residual RMS", "m/s"),
    ("turn_med", "Turn rate |median|", "deg/s"),
    ("turn_p99", "Turn rate |p99|", "deg/s"),
    ("heading_change_med", "Heading change / window", "deg"),
]


def window_metrics(d: dict, dt: float, cal: dict) -> dict:
    roll, pitch, alt, course = d["roll"], d["pitch"], d["alt"], d["course"]
    dr = np.abs(np.diff(roll, axis=1))
    turn = np.abs(np.diff(course, axis=1)) / dt
    climb = np.diff(alt, axis=1) / dt
    resid = climb - (cal["climb_gain"] * np.sin(pitch[:, :-1]) + cal["climb_bias"])
    return {
        "roll_std": D(roll.std()),
        "roll_abs_med": D(np.median(np.abs(roll))),
        "roll_p99": D(np.percentile(np.abs(roll), 99)),
        "bank80_pct": 100.0 * (np.abs(roll) > np.radians(80)).mean(),
        "roll_step_med": D(np.median(dr)),
        "roll_step_p99": D(np.percentile(dr, 99)),
        "left_pct": 100.0 * (roll.mean(1) < 0).mean(),
        "pitch_mean": D(pitch.mean()),
        "pitch_std": D(pitch.std()),
        "pitch_step_med": D(np.median(np.abs(np.diff(pitch, axis=1)))),
        "alt_std": alt.std(),
        "alt_end_std": alt[:, -1].std(),
        "alt_step_p99": np.percentile(np.abs(np.diff(alt, axis=1)), 99),
        "climb_resid": float(np.sqrt((resid ** 2).mean())),
        "turn_med": D(np.median(turn)),
        "turn_p99": D(np.percentile(turn, 99)),
        "heading_change_med": D(np.median(np.abs(course[:, -1]))),
    }


def summary_features(d: dict, dt: float) -> np.ndarray:
    """Per-window summary vector used for MMD (sign-invariant where needed)."""
    roll, pitch, alt, course = d["roll"], d["pitch"], d["alt"], d["course"]
    return np.stack([
        roll.mean(1), roll.std(1), np.abs(np.diff(roll, axis=1)).mean(1),
        pitch.mean(1), pitch.std(1), np.abs(np.diff(pitch, axis=1)).mean(1),
        alt[:, -1], alt.std(1), np.abs(np.diff(alt, axis=1)).mean(1),
        course[:, -1], (np.abs(np.diff(course, axis=1)) / dt).mean(1),
    ], 1)


def mmd2(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased squared MMD with an RBF kernel of bandwidth ``sigma``."""
    def k(A, B):
        sq = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-sq / (2 * sigma ** 2))
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2 * Kxy.mean())


def nn_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For every row of A, the Euclidean distance to its nearest row of B."""
    a2 = (A ** 2).sum(1)[:, None]
    b2 = (B ** 2).sum(1)[None, :]
    return np.sqrt(np.maximum(a2 + b2 - 2 * A @ B.T, 0.0)).min(1)


def raw_vectors(d: dict, stats: dict) -> np.ndarray:
    """Flattened normalised (roll, pitch, alt) windows."""
    x = np.stack([d["roll"], d["pitch"], d["alt"]], 1)
    mean = np.asarray(stats["mean"])[None, :, None]
    std = np.asarray(stats["std"])[None, :, None]
    return ((x - mean) / std).reshape(len(x), -1)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_paths(real: dict, gen: dict, out_path: Path, n_show: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, d, title, color in ((axes[0], real, "Real (GPS, held-out logs)", "tab:blue"),
                                (axes[1], gen, "Generated (dead-reckoned)", "tab:orange")):
        for i in rng.choice(len(d["N"]), min(n_show, len(d["N"])), replace=False):
            ax.plot(d["E"][i], d["N"][i], color=color, alpha=0.7, lw=1.2)
        ax.plot(0, 0, "k.", ms=8)
        ax.set_title(title)
        ax.set_xlabel("East (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].set_ylabel("North (m), initial course = north")
    fig.suptitle("Top-down paths, 12.8 s windows")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_dead_reckoning(real: dict, rec: dict, out_path: Path, n_show: int = 6):
    fig, ax = plt.subplots(figsize=(8, 7))
    idx = np.linspace(0, len(real["N"]) - 1, n_show).astype(int)
    for j, i in enumerate(idx):
        c = plt.cm.tab10(j)
        ax.plot(real["E"][i], real["N"][i], color=c, lw=1.5, label="GPS" if j == 0 else None)
        ax.plot(rec["E"][i], rec["N"][i], color=c, lw=1.2, ls="--",
                label="Rebuilt from roll" if j == 0 else None)
    ax.plot(0, 0, "k.", ms=8)
    ax.set_title("Dead-reckoning check (held-out logs)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_histograms(real: dict, gen: dict, dt: float, out_path: Path):
    panels = [
        ("Roll (deg)", D(real["roll"]), D(gen["roll"])),
        ("Pitch (deg)", D(real["pitch"]), D(gen["pitch"])),
        ("Alt change since window start (m)", real["alt"], gen["alt"]),
        ("Roll step (deg)", D(np.diff(real["roll"], axis=1)), D(np.diff(gen["roll"], axis=1))),
        ("Turn rate (deg/s)", D(np.diff(real["course"], axis=1)) / dt, D(np.diff(gen["course"], axis=1)) / dt),
        ("Heading change per window (deg)", D(real["course"][:, -1]), D(gen["course"][:, -1])),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, r, g) in zip(axes.flat, panels):
        r, g = np.ravel(r), np.ravel(g)
        lo, hi = np.percentile(np.concatenate([r, g]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(r, bins=bins, density=True, alpha=0.55, color="tab:blue", label="Real (all logs, mirrored)")
        ax.hist(g, bins=bins, density=True, alpha=0.55, color="tab:orange", label="Generated")
        ax.set_title(title)
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_loss_curve(loss_csv: Path, out_path: Path):
    if not loss_csv.exists():
        logger.warning(f"Loss CSV not found: {loss_csv} — skipping loss plot.")
        return
    df = pd.read_csv(loss_csv)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train loss")
    ax.plot(df["epoch"], df["val_loss"], label="Val loss (held-out logs)", ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_series(real: dict, gen: dict, out_path: Path, n_show: int = 4):
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    t = np.arange(real["roll"].shape[1])
    for col, (d, name) in enumerate(((real, "Real (held-out)"), (gen, "Generated"))):
        idx = np.linspace(0, len(d["roll"]) - 1, n_show).astype(int)
        for i in idx:
            axes[0, col].plot(t, D(d["roll"][i]), lw=1)
            axes[1, col].plot(t, D(d["pitch"][i]), lw=1)
            axes[2, col].plot(t, d["alt"][i], lw=1)
        axes[0, col].set_title(name)
    for r, lab in enumerate(("Roll (deg)", "Pitch (deg)", "Alt change (m)")):
        axes[r, 0].set_ylabel(lab)
    axes[2, 0].set_xlabel("Step (0.1 s)")
    axes[2, 1].set_xlabel("Step (0.1 s)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(config: dict, generated_csv: str, checkpoint: str = None) -> dict:
    out_dir = Path(config["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seq, stride = config["data"]["sequence_length"], config["data"]["stride"]

    # ── Real data, split by log exactly as in training ───────────────────────
    split = load_split(config)
    dt = split["dt_s"]
    train_ds = TrajectoryDataset.from_config(
        select_features(split["train"], split["feat_idx"]), config["data"], dt)
    stats, cal, speed = train_ds.stats, calibrate_guidance(train_ds), split["mean_speed_ms"]
    if checkpoint:
        ck = torch.load(checkpoint, map_location="cpu")
        if "norm" in ck:
            stats, cal, speed = ck["norm"], ck["guidance_calibration"], ck["mean_speed_ms"]
            if ck.get("val_files") and ck["val_files"] != split["val_files"]:
                logger.warning("Checkpoint was trained with a different val split than the current config.")

    tr_raw = real_windows(split["train"], seq, stride)
    va_raw = real_windows(split["val"], seq, stride)
    sym = config["data"].get("mirror_roll", False)
    tr = concat(tr_raw, mirror(tr_raw)) if sym else tr_raw
    va = concat(va_raw, mirror(va_raw)) if sym else va_raw
    # Rebuild generated paths here so they use exactly the same dead reckoning
    # as the real-data check below (independent of how the CSV was written).
    gen = dead_reckon(load_generated(generated_csv, seq), speed, dt)
    logger.info(f"Windows: real-train {len(tr_raw['roll'])}, real-val {len(va_raw['roll'])}"
                f"{' (x2 mirrored)' if sym else ''}, generated {len(gen['roll'])}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    m_tr, m_va, m_gen = (window_metrics(d, dt, cal) for d in (tr, va, gen))
    m_va_raw = window_metrics(va_raw, dt, cal)

    # Dead-reckoning check on real held-out windows
    rec = dead_reckon(va_raw, speed, dt)
    end_err = np.hypot(rec["N"][:, -1] - va_raw["N"][:, -1], rec["E"][:, -1] - va_raw["E"][:, -1])
    path_len = (va_raw["speed"][:, :-1] * dt).sum(1)
    hdg_err = D(np.abs(rec["course"][:, -1] - va_raw["course"][:, -1]))

    # MMD on summary features, standardised with real-train
    F_tr, F_va, F_gen = (summary_features(d, dt) for d in (tr, va, gen))
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-9
    F_tr, F_va, F_gen = ((F - mu) / sd for F in (F_tr, F_va, F_gen))
    allF = np.concatenate([F_tr, F_va])
    sigma = np.sqrt(np.median(((allF[:, None] - allF[None]) ** 2).sum(-1)) / 2)
    mmd_ref = mmd2(F_tr, F_va, sigma)
    mmd_gen_va = mmd2(F_gen, F_va, sigma)
    mmd_gen_tr = mmd2(F_gen, F_tr, sigma)

    # Memorisation: nearest training window
    R_tr, R_va, R_gen = (raw_vectors(d, stats) for d in (tr, va, gen))
    nn_va, nn_gen = nn_distance(R_va, R_tr), nn_distance(R_gen, R_tr)
    copy_thr = np.percentile(nn_va, 5)
    copied_pct = 100.0 * (nn_gen < copy_thr).mean()

    # ── Report ───────────────────────────────────────────────────────────────
    lines = []
    sep = "-" * 92
    lines += [sep, "  EVALUATION REPORT — generated vs real AUTO flight", sep,
              f"  real-train: {len(split['train_files'])} logs, {len(tr_raw['roll'])} windows | "
              f"real-val: {len(split['val_files'])} logs, {len(va_raw['roll'])} windows | "
              f"generated: {len(gen['roll'])}",
              f"  real sets {'include their mirror image' if sym else 'are not mirrored'}; "
              f"dt = {dt:.2f} s; window = {seq * dt:.1f} s", ""]
    lines.append(f"  {'metric':<28}{'unit':>6}{'real-train':>12}{'real-val':>12}{'generated':>12}   gen vs real")
    for key, label, unit in METRICS:
        a, b, g = m_tr[key], m_va[key], m_gen[key]
        lo, hi = min(a, b), max(a, b)
        spread = max(hi - lo, 0.15 * max(abs(a), abs(b)), 1e-9)
        flag = "ok" if lo - spread <= g <= hi + spread else ("HIGH" if g > hi else "LOW")
        lines.append(f"  {label:<28}{unit:>6}{a:>12.2f}{b:>12.2f}{g:>12.2f}   {flag}")
    lines += ["",
              f"  Left-turn windows in raw held-out logs (not mirrored): {m_va_raw['left_pct']:.0f}%",
              "",
              "  DISTRIBUTION (MMD^2 on 11 per-window summary features, lower = closer)",
              f"    real-train vs real-val (reference) : {mmd_ref:.4f}",
              f"    generated  vs real-val             : {mmd_gen_va:.4f}",
              f"    generated  vs real-train           : {mmd_gen_tr:.4f}",
              "",
              "  MEMORISATION (distance to the nearest training window, normalised units)",
              f"    real-val  median : {np.median(nn_va):.2f}",
              f"    generated median : {np.median(nn_gen):.2f}",
              f"    generated closer than the closest 5% of real-val: {copied_pct:.0f}% (about 5% expected)",
              "",
              "  DEAD-RECKONING CHECK (real held-out roll -> path, vs GPS)",
              f"    end-point error   median {np.median(end_err):.0f} m (p90 {np.percentile(end_err, 90):.0f} m) "
              f"over a median {np.median(path_len):.0f} m path",
              f"    heading error     median {np.median(hdg_err):.0f} deg (p90 {np.percentile(hdg_err, 90):.0f} deg)",
              f"    -> generated paths are indicative only; roll/pitch/alt are the direct comparison",
              sep]
    report = "\n".join(lines)
    print(report)
    (out_dir / "eval_report.txt").write_text(report + "\n", encoding="utf-8")

    result = {
        "real_train": m_tr, "real_val": m_va, "generated": m_gen,
        "mmd2": {"train_vs_val": mmd_ref, "gen_vs_val": mmd_gen_va, "gen_vs_train": mmd_gen_tr},
        "memorisation": {"nn_val_median": float(np.median(nn_va)),
                         "nn_gen_median": float(np.median(nn_gen)), "copied_pct": copied_pct},
        "dead_reckoning": {"end_err_median_m": float(np.median(end_err)),
                           "heading_err_median_deg": float(np.median(hdg_err))},
        "val_files": split["val_files"],
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_paths(va_raw, gen, out_dir / "eval_paths.png")
    plot_dead_reckoning(va_raw, rec, out_dir / "eval_dead_reckoning.png")
    plot_histograms(concat(tr, va), gen, dt, out_dir / "eval_histograms.png")
    plot_series(va_raw, gen, out_dir / "eval_series.png")
    plot_loss_curve(Path(config["data"]["checkpoint_dir"]) / "loss_history.csv",
                    out_dir / "eval_loss_curve.png")
    logger.info(f"Report and plots saved to {out_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trajectory diffusion model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--generated", type=str, default=None,
                        help="Generated CSV (default: sampling.output_csv)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint whose stored stats/calibration to use (optional)")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    evaluate(cfg, args.generated or cfg["sampling"]["output_csv"], args.checkpoint)
