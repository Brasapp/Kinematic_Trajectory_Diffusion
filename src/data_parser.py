"""
data_parser.py
==============
Parses ArduPilot DataFlash binary (.bin) logs and produces normalised
PyTorch trajectory tensors for diffusion model training.

Pipeline
--------
1.  Read each ``.bin`` log once with ``pymavlink``.
2.  Take the state from a single, self-consistent source:

    - ``XKF1`` (EKF3, core 0)  → PN/PE/PD position, VN/VE/VD velocity,
      roll/pitch/yaw.  Used by ArduPilot 4.x.
    - ``NKF1`` (EKF2, core 0)  → same fields, for older firmware.
    - Fallback when neither exists: ``GPS`` (the receiver with the most 3D
      fixes, 3D fixes only) for
      position/velocity and ``ATT`` (or ``AHR2``) for attitude.

    Position streams are never mixed (the old parser interleaved GPS and POS,
    which created a sawtooth in altitude and an absurd mean speed).
3.  Resample everything onto a uniform time grid (``resample_hz``).
    Angles are interpolated through sin/cos so ±180° wraps (inverted flight,
    north crossings) stay correct.
4.  Keep only samples that are in one of the selected flight modes, flying
    (horizontal speed ≥ ``min_speed_ms``) and densely logged (source sample
    spacing ≤ ``max_gap_s``).  Contiguous runs become
    segments; runs shorter than ``min_mission_samples`` are dropped.
5.  ``TrajectoryDataset`` z-score normalises and cuts sliding windows.

Every segment carries all columns in ``FULL_COLUMNS``; callers choose the
model features with ``feature_names`` (default ``roll, pitch, alt``).

Run ``python -m src.data_parser --data_dir data/`` to print per-log
diagnostics (speed, bank angle, altitude, segment lengths).
"""

import argparse
import hashlib
import logging
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from pymavlink.DFReader import DFReader_binary
except ImportError as exc:
    raise ImportError("pymavlink is required: pip install pymavlink") from exc

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EARTH_RADIUS_M = 6_371_000.0

# ArduPlane mode numbers (ModeNum in the MODE message)
PLANE_MODES = {
    "MANUAL": 0, "CIRCLE": 1, "STABILIZE": 2, "TRAINING": 3, "ACRO": 4,
    "FBWA": 5, "FBWB": 6, "CRUISE": 7, "AUTOTUNE": 8, "AUTO": 10, "RTL": 11,
    "LOITER": 12, "TAKEOFF": 13, "AVOID_ADSB": 14, "GUIDED": 15,
    "QSTABILIZE": 17, "QHOVER": 18, "QLOITER": 19, "QLAND": 20, "QRTL": 21,
    "QAUTOTUNE": 22, "QACRO": 23, "THERMAL": 24, "LOITER_ALT_QLAND": 25,
}
COPTER_MODES = {
    "STABILIZE": 0, "ACRO": 1, "ALT_HOLD": 2, "AUTO": 3, "GUIDED": 4,
    "LOITER": 5, "RTL": 6, "CIRCLE": 7, "LAND": 9, "POSHOLD": 16,
}

# Columns of every parsed segment, in order.
FULL_COLUMNS = [
    "N", "E", "alt",        # m, local frame; alt is positive up, relative to EKF origin / first fix
    "vN", "vE", "vD",       # m/s
    "roll", "pitch", "yaw", # rad; roll and yaw wrapped to (-pi, pi]
    "airspeed",             # m/s (CTUN.As, NaN if not logged)
]
COL = {name: i for i, name in enumerate(FULL_COLUMNS)}
DEFAULT_FEATURES = ("roll", "pitch", "alt")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SMALL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def latlon_to_ned(lat, lon, alt, lat0, lon0, alt0) -> np.ndarray:
    r"""Equirectangular WGS-84 → local NED (metres). Valid for areas < ~50 km.

    .. math::
        N = R(\phi-\phi_0),\quad E = R\cos\phi_0(\lambda-\lambda_0),\quad D = -(h-h_0)
    """
    phi, phi0 = np.deg2rad(lat), math.radians(lat0)
    lam, lam0 = np.deg2rad(lon), math.radians(lon0)
    N = EARTH_RADIUS_M * (phi - phi0)
    E = EARTH_RADIUS_M * math.cos(phi0) * (lam - lam0)
    D = -(np.asarray(alt) - alt0)
    return np.stack([N, E, D], axis=-1)


def wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def _sorted_unique(t: np.ndarray, *cols: np.ndarray):
    """Sort by time and drop duplicate timestamps."""
    if len(t) == 0:
        return (t, *cols)
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return (t[keep], *(c[order][keep] for c in cols))


def _interp(t_src, y_src, t_dst) -> np.ndarray:
    """Linear interpolation; NaN outside the source range."""
    if len(t_src) < 2:
        return np.full(len(t_dst), np.nan)
    return np.interp(t_dst, t_src, y_src, left=np.nan, right=np.nan)


def _interp_angle(t_src, a_src, t_dst) -> np.ndarray:
    """Interpolate an angle (rad) through sin/cos so wraps are handled."""
    s = _interp(t_src, np.sin(a_src), t_dst)
    c = _interp(t_src, np.cos(a_src), t_dst)
    return np.arctan2(s, c)


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  RAW EXTRACTION (single pass)
# ──────────────────────────────────────────────────────────────────────────────

_EKF_FIELDS = ("PN", "PE", "PD", "VN", "VE", "VD", "Roll", "Pitch", "Yaw")


def _read_log(path: Path) -> dict:
    """Read every message type we need in one pass. Returns raw numpy arrays."""
    log = DFReader_binary(str(path), zero_time_base=True)

    ekf = {"XKF1": {k: [] for k in ("t",) + _EKF_FIELDS},
           "NKF1": {k: [] for k in ("t",) + _EKF_FIELDS}}
    gps = {k: [] for k in ("t", "inst", "lat", "lng", "alt", "spd", "crs", "vz")}
    att = {k: [] for k in ("t", "roll", "pitch", "yaw")}
    ahr2 = {k: [] for k in ("t", "roll", "pitch", "yaw")}
    ctun = {"t": [], "as": []}
    modes: List[Tuple[float, int]] = []
    firmware = None

    want = {"XKF1", "NKF1", "GPS", "ATT", "AHR2", "CTUN", "MODE", "MSG"}
    while True:
        m = log.recv_match(type=want, blocking=False)
        if m is None:
            break
        typ = m.get_type()
        t = float(getattr(m, "TimeUS", 0))

        if typ in ekf:
            if int(getattr(m, "C", 0)) != 0:        # core 0 only
                continue
            d = ekf[typ]
            d["t"].append(t)
            for k in _EKF_FIELDS:
                d[k].append(float(getattr(m, k, np.nan)))
        elif typ == "GPS":
            if int(getattr(m, "Status", 3)) < 3:
                continue                             # need a 3D fix
            gps["t"].append(t)
            gps["inst"].append(int(getattr(m, "I", getattr(m, "Instance", 0))))
            gps["lat"].append(m.Lat); gps["lng"].append(m.Lng); gps["alt"].append(m.Alt)
            gps["spd"].append(getattr(m, "Spd", np.nan))
            gps["crs"].append(getattr(m, "GCrs", np.nan))
            gps["vz"].append(getattr(m, "VZ", np.nan))
        elif typ in ("ATT", "AHR2"):
            d = att if typ == "ATT" else ahr2
            d["t"].append(t); d["roll"].append(m.Roll)
            d["pitch"].append(m.Pitch); d["yaw"].append(m.Yaw)
        elif typ == "CTUN":
            ctun["t"].append(t); ctun["as"].append(float(getattr(m, "As", np.nan)))
        elif typ == "MODE":
            num = getattr(m, "ModeNum", getattr(m, "Mode", None))
            if isinstance(num, (int, float)):
                modes.append((t, int(num)))
        elif typ == "MSG" and firmware is None:
            text = str(getattr(m, "Message", ""))
            if text.startswith("Ardu"):
                firmware = text

    def arr(d):
        return {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}

    # Keep one receiver only: the one with the most 3D fixes (the GPS is not
    # always wired to the first port).
    g = arr(gps)
    if len(g["t"]):
        insts, counts = np.unique(g["inst"], return_counts=True)
        sel = g["inst"] == insts[np.argmax(counts)]
        g = {k: v[sel] for k, v in g.items()}

    return {
        "XKF1": arr(ekf["XKF1"]), "NKF1": arr(ekf["NKF1"]),
        "GPS": g, "ATT": arr(att), "AHR2": arr(ahr2), "CTUN": arr(ctun),
        "modes": sorted(modes), "firmware": firmware,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  STATE ON A UNIFORM GRID
# ──────────────────────────────────────────────────────────────────────────────

def _state_on_grid(raw: dict, resample_hz: float):
    """Build the full state [T, len(FULL_COLUMNS)] on a uniform time grid.

    Returns (t_grid_us, state, source_name, gap_s) where ``gap_s`` is the
    spacing of the real source samples around each grid point.
    """
    step_us = 1e6 / resample_hz
    ekf_name = "XKF1" if len(raw["XKF1"]["t"]) > 10 else ("NKF1" if len(raw["NKF1"]["t"]) > 10 else None)

    if ekf_name is not None:
        e = raw[ekf_name]
        t, *vals = _sorted_unique(e["t"], *(e[k] for k in _EKF_FIELDS))
        e = dict(zip(_EKF_FIELDS, vals))
        grid = np.arange(t[0], t[-1], step_us)
        N, E = _interp(t, e["PN"], grid), _interp(t, e["PE"], grid)
        alt = -_interp(t, e["PD"], grid)
        vN, vE, vD = (_interp(t, e[k], grid) for k in ("VN", "VE", "VD"))
        roll = _interp_angle(t, np.deg2rad(e["Roll"]), grid)
        pitch = _interp(t, np.deg2rad(e["Pitch"]), grid)
        yaw = _interp_angle(t, np.deg2rad(e["Yaw"]), grid)
        source = ekf_name
    else:
        g = raw["GPS"]
        if len(g["t"]) < 10:
            raise ValueError("no EKF (XKF1/NKF1) and too few GPS fixes")
        t, lat, lng, galt, spd, crs, vz = _sorted_unique(
            g["t"], g["lat"], g["lng"], g["alt"], g["spd"], g["crs"], g["vz"])
        grid = np.arange(t[0], t[-1], step_us)
        ned = latlon_to_ned(lat, lng, galt, lat[0], lng[0], galt[0])
        N, E = _interp(t, ned[:, 0], grid), _interp(t, ned[:, 1], grid)
        alt = -_interp(t, ned[:, 2], grid)
        crs_r = np.deg2rad(crs)
        vN = _interp(t, spd * np.cos(crs_r), grid)
        vE = _interp(t, spd * np.sin(crs_r), grid)
        vD = _interp(t, vz, grid)
        a = raw["ATT"] if len(raw["ATT"]["t"]) > 10 else raw["AHR2"]
        at, ar, ap, ay = _sorted_unique(a["t"], a["roll"], a["pitch"], a["yaw"])
        roll = _interp_angle(at, np.deg2rad(ar), grid)
        pitch = _interp(at, np.deg2rad(ap), grid)
        yaw = _interp_angle(at, np.deg2rad(ay), grid)
        source = "GPS+ATT"

    c = raw["CTUN"]
    if len(c["t"]) > 1:
        ct, cas = _sorted_unique(c["t"], c["as"])
        airspeed = _interp(ct, cas, grid)
    else:
        airspeed = np.full(len(grid), np.nan)

    state = np.stack([N, E, alt, vN, vE, vD, roll, pitch, yaw, airspeed], axis=-1)

    # Seconds since the last real source sample at each grid point. Used to drop
    # stretches where the log was too sparse to resample honestly.
    src_t = t
    idx = np.clip(np.searchsorted(src_t, grid, side="right") - 1, 0, len(src_t) - 1)
    nxt = np.clip(idx + 1, 0, len(src_t) - 1)
    gap_s = (src_t[nxt] - src_t[idx]) / 1e6
    return grid, state, source, gap_s


def _mode_on_grid(modes: List[Tuple[float, int]], grid: np.ndarray) -> np.ndarray:
    """Flight-mode number at each grid time (-1 before the first MODE message)."""
    out = np.full(len(grid), -1, dtype=int)
    for ts, num in modes:
        out[grid >= ts] = num
    return out


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) index pairs of contiguous True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _resolve_modes(mode_names: Sequence[str], firmware: Optional[str]) -> set:
    table = COPTER_MODES if firmware and "Copter" in firmware else PLANE_MODES
    nums = set()
    for name in mode_names:
        key = str(name).upper()
        if key not in table:
            raise ValueError(f"Unknown flight mode '{name}' for {firmware or 'ArduPlane'}")
        nums.add(table[key])
    return nums


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def parse_bin_file_full(
    path: str,
    min_mission_samples: int = 250,
    resample_hz: float = 10.0,
    modes: Sequence[str] = ("AUTO",),
    min_speed_ms: float = 10.0,
    max_gap_s: float = 0.3,
    gps_timeout_s: float = 1.0,
    max_pos_jump_m: float = 3.0,
) -> Tuple[List[np.ndarray], dict]:
    """Parse one log into full-state segments.

    Returns
    -------
    segments : list of ndarray[T_i, len(FULL_COLUMNS)]
    info     : dict with firmware, source, per-segment start times, etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    raw = _read_log(path)
    grid, state, source, gap_s = _state_on_grid(raw, resample_hz)

    if raw["modes"]:
        wanted = _resolve_modes(modes, raw["firmware"])
        in_mode = np.isin(_mode_on_grid(raw["modes"], grid), list(wanted))
    else:
        logger.warning(f"{path.name}: no MODE messages — using all flying samples.")
        in_mode = np.ones(len(grid), dtype=bool)

    speed_h = np.hypot(state[:, COL["vN"]], state[:, COL["vE"]])
    core_ok = np.isfinite(state[:, :COL["airspeed"]]).all(axis=1)
    dense = gap_s <= max_gap_s

    # GPS quality: without a recent 3D fix the EKF position is dead-reckoning
    # and can drift by hundreds of metres.
    gps_t = np.sort(raw["GPS"]["t"])
    if len(gps_t) == 0:
        raise ValueError("no GPS 3D fix in log (EKF position is unaided)")
    j = np.clip(np.searchsorted(gps_t, grid, side="right") - 1, 0, len(gps_t) - 1)
    gps_ok = (grid >= gps_t[0]) & ((grid - gps_t[j]) <= gps_timeout_s * 1e6)

    # EKF resets / glitches: position step not explained by velocity.
    dt = 1.0 / resample_hz
    pos = np.stack([state[:, COL["N"]], state[:, COL["E"]], -state[:, COL["alt"]]], axis=1)
    vel = state[:, COL["vN"]:COL["vD"] + 1]
    step_err = np.linalg.norm(np.diff(pos, axis=0) - 0.5 * (vel[1:] + vel[:-1]) * dt, axis=1)
    glitch = np.zeros(len(grid), dtype=bool)
    bad = np.flatnonzero(~(step_err <= max_pos_jump_m))  # also catches NaN
    glitch[bad] = True
    glitch[bad + 1] = True

    keep = in_mode & (speed_h >= min_speed_ms) & core_ok & dense & gps_ok & ~glitch

    segments, starts = [], []
    for a, b in _runs(keep):
        if b - a >= min_mission_samples:
            segments.append(state[a:b].astype(np.float32))
            starts.append(grid[a] / 1e6)

    info = {
        "file": path.name, "firmware": raw["firmware"], "source": source,
        "dt_s": 1.0 / resample_hz, "segment_start_s": starts,
        "flying_s": float(((speed_h >= min_speed_ms) & core_ok).sum()) / resample_hz,
        "in_mode_flying_s": float(keep.sum()) / resample_hz,
        "sparse_flying_s": float(((speed_h >= min_speed_ms) & ~dense).sum()) / resample_hz,
        "bad_gps_or_ekf_s": float(((speed_h >= min_speed_ms) & (~gps_ok | glitch)).sum()) / resample_hz,
    }
    return segments, info


def _select(segments: List[np.ndarray], feature_names: Sequence[str]) -> List[np.ndarray]:
    idx = [COL[f] for f in feature_names]
    return [np.ascontiguousarray(s[:, idx]) for s in segments]


def _mean_speed(segments: List[np.ndarray]) -> float:
    """Sample-weighted mean 3D ground speed over all segments (m/s)."""
    if not segments:
        return float("nan")
    v = np.concatenate([s[:, COL["vN"]:COL["vD"] + 1] for s in segments])
    return float(np.linalg.norm(v, axis=1).mean())


def parse_bin_file(
    path: str,
    min_mission_samples: int = 250,
    resample_hz: float = 10.0,
    modes: Sequence[str] = ("AUTO",),
    min_speed_ms: float = 10.0,
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    max_gap_s: float = 0.3,
) -> tuple:
    """Parse one log. Returns (trajectories, dt_s, mean_speed_ms)."""
    segments, info = parse_bin_file_full(path, min_mission_samples, resample_hz, modes, min_speed_ms, max_gap_s)
    if not segments:
        raise ValueError(
            f"No {'/'.join(modes)} flying segments with >= {min_mission_samples} samples in {Path(path).name}."
        )
    return _select(segments, feature_names), info["dt_s"], _mean_speed(segments)


def parse_directory(
    data_dir: str,
    min_mission_samples: int = 250,
    resample_hz: float = 10.0,
    modes: Sequence[str] = ("AUTO",),
    min_speed_ms: float = 10.0,
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    return_meta: bool = False,
    max_gap_s: float = 0.3,
    gps_timeout_s: float = 1.0,
    max_pos_jump_m: float = 3.0,
    use_cache: bool = True,
) -> tuple:
    """Parse every ``.bin`` in ``data_dir`` (byte-identical duplicates skipped).

    Parsed segments are cached in ``<data_dir>/.cache/`` keyed on the parser
    settings, the parser source and each log's name/size/mtime, so repeated
    runs skip the slow log decoding.  Pass ``feature_names=FULL_COLUMNS`` to get
    every column (N, E, velocities, yaw, ... -- used by evaluation).

    Returns
    -------
    trajectories  : list of ndarray[T_i, len(feature_names)]
    dt_s          : float — uniform sample interval (1 / resample_hz)
    mean_speed_ms : float — sample-weighted mean 3D ground speed
    meta          : list of dict (only if ``return_meta``) — one per trajectory,
                    with the source file name, so splits can be done per log.
    """
    data_dir = Path(data_dir)
    bin_files = sorted(data_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {data_dir}")

    params = (min_mission_samples, resample_hz, tuple(modes), min_speed_ms,
              max_gap_s, gps_timeout_s, max_pos_jump_m)
    key_src = repr(params) + Path(__file__).read_text(encoding="utf-8") + "".join(
        f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns};" for f in bin_files)
    cache_file = data_dir / ".cache" / f"parsed_{hashlib.md5(key_src.encode()).hexdigest()[:16]}.pkl"
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as fh:
            all_segments, meta, mean_speed = pickle.load(fh)
        logger.info(f"Loaded {len(all_segments)} parsed segment(s) from cache {cache_file.name}")
        out = (_select(all_segments, feature_names), 1.0 / resample_hz, mean_speed)
        return out + (meta,) if return_meta else out

    all_segments: List[np.ndarray] = []
    meta: List[dict] = []
    seen: Dict[str, str] = {}
    for f in bin_files:
        digest = _file_md5(f)
        if digest in seen:
            logger.info(f"{f.name}: skipped — identical to {seen[digest]}")
            continue
        seen[digest] = f.name
        try:
            segs, info = parse_bin_file_full(f, min_mission_samples, resample_hz, modes, min_speed_ms,
                                              max_gap_s, gps_timeout_s, max_pos_jump_m)
        except ValueError as e:
            logger.warning(f"{f.name}: skipped — {e}")
            continue
        if not segs:
            logger.info(f"{f.name}: no usable segments ({info['in_mode_flying_s']:.0f}s in mode & flying)")
            continue
        all_segments.extend(segs)
        meta.extend({"file": f.name, "segment": i, "start_s": s}
                    for i, s in enumerate(info["segment_start_s"]))
        logger.info(f"{f.name}: {len(segs)} segment(s), {sum(map(len, segs))} samples [{info['source']}]")

    if not all_segments:
        raise ValueError(f"No valid trajectories extracted from {data_dir}")

    mean_speed = _mean_speed(all_segments)
    if not 3.0 < mean_speed < 150.0:
        logger.warning(f"Mean speed {mean_speed:.1f} m/s looks implausible — check the logs.")

    if use_cache:
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, "wb") as fh:
            pickle.dump((all_segments, meta, mean_speed), fh)

    out = (_select(all_segments, feature_names), 1.0 / resample_hz, mean_speed)
    return out + (meta,) if return_meta else out


def parse_directory_from_config(data_cfg: dict, return_meta: bool = False,
                                feature_names: Optional[Sequence[str]] = None) -> tuple:
    """``parse_directory`` with arguments taken from the ``data:`` config block."""
    return parse_directory(
        data_cfg["data_dir"],
        min_mission_samples=data_cfg.get("min_mission_samples", 250),
        resample_hz=data_cfg.get("resample_hz", 10.0),
        modes=data_cfg.get("modes", ["AUTO"]),
        min_speed_ms=data_cfg.get("min_speed_ms", 10.0),
        feature_names=feature_names or data_cfg.get("feature_names", list(DEFAULT_FEATURES)),
        return_meta=return_meta,
        max_gap_s=data_cfg.get("max_gap_s", 0.3),
        gps_timeout_s=data_cfg.get("gps_timeout_s", 1.0),
        max_pos_jump_m=data_cfg.get("max_pos_jump_m", 3.0),
    )


def split_by_log(meta: List[dict], lengths: Sequence[int], val_fraction: float = 0.15,
                 seed: int = 42) -> Tuple[List[int], List[int]]:
    """Split segment indices into train / val so that no log file is in both.

    Log files are shuffled with ``seed`` and moved to validation until it holds
    at least ``val_fraction`` of all (usable) samples.
    """
    files = sorted({m["file"] for m in meta})
    random.Random(seed).shuffle(files)
    total = float(sum(lengths))
    val_files, val_n = set(), 0
    for f in files:
        if val_n >= val_fraction * total:
            break
        n = sum(k for m, k in zip(meta, lengths) if m["file"] == f)
        if n == 0:
            continue
        val_files.add(f)
        val_n += n
    train_idx = [i for i, m in enumerate(meta) if m["file"] not in val_files]
    val_idx = [i for i, m in enumerate(meta) if m["file"] in val_files]
    return train_idx, val_idx


def load_split(config: dict) -> dict:
    """Parse the logs once (all columns) and split them by log file.

    Returns a dict with ``train`` / ``val`` lists of full-column segments
    (columns = FULL_COLUMNS), the model feature indices ``feat_idx``,
    ``train_files`` / ``val_files``, ``dt_s`` and ``mean_speed_ms``
    (mean ground speed of the training segments).
    """
    data_cfg = config["data"]
    segs, dt, _, meta = parse_directory_from_config(
        data_cfg, return_meta=True, feature_names=FULL_COLUMNS)
    seq = data_cfg["sequence_length"]
    lengths = [len(x) if len(x) >= seq else 0 for x in segs]   # unusable segments weigh 0
    tr, va = split_by_log(meta, lengths, data_cfg.get("val_fraction", 0.15),
                          config["training"].get("seed", 42))
    names = list(data_cfg.get("feature_names", DEFAULT_FEATURES))
    train = [segs[i] for i in tr]
    return {
        "train": train, "val": [segs[i] for i in va],
        "train_files": sorted({meta[i]["file"] for i in tr}),
        "val_files": sorted({meta[i]["file"] for i in va}),
        "feat_idx": [COL[n] for n in names], "feature_names": names,
        "dt_s": dt, "mean_speed_ms": _mean_speed(train),
    }


def select_features(segments: List[np.ndarray], feat_idx: Sequence[int]) -> List[np.ndarray]:
    """Pick model channels from full-column segments."""
    return [np.ascontiguousarray(x[:, list(feat_idx)]) for x in segments]


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PYTORCH DATASET
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """PyTorch Dataset wrapping parsed UAV trajectory segments.

    Applies a sliding window over each trajectory to produce fixed-length
    samples of shape ``[F, sequence_length]`` (channel-first, as
    required by ``nn.Conv1d``).

    Z-score normalisation is computed across the entire dataset and applied
    per feature channel.  The mean and std are stored as attributes so they
    can be used to denormalise generated trajectories during sampling.

    Parameters
    ----------
    trajectories : list of ndarray[T_i, F]
        Raw parsed trajectories (output of :func:`parse_directory`).
    sequence_length : int
        Fixed window length in samples.
    stride : int
        Sliding window step.  ``stride = sequence_length`` → no overlap.
    relative_channels : list of int
        Channels re-expressed relative to each window's first sample
        (e.g. altitude → altitude change since the window start).
    mirror_channels : list of int
        Channels that flip sign under a left/right mirror of the flight
        (roll).  Their normalisation is made symmetric (mean 0), so a
        mirrored sample is exactly ``-x`` in normalised space and training
        can flip them at random (see ``mirror_batch``).
    """

    def __init__(
        self,
        trajectories: List[np.ndarray],
        sequence_length: int = 256,
        stride: int = 128,
        mean_dt_s: float = 0.1,
        relative_channels: Sequence[int] = (),
        mirror_channels: Sequence[int] = (),
        stats: Optional[dict] = None,
    ):
        super().__init__()
        self.seq_len    = sequence_length
        self.stride     = stride
        self.mean_dt_s  = mean_dt_s  # uniform sample interval (s/step)
        self.relative_channels = list(relative_channels)
        self.mirror_channels   = list(mirror_channels)

        # ── Sliding-window segmentation ───────────────────────────────────────
        windows = []
        self.window_index = []   # (segment index, start sample) per window
        for k, traj in enumerate(trajectories):
            T = len(traj)
            for start in range(0, T - sequence_length + 1, stride):
                windows.append(traj[start : start + sequence_length])  # [L, F]
                self.window_index.append((k, start))

        if not windows:
            raise ValueError(
                "No windows could be extracted. "
                "Check sequence_length and stride relative to trajectory lengths."
            )

        # Stack into [N, L, F]
        data = np.stack(windows, axis=0).astype(np.float32)

        for c in self.relative_channels:
            data[:, :, c] -= data[:, :1, c]

        # ── Z-score normalisation (per feature channel, across all windows) ──
        if stats is not None:
            # Reuse statistics from another split (e.g. train stats for val).
            self.mean = np.asarray(stats["mean"], dtype=np.float32).copy()
            self.std  = np.asarray(stats["std"],  dtype=np.float32).copy()
        else:
            self.mean = data.mean(axis=(0, 1))          # [F]
            self.std  = data.std(axis=(0, 1)) + 1e-8    # [F]  (avoid div/0)
            for c in self.mirror_channels:
                # Statistics of the data plus its mirror image: mean 0, RMS scale.
                self.mean[c] = 0.0
                self.std[c]  = float(np.sqrt(np.mean(data[:, :, c] ** 2))) + 1e-8
        data_norm = (data - self.mean) / self.std   # [N, L, F]

        # Transpose to channel-first: [N, 7, L]
        self.data = torch.from_numpy(data_norm.transpose(0, 2, 1))  # [N, 7, L]

        logger.info(
            f"TrajectoryDataset: {len(self.data)} windows | "
            f"shape {tuple(self.data.shape[1:])} | "
            f"from {len(trajectories)} trajectory segment(s)."
        )

    @classmethod
    def from_config(cls, trajectories: List[np.ndarray], data_cfg: dict, mean_dt_s: float,
                    stats: Optional[dict] = None):
        """Build the dataset with the window and augmentation settings from the
        ``data:`` config block (used by train, sample and evaluate alike)."""
        names = list(data_cfg.get("feature_names", DEFAULT_FEATURES))
        relative = [names.index("alt")] if data_cfg.get("relative_alt", False) and "alt" in names else []
        mirror = [names.index("roll")] if data_cfg.get("mirror_roll", False) and "roll" in names else []
        return cls(
            trajectories,
            sequence_length=data_cfg["sequence_length"],
            stride=data_cfg["stride"],
            mean_dt_s=mean_dt_s,
            relative_channels=relative,
            mirror_channels=mirror,
            stats=stats,
        )

    @property
    def stats(self) -> dict:
        """Normalisation statistics (stored in checkpoints)."""
        return {"mean": [float(v) for v in self.mean], "std": [float(v) for v in self.std],
                "relative_channels": self.relative_channels,
                "mirror_channels": self.mirror_channels,
                "mean_dt_s": self.mean_dt_s}

    def mirror_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Mirror a random half of a normalised batch [B, F, L] left/right."""
        if not self.mirror_channels:
            return x
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        x = x.clone()
        for c in self.mirror_channels:
            x[flip, c] = -x[flip, c]
        return x

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a normalised trajectory window of shape ``[7, L]``."""
        return self.data[idx]

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse Z-score normalisation.

        Parameters
        ----------
        x : Tensor[..., 7, L]

        Returns
        -------
        x_raw : Tensor[..., 7, L]  — in original physical units
        """
        mean = torch.tensor(self.mean, device=x.device, dtype=x.dtype)
        std  = torch.tensor(self.std,  device=x.device, dtype=x.dtype)
        # Reshape to [..., 7, 1] for broadcasting along L
        return x * std[:, None] + mean[:, None]


# ──────────────────────────────────────────────────────────────────────────────
# 7.  DIAGNOSTICS CLI
# ──────────────────────────────────────────────────────────────────────────────

def _diagnose(data_dir: str, **kw) -> None:
    """Print per-log statistics so data problems are visible before training."""
    files = sorted(Path(data_dir).glob("*.bin"))
    seen: Dict[str, str] = {}
    all_segs: List[np.ndarray] = []
    hdr = f"{'file':<28}{'src':>8}{'fly_s':>7}{'use_s':>7}{'sparse':>7}{'badnav':>7}{'segs':>5}{'spd':>7}{'as':>6}{'|roll|99':>9}{'pitch':>13}{'alt':>14}"
    print(hdr); print("-" * len(hdr))
    for f in files:
        d = _file_md5(f)
        if d in seen:
            print(f"{f.name[:27]:<28}  duplicate of {seen[d]}"); continue
        seen[d] = f.name
        try:
            segs, info = parse_bin_file_full(f, **kw)
        except Exception as e:  # report and continue
            print(f"{f.name[:27]:<28}  ERROR {e}"); continue
        line = f"{f.name[:27]:<28}{info['source']:>8}{info['flying_s']:>7.0f}{info['in_mode_flying_s']:>7.0f}{info['sparse_flying_s']:>7.0f}{info['bad_gps_or_ekf_s']:>7.0f}{len(segs):>5}"
        if segs:
            s = np.concatenate(segs); all_segs.extend(segs)
            spd = np.linalg.norm(s[:, COL["vN"]:COL["vD"] + 1], axis=1).mean()
            asp = np.nanmean(s[:, COL["airspeed"]]) if np.isfinite(s[:, COL["airspeed"]]).any() else float("nan")
            r99 = np.degrees(np.percentile(np.abs(s[:, COL["roll"]]), 99))
            p = np.degrees(s[:, COL["pitch"]]); a = s[:, COL["alt"]]
            line += f"{spd:>7.1f}{asp:>6.1f}{r99:>9.0f}{p.min():>6.0f}..{p.max():<5.0f}{a.min():>6.0f}..{a.max():<6.0f}"
        print(line)
    if all_segs:
        s = np.concatenate(all_segs)
        L = kw.get("min_mission_samples", 250)
        print(f"\nTotal: {len(all_segs)} segments, {len(s)} samples "
              f"({len(s) * kw.get('resample_hz', 10.0) ** -1 / 60:.1f} min), "
              f"mean speed {_mean_speed(all_segs):.1f} m/s, segments >= {L} samples.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description="Print per-log parsing diagnostics")
    ap.add_argument("--data_dir", default="data/")
    ap.add_argument("--resample_hz", type=float, default=10.0)
    ap.add_argument("--min_mission_samples", type=int, default=256)
    ap.add_argument("--min_speed_ms", type=float, default=10.0)
    ap.add_argument("--modes", nargs="+", default=["AUTO"])
    ap.add_argument("--max_gap_s", type=float, default=0.3)
    ap.add_argument("--config", default=None, help="take all settings from a YAML config instead")
    a = ap.parse_args()
    if a.config:
        import yaml
        d = yaml.safe_load(open(a.config))["data"]
        _diagnose(d["data_dir"], min_mission_samples=d.get("min_mission_samples", 256),
                  resample_hz=d.get("resample_hz", 10.0), modes=d.get("modes", ["AUTO"]),
                  min_speed_ms=d.get("min_speed_ms", 10.0), max_gap_s=d.get("max_gap_s", 0.3),
                  gps_timeout_s=d.get("gps_timeout_s", 1.0), max_pos_jump_m=d.get("max_pos_jump_m", 3.0))
    else:
        _diagnose(a.data_dir, min_mission_samples=a.min_mission_samples, resample_hz=a.resample_hz,
                  modes=a.modes, min_speed_ms=a.min_speed_ms, max_gap_s=a.max_gap_s)
