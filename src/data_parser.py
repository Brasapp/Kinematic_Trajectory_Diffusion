"""
data_parser.py
==============
Parses ArduPilot DataFlash binary (.bin) logs and produces normalised
PyTorch trajectory tensors for diffusion model training.

Pipeline
--------
1.  Open a ``.bin`` file with ``pymavlink.DFReader.DFReader_binary``.
2.  Extract ``MODE`` messages to locate **AUTO-mode** flight segments.
    Only segments with at least ``min_mission_samples`` data points are kept.
3.  Within each AUTO segment extract:
    - ``GPS`` (or ``POS``) messages → Lat, Lon, Alt
    - ``AHR2`` messages           → Yaw (heading, degrees)
    - ``NKF1`` messages           → VN, VE, VD (velocity m/s, EKF estimate)
      Fallback: numerical differentiation of NED position if NKF1 is absent.
4.  Convert Lat/Lon/Alt to local NED (North-East-Down) metres using the
    equirectangular projection anchored at the first valid GPS fix.
5.  Align all message streams to a common timestamp grid via linear
    interpolation.
6.  Produce a feature tensor of shape ``[T, 7]`` with columns
    ``[N, E, D, vN, vE, vD, yaw_rad]``.
7.  ``TrajectoryDataset`` wraps all parsed tensors, applies Z-score
    normalisation, and yields fixed-length sliding windows suitable for
    the diffusion model.

Coordinate Conversion
---------------------
Equirectangular approximation for converting WGS-84 lat/lon/alt to local NED:

.. math::

    N = R \\cdot (\\phi - \\phi_0), \\qquad
    E = R \\cdot \\cos(\\phi_0) \\cdot (\\lambda - \\lambda_0), \\qquad
    D = -(h - h_0)

where :math:`R = 6{,}371{,}000` m is Earth's mean radius,
:math:`(\\phi_0, \\lambda_0, h_0)` is the reference origin (first GPS fix),
and :math:`h` is altitude (m).

AUTO Mode Detection
-------------------
ArduPilot ``MODE`` log messages carry a numeric mode field.
The AUTO mode numbers are:
- ArduCopter : ``3``
- ArduPlane  : ``10``
Both are detected automatically.
"""

import os
import math
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

try:
    from pymavlink.DFReader import DFReader_binary
except ImportError as exc:
    raise ImportError(
        "pymavlink is required: pip install pymavlink"
    ) from exc

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EARTH_RADIUS_M        = 6_371_000.0    # mean Earth radius in metres
AUTO_MODE_COPTER      = 3              # ArduCopter AUTO mode number
AUTO_MODE_PLANE       = 10             # ArduPlane  AUTO mode number
AUTO_MODE_NAMES       = {"auto"}       # lower-case mode name fallback


# ──────────────────────────────────────────────────────────────────────────────
# 1.  COORDINATE CONVERSION
# ──────────────────────────────────────────────────────────────────────────────

def latlon_to_ned(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    lat0: float,
    lon0: float,
    alt0: float,
) -> np.ndarray:
    r"""Convert WGS-84 geodetic coordinates to local NED (metres).

    Uses the equirectangular (flat-Earth) approximation — valid for
    flight areas smaller than ~50 km across.

    .. math::

        N = R \cdot (\phi - \phi_0), \qquad
        E = R \cdot \cos(\phi_0) \cdot (\lambda - \lambda_0), \qquad
        D = -(h - h_0)

    Parameters
    ----------
    lat, lon, alt : ndarray[T] — geodetic coordinates (degrees, degrees, metres)
    lat0, lon0, alt0 : float  — reference origin (first valid GPS fix)

    Returns
    -------
    ned : ndarray[T, 3] — local NED coordinates in metres
    """
    phi   = np.deg2rad(lat)
    phi0  = math.radians(lat0)
    lam   = np.deg2rad(lon)
    lam0  = math.radians(lon0)

    N =  EARTH_RADIUS_M * (phi - phi0)
    E =  EARTH_RADIUS_M * math.cos(phi0) * (lam - lam0)
    D = -(alt - alt0)

    return np.stack([N, E, D], axis=-1)   # [T, 3]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  AUTO-MODE SEGMENT DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def _is_auto_mode(mode_val) -> bool:
    """Return True if the raw mode value corresponds to ArduPilot AUTO mode."""
    if isinstance(mode_val, (int, float)):
        return int(mode_val) in (AUTO_MODE_COPTER, AUTO_MODE_PLANE)
    if isinstance(mode_val, str):
        return mode_val.strip().lower() in AUTO_MODE_NAMES
    return False


def _extract_auto_intervals(
    log: DFReader_binary,
) -> List[Tuple[float, float]]:
    """Parse MODE messages and return (start_us, end_us) tuples for each
    AUTO-mode interval.

    Parameters
    ----------
    log : DFReader_binary — rewound to the beginning

    Returns
    -------
    intervals : list of (start_us, end_us) in microseconds
    """
    log.rewind()
    mode_events: List[Tuple[float, bool]] = []

    while True:
        msg = log.recv_match(type="MODE", blocking=False)
        if msg is None:
            break
        timestamp_us = getattr(msg, "TimeUS", None) or getattr(msg, "_timestamp", 0) * 1e6
        mode_val = getattr(msg, "Mode", None)
        if mode_val is None:
            continue
        mode_events.append((float(timestamp_us), _is_auto_mode(mode_val)))

    if not mode_events:
        logger.warning("No MODE messages found — treating entire log as AUTO.")
        return [(0.0, float("inf"))]

    intervals: List[Tuple[float, float]] = []
    in_auto  = False
    start_us = 0.0

    for ts, is_auto in mode_events:
        if is_auto and not in_auto:
            start_us = ts
            in_auto  = True
        elif not is_auto and in_auto:
            intervals.append((start_us, ts))
            in_auto = False

    # If still in AUTO at end of log
    if in_auto:
        intervals.append((start_us, float("inf")))

    logger.info(f"Found {len(intervals)} AUTO-mode interval(s).")
    return intervals


# ──────────────────────────────────────────────────────────────────────────────
# 3.  RAW DATA EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def _extract_streams(log: DFReader_binary) -> dict:
    """Extract all relevant message streams from the log into numpy arrays.

    Streams extracted
    -----------------
    gps  : TimeUS, Lat, Lng, Alt
    ahr2 : TimeUS, Yaw  (degrees)
    nkf1 : TimeUS, VN, VE, VD  (velocity, m/s)  — may be empty
    mode : TimeUS, Mode (int)

    Returns
    -------
    dict with keys 'gps', 'ahr2', 'nkf1', each a dict of arrays keyed by field name.
    """
    streams = {
        "gps":  {"t": [], "lat": [], "lng": [], "alt": []},
        "ahr2": {"t": [], "yaw": [], "roll": [], "pitch": []},
        "nkf1": {"t": [], "vn": [], "ve": [], "vd": []},
    }
    accepted_gps = ("GPS", "POS")

    log.rewind()

    while True:
        msg = log.recv_match(type={"GPS", "POS", "AHR2", "NKF1"}, blocking=False)
        if msg is None:
            break

        mtype = msg.get_type()
        ts = float(getattr(msg, "TimeUS", 0))

        if mtype in accepted_gps:
            lat = getattr(msg, "Lat", None)
            lng = getattr(msg, "Lng", getattr(msg, "Lon", None))
            alt = getattr(msg, "Alt", None)
            if lat is not None and lng is not None and alt is not None:
                streams["gps"]["t"].append(ts)
                streams["gps"]["lat"].append(lat)
                streams["gps"]["lng"].append(lng)
                streams["gps"]["alt"].append(alt)

        elif mtype == "AHR2":
            yaw   = getattr(msg, "Yaw",   None)
            roll  = getattr(msg, "Roll",  None)
            pitch = getattr(msg, "Pitch", None)
            if yaw is not None:
                streams["ahr2"]["t"].append(ts)
                streams["ahr2"]["yaw"].append(yaw)
                streams["ahr2"]["roll"].append(roll if roll is not None else 0.0)
                streams["ahr2"]["pitch"].append(pitch if pitch is not None else 0.0)

        elif mtype == "NKF1":
            vn = getattr(msg, "VN", None)
            ve = getattr(msg, "VE", None)
            vd = getattr(msg, "VD", None)
            if vn is not None:
                streams["nkf1"]["t"].append(ts)
                streams["nkf1"]["vn"].append(vn)
                streams["nkf1"]["ve"].append(ve)
                streams["nkf1"]["vd"].append(vd)

    # Convert to numpy
    for key in streams:
        for field in streams[key]:
            streams[key][field] = np.array(streams[key][field], dtype=np.float64)

    return streams


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TRAJECTORY ASSEMBLY
# ──────────────────────────────────────────────────────────────────────────────

def _interp(t_src: np.ndarray, y_src: np.ndarray,
            t_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation with extrapolation clamped to boundary values."""
    if len(t_src) < 2:
        return np.full(len(t_dst), y_src[0] if len(y_src) else np.nan)
    f = interp1d(t_src, y_src, kind="linear", bounds_error=False,
                 fill_value=(y_src[0], y_src[-1]))
    return f(t_dst)


def _build_trajectory(
    streams: dict,
    start_us: float,
    end_us: float,
) -> Optional[tuple]:
    """Build a [T, 9] trajectory array for a single AUTO-mode interval.

    Feature columns: [N, E, D, vN, vE, vD, yaw_rad, roll_rad, pitch_rad]

    Also estimates the mean GPS sample interval dt_s (seconds/step) from
    the actual log timestamps, used later for the coordinated-turn constraint.

    Parameters
    ----------
    streams  : raw message streams from _extract_streams
    start_us : interval start in microseconds
    end_us   : interval end in microseconds

    Returns
    -------
    (traj, dt_s) : (ndarray[T, 9], float) or None if too few points
    """
    gps = streams["gps"]
    t_gps = gps["t"]

    # Mask GPS to this AUTO interval
    mask = (t_gps >= start_us) & (t_gps <= end_us)
    if mask.sum() < 2:
        return None

    t_ref = t_gps[mask]
    lat   = gps["lat"][mask]
    lng   = gps["lng"][mask]
    alt   = gps["alt"][mask]

    # Mean timestep in seconds (from actual log timestamps)
    dt_s = float(np.mean(np.diff(t_ref)) / 1e6) if len(t_ref) > 1 else 0.1
    dt_s = max(dt_s, 1e-4)  # guard against degenerate logs

    # Reference origin = first GPS fix in this interval
    lat0, lon0, alt0 = lat[0], lng[0], alt[0]

    # Convert to NED
    ned = latlon_to_ned(lat, lng, alt, lat0, lon0, alt0)  # [T, 3]

    # Attitude: yaw / roll / pitch (AHR2, interpolated onto GPS timestamps)
    yaw_deg   = _interp(streams["ahr2"]["t"], streams["ahr2"]["yaw"],   t_ref)
    roll_deg  = _interp(streams["ahr2"]["t"], streams["ahr2"]["roll"],  t_ref)
    pitch_deg = _interp(streams["ahr2"]["t"], streams["ahr2"]["pitch"], t_ref)

    yaw_rad   = np.deg2rad(yaw_deg)    # [T]
    roll_rad  = np.deg2rad(roll_deg)   # [T]
    pitch_rad = np.deg2rad(pitch_deg)  # [T]

    # Velocity: prefer NKF1, fallback to central differences of NED
    if len(streams["nkf1"]["t"]) > 1:
        vn = _interp(streams["nkf1"]["t"], streams["nkf1"]["vn"], t_ref)
        ve = _interp(streams["nkf1"]["t"], streams["nkf1"]["ve"], t_ref)
        vd = _interp(streams["nkf1"]["t"], streams["nkf1"]["vd"], t_ref)
    else:
        logger.debug("NKF1 not found — computing velocity by finite differences.")
        dt = np.diff(t_ref) / 1e6  # convert us -> s
        dt = np.where(dt == 0, 1e-6, dt)  # avoid division by zero

        def _central_diff(x, dt):
            grad = np.empty_like(x)
            grad[1:-1] = (x[2:] - x[:-2]) / (dt[1:] + dt[:-1])
            grad[0]    = (x[1] - x[0]) / dt[0]
            grad[-1]   = (x[-1] - x[-2]) / dt[-1]
            return grad

        vn = _central_diff(ned[:, 0], dt)
        ve = _central_diff(ned[:, 1], dt)
        vd = _central_diff(ned[:, 2], dt)

    # Stack into [T, 9]: position + velocity + attitude
    traj = np.stack([
        ned[:, 0],   # N   (m)
        ned[:, 1],   # E   (m)
        ned[:, 2],   # D   (m)
        vn,          # vN  (m/s)
        ve,          # vE  (m/s)
        vd,          # vD  (m/s)
        yaw_rad,     # yaw   (rad)
        roll_rad,    # roll  (rad)
        pitch_rad,   # pitch (rad)
    ], axis=-1).astype(np.float32)

    return traj, dt_s


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def parse_bin_file(
    path: str,
    min_mission_samples: int = 100,
) -> tuple:
    """Parse an ArduPilot ``.bin`` log and extract AUTO-mode trajectory segments.

    Parameters
    ----------
    path : str
        Path to the ``.bin`` DataFlash log file.
    min_mission_samples : int
        Minimum number of GPS samples required for a segment to be returned.
        Segments shorter than this are silently discarded.

    Returns
    -------
    trajectories : list of ndarray[T_i, 9]
    mean_dt_s    : float — mean GPS sample interval in seconds

    Raises
    ------
    FileNotFoundError  : if path does not exist
    ValueError         : if no valid AUTO segments are found in the log
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    logger.info(f"Parsing log: {path.name}")

    log = DFReader_binary(str(path), zero_time_base=True)

    # Step 1: find AUTO-mode time intervals
    intervals = _extract_auto_intervals(log)

    # Step 2: extract all raw message streams in a single pass
    streams = _extract_streams(log)

    if len(streams["gps"]["t"]) == 0:
        raise ValueError(f"No GPS/POS messages found in {path.name}")

    # Step 3: build a trajectory for each AUTO interval
    trajectories: List[np.ndarray] = []
    dt_samples: List[float] = []
    for i, (start_us, end_us) in enumerate(intervals):
        result = _build_trajectory(streams, start_us, end_us)
        if result is None:
            continue
        traj, dt_s = result
        if len(traj) < min_mission_samples:
            logger.debug(
                f"Interval {i}: skipped "
                f"({len(traj)} samples < {min_mission_samples})"
            )
            continue
        trajectories.append(traj)
        logger.info(f"Interval {i}: kept — {len(traj)} samples.")

    if not trajectories:
        raise ValueError(
            f"No AUTO-mode segments with >= {min_mission_samples} samples found in {path.name}. "
            "Check that the log contains a long enough AUTO mission."
        )

    mean_dt_s = float(np.mean(dt_samples)) if dt_samples else 0.1
    return trajectories, mean_dt_s


def parse_directory(
    data_dir: str,
    min_mission_samples: int = 100,
) -> tuple:
    """Parse all ``.bin`` files in a directory.

    Parameters
    ----------
    data_dir : str
        Directory containing ``.bin`` files.
    min_mission_samples : int
        Forwarded to :func:`parse_bin_file`.

    Returns
    -------
    all_trajectories : list of ndarray[T_i, 9]
    mean_dt_s        : float — mean GPS sample interval in seconds
    """
    data_dir = Path(data_dir)
    bin_files = sorted(data_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {data_dir}")

    all_trajectories: List[np.ndarray] = []
    all_dts: List[float] = []
    for f in bin_files:
        try:
            trajs, dt_s = parse_bin_file(str(f), min_mission_samples)
            all_trajectories.extend(trajs)
            all_dts.append(dt_s)
            logger.info(f"{f.name}: {len(trajs)} segment(s) extracted.")
        except ValueError as e:
            logger.warning(f"{f.name}: skipped — {e}")

    if not all_trajectories:
        raise ValueError(f"No valid trajectories extracted from {data_dir}")

    mean_dt_s = float(np.mean(all_dts)) if all_dts else 0.1
    return all_trajectories, mean_dt_s


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PYTORCH DATASET
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """PyTorch Dataset wrapping parsed UAV trajectory segments.

    Applies a sliding window over each trajectory to produce fixed-length
    samples of shape ``[features, sequence_length]`` (channel-first, as
    required by ``nn.Conv1d``).

    Z-score normalisation is computed across the entire dataset and applied
    per feature channel.  The mean and std are stored as attributes so they
    can be used to denormalise generated trajectories during sampling.

    Parameters
    ----------
    trajectories : list of ndarray[T_i, 7]
        Raw parsed trajectories (output of :func:`parse_directory`).
    sequence_length : int
        Fixed window length in samples.
    stride : int
        Sliding window step.  ``stride = sequence_length`` → no overlap.
    """

    def __init__(
        self,
        trajectories: List[np.ndarray],
        sequence_length: int = 256,
        stride: int = 128,
        mean_dt_s: float = 0.1,
    ):
        super().__init__()
        self.seq_len    = sequence_length
        self.stride     = stride
        self.mean_dt_s  = mean_dt_s  # GPS sample interval (s/step), for coordinated-turn constraint

        # ── Sliding-window segmentation ───────────────────────────────────────
        windows = []
        for traj in trajectories:
            T = len(traj)
            for start in range(0, T - sequence_length + 1, stride):
                windows.append(traj[start : start + sequence_length])  # [L, 7]

        if not windows:
            raise ValueError(
                "No windows could be extracted. "
                "Check sequence_length and stride relative to trajectory lengths."
            )

        # Stack into [N, L, 7]
        data = np.stack(windows, axis=0).astype(np.float32)

        # ── Z-score normalisation (per feature channel, across all windows) ──
        # mean/std computed over (N, L) dimensions → shape [7]
        self.mean = data.mean(axis=(0, 1))   # [7]
        self.std  = data.std(axis=(0, 1)) + 1e-8  # [7]  (avoid div/0)
        data_norm = (data - self.mean) / self.std  # [N, L, 7]

        # Transpose to channel-first: [N, 7, L]
        self.data = torch.from_numpy(data_norm.transpose(0, 2, 1))  # [N, 7, L]

        logger.info(
            f"TrajectoryDataset: {len(self.data)} windows | "
            f"shape {tuple(self.data.shape[1:])} | "
            f"from {len(trajectories)} trajectory segment(s)."
        )

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
