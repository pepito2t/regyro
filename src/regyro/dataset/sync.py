import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from regyro.classical.video_pipeline import estimate_gyro, read_frame_rate
from regyro.dataset.telemetry import (
    AxisMapping,
    GyroSamples,
    TelemetryError,
    find_axis_mapping,
    load_gyro,
)
from regyro.lens_profile import LensProfile
from regyro.errors import RegyroError

logger = logging.getLogger(__name__)

MAX_OFFSET_S = 2.0
CALIBRATION_SECONDS = 20.0
MIN_MOTION_RAD_S = 0.05


class SyncError(RegyroError):
    pass


@dataclass(frozen=True)
class SyncCalibration:
    """How a video's embedded IMU relates to its image stream."""

    time_offset_s: float
    axis_mapping: AxisMapping
    rmse_rad_s: float


def _parabolic_peak(scores: np.ndarray, peak: int) -> float:
    """Sub-sample peak location by fitting a parabola through the peak and neighbours."""
    if peak <= 0 or peak >= len(scores) - 1:
        return float(peak)
    left, middle, right = scores[peak - 1], scores[peak], scores[peak + 1]
    denominator = left - 2.0 * middle + right
    if abs(denominator) < 1e-12:
        return float(peak)
    return float(peak) + 0.5 * (left - right) / denominator


def estimate_time_offset(
    estimated: np.ndarray,
    gyro: GyroSamples,
    frame_times: np.ndarray,
    fps: float,
) -> float:
    """Cross-correlate motion magnitude to find the IMU-to-video time offset in seconds.

    Magnitude is used rather than per-axis rates so the offset can be found before
    the axis mapping is known.
    """
    estimated_magnitude = np.linalg.norm(estimated, axis=1)
    if estimated_magnitude.std() < MIN_MOTION_RAD_S:
        raise SyncError("video segment is too static to synchronise")

    max_lag = int(round(MAX_OFFSET_S * fps))
    lags = np.arange(-max_lag, max_lag + 1)
    reference = estimated_magnitude - estimated_magnitude.mean()

    scores = np.empty(len(lags))
    for index, lag in enumerate(lags):
        shifted = gyro.resample(frame_times + lag / fps)
        magnitude = np.linalg.norm(shifted, axis=1)
        magnitude = magnitude - magnitude.mean()
        norm = np.linalg.norm(reference) * np.linalg.norm(magnitude)
        scores[index] = float(reference @ magnitude / norm) if norm > 0 else 0.0

    best = int(np.argmax(scores))
    return float(lags[0] + _parabolic_peak(scores, best)) / fps


def calibrate_video(
    video_path: Path | str,
    profile: LensProfile,
    calibration_seconds: float = CALIBRATION_SECONDS,
) -> SyncCalibration:
    """Determine the time offset and IMU-to-camera axis mapping for one video."""
    try:
        gyro = load_gyro(video_path)
    except TelemetryError as exc:
        raise SyncError(str(exc)) from exc

    fps = read_frame_rate(video_path)
    probe = estimate_gyro(video_path, profile, max_frames=int(calibration_seconds * fps) + 1)
    estimated = probe.angular_velocities
    # Each estimate spans a frame interval, so it is timestamped at the interval centre.
    frame_times = (np.arange(len(estimated)) + 0.5) / fps

    offset = estimate_time_offset(estimated, gyro, frame_times, fps)
    aligned = gyro.resample(frame_times + offset)

    # Mapping takes IMU axes onto camera axes, which is the direction the builder needs.
    mapping, rmse = find_axis_mapping(aligned, estimated)
    logger.info("offset %.4f s, mapping %s, rmse %.4f rad/s", offset, mapping.to_string(), rmse)
    return SyncCalibration(time_offset_s=offset, axis_mapping=mapping, rmse_rad_s=rmse)
