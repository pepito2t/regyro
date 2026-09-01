import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from regyro.classical.rotation_estimator import (
    RotationEstimationError,
    estimate_frame_pair,
)
from regyro.lens_profile import LensProfile

logger = logging.getLogger(__name__)

ANALYSIS_WIDTH = 960
RANDOM_SEED = 42


class VideoError(Exception):
    pass


@dataclass
class GyroEstimate:
    angular_velocities: np.ndarray
    sample_rate_hz: float
    failed_pairs: int


def _read_gray_frames(video_path: Path | str):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoError(f"cannot open video {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise VideoError(f"invalid frame rate in {video_path}")

    def frames():
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray.shape[1] > ANALYSIS_WIDTH:
                    scale = ANALYSIS_WIDTH / gray.shape[1]
                    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                yield gray
        finally:
            capture.release()

    return fps, frames()


def _fill_failed_pairs(rotation_vectors: list[np.ndarray | None]) -> tuple[np.ndarray, int]:
    valid_indices = [i for i, vec in enumerate(rotation_vectors) if vec is not None]
    if not valid_indices:
        raise VideoError("rotation estimation failed on every frame pair")
    failed = len(rotation_vectors) - len(valid_indices)
    valid_values = np.array([rotation_vectors[i] for i in valid_indices])
    all_indices = np.arange(len(rotation_vectors))
    filled = np.column_stack(
        [np.interp(all_indices, valid_indices, valid_values[:, axis]) for axis in range(3)]
    )
    return filled, failed


def estimate_gyro(video_path: Path | str, profile: LensProfile) -> GyroEstimate:
    """Estimate per-frame angular velocities (rad/s, camera frame) for a whole video."""
    fps, frames = _read_gray_frames(video_path)
    rng = np.random.default_rng(RANDOM_SEED)

    rotation_vectors: list[np.ndarray | None] = []
    prev_gray = None
    for frame_index, gray in enumerate(frames):
        if prev_gray is not None:
            try:
                result = estimate_frame_pair(prev_gray, gray, profile, rng)
                rotation_vectors.append(result.rotation_vector)
            except RotationEstimationError as exc:
                logger.warning("frame %d: %s", frame_index, exc)
                rotation_vectors.append(None)
        prev_gray = gray

    if not rotation_vectors:
        raise VideoError(f"video {video_path} has fewer than 2 frames")

    filled, failed = _fill_failed_pairs(rotation_vectors)
    if failed:
        logger.warning("interpolated %d/%d failed frame pairs", failed, len(rotation_vectors))
    return GyroEstimate(
        angular_velocities=filled * fps,
        sample_rate_hz=fps,
        failed_pairs=failed,
    )
