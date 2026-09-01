from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from regyro.lens_profile import LensProfile

MAX_FEATURES = 400
FEATURE_QUALITY = 0.01
MIN_FEATURE_DISTANCE_PX = 12
MIN_TRACKED_POINTS = 12
RANSAC_ITERATIONS = 100
RANSAC_INLIER_ANGLE_RAD = 0.004
RANSAC_SAMPLE_SIZE = 3


class RotationEstimationError(Exception):
    pass


@dataclass
class FramePairResult:
    rotation_vector: np.ndarray
    inlier_ratio: float
    tracked_points: int


def track_features(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Track corners from prev_gray to gray, returning matched pixel coords (N, 2) each."""
    corners = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=MAX_FEATURES,
        qualityLevel=FEATURE_QUALITY,
        minDistance=MIN_FEATURE_DISTANCE_PX,
    )
    if corners is None or len(corners) < MIN_TRACKED_POINTS:
        return np.empty((0, 2)), np.empty((0, 2))

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None)
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, tracked, None)
    round_trip_error = np.linalg.norm((corners - back).reshape(-1, 2), axis=1)
    valid = (status.ravel() == 1) & (back_status.ravel() == 1) & (round_trip_error < 1.0)
    return corners.reshape(-1, 2)[valid], tracked.reshape(-1, 2)[valid]


def best_fit_rotation(bearings_a: np.ndarray, bearings_b: np.ndarray) -> np.ndarray:
    """Least-squares rotation R such that R @ a ≈ b (Kabsch)."""
    correlation = bearings_b.T @ bearings_a
    u, _, vt = np.linalg.svd(correlation)
    sign = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, sign]) @ vt


def ransac_rotation(
    bearings_a: np.ndarray,
    bearings_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(bearings_a)
    best_inliers = np.zeros(count, dtype=bool)
    for _ in range(RANSAC_ITERATIONS):
        sample = rng.choice(count, size=RANSAC_SAMPLE_SIZE, replace=False)
        candidate = best_fit_rotation(bearings_a[sample], bearings_b[sample])
        residual_cos = np.sum((bearings_a @ candidate.T) * bearings_b, axis=1)
        inliers = np.arccos(np.clip(residual_cos, -1.0, 1.0)) < RANSAC_INLIER_ANGLE_RAD
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers.sum() < RANSAC_SAMPLE_SIZE:
        raise RotationEstimationError("not enough RANSAC inliers")
    refined = best_fit_rotation(bearings_a[best_inliers], bearings_b[best_inliers])
    return refined, best_inliers


def estimate_frame_pair(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    profile: LensProfile,
    rng: np.random.Generator,
) -> FramePairResult:
    points_a, points_b = track_features(prev_gray, gray)
    if len(points_a) < MIN_TRACKED_POINTS:
        raise RotationEstimationError(f"only {len(points_a)} tracked points")

    height, width = gray.shape
    bearings_a = profile.undistort_to_bearings(points_a, width, height)
    bearings_b = profile.undistort_to_bearings(points_b, width, height)

    rotation, inliers = ransac_rotation(bearings_a, bearings_b, rng)
    # World is static: bearings move by R, so the camera rotated by R^T.
    camera_rotation = Rotation.from_matrix(rotation.T)
    return FramePairResult(
        rotation_vector=camera_rotation.as_rotvec(),
        inlier_ratio=float(inliers.mean()),
        tracked_points=len(points_a),
    )
