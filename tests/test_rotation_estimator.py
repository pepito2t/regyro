import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from regyro.classical.rotation_estimator import (
    RotationEstimationError,
    best_fit_rotation,
    ransac_rotation,
)


def random_bearings(rng: np.random.Generator, count: int) -> np.ndarray:
    vectors = rng.normal(size=(count, 3))
    vectors[:, 2] = np.abs(vectors[:, 2]) + 1.0
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def rotation_angle_error(estimated: np.ndarray, expected: Rotation) -> float:
    delta = Rotation.from_matrix(estimated) * expected.inv()
    return float(np.linalg.norm(delta.as_rotvec()))


def test_best_fit_rotation_recovers_exact_rotation():
    rng = np.random.default_rng(0)
    true_rotation = Rotation.from_rotvec([0.02, -0.05, 0.1])
    bearings = random_bearings(rng, 50)
    rotated = bearings @ true_rotation.as_matrix().T

    estimated = best_fit_rotation(bearings, rotated)

    assert rotation_angle_error(estimated, true_rotation) < 1e-10


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(1)
    true_rotation = Rotation.from_rotvec([0.03, 0.01, -0.04])
    bearings = random_bearings(rng, 100)
    rotated = bearings @ true_rotation.as_matrix().T
    rotated[:20] = random_bearings(rng, 20)

    estimated, inliers = ransac_rotation(bearings, rotated, rng)

    assert rotation_angle_error(estimated, true_rotation) < 1e-3
    assert inliers[20:].mean() > 0.95
    assert inliers[:20].mean() < 0.2


def test_ransac_fails_on_pure_noise():
    rng = np.random.default_rng(2)
    with pytest.raises(RotationEstimationError):
        ransac_rotation(random_bearings(rng, 30), random_bearings(rng, 30), rng)
