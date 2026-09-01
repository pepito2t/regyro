import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from regyro.classical.rotation_estimator import estimate_frame_pair, ransac_rotation
from regyro.lens_profile import LensProfile, LensProfileError

CAMERA_MATRIX = [[1000.0, 0.0, 1920.0], [0.0, 1000.0, 1080.0], [0.0, 0.0, 1.0]]
DISTORTION = [0.05, -0.01, 0.002, -0.0005]


@pytest.fixture
def profile(tmp_path) -> LensProfile:
    data = {
        "calib_dimension": {"w": 3840, "h": 2160},
        "fisheye_params": {"camera_matrix": CAMERA_MATRIX, "distortion_coeffs": DISTORTION},
    }
    path = tmp_path / "lens.json"
    path.write_text(json.dumps(data))
    return LensProfile.from_json(path)


def test_from_json_missing_params(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"calib_dimension": {"w": 100, "h": 100}}))
    with pytest.raises(LensProfileError):
        LensProfile.from_json(path)


def test_camera_matrix_scaling(profile):
    scaled = profile.camera_matrix_for(1920, 1080)
    assert scaled[0, 0] == pytest.approx(500.0)
    assert scaled[1, 2] == pytest.approx(540.0)


def test_undistort_round_trip_with_known_rotation(profile):
    """Project bearings through the fisheye model, then recover the rotation between frames."""
    rng = np.random.default_rng(3)
    true_rotation = Rotation.from_rotvec([0.01, -0.02, 0.03])

    bearings = rng.normal(size=(80, 3))
    bearings[:, 2] = np.abs(bearings[:, 2]) + 2.0
    bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
    rotated = bearings @ true_rotation.as_matrix().T

    camera_matrix = np.asarray(CAMERA_MATRIX)
    distortion = np.asarray(DISTORTION)
    points_a, _ = cv2.fisheye.projectPoints(
        bearings.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, distortion
    )
    points_b, _ = cv2.fisheye.projectPoints(
        rotated.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, distortion
    )

    bearings_a = profile.undistort_to_bearings(points_a.reshape(-1, 2), 3840, 2160)
    bearings_b = profile.undistort_to_bearings(points_b.reshape(-1, 2), 3840, 2160)
    estimated, _ = ransac_rotation(bearings_a, bearings_b, rng)

    delta = Rotation.from_matrix(estimated) * true_rotation.inv()
    assert np.linalg.norm(delta.as_rotvec()) < 1e-4
