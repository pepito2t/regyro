"""Builds a miniature ASL/EuRoC sequence on disk and reads it back.

The point is the frame conversion: gyro is recorded in the IMU frame and must come
out expressed in the camera frame, via both sensors' body transforms.
"""

import cv2
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from regyro.dataset.asl import AslError, load_sequence
from regyro.dataset.build import build_from_source
from regyro.lens_profile import FISHEYE_MODEL, RADTAN_MODEL

IMAGE_SIZE = (160, 120)
FRAME_COUNT = 6
GYRO_RATE = np.array([0.3, -0.2, 0.7])
NS = 1_000_000_000


def _transform(rotation: Rotation) -> dict:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    return {"cols": 4, "rows": 4, "data": matrix.flatten().tolist()}


def write_sequence(
    root,
    cam_rotation: Rotation = Rotation.identity(),
    distortion_model: str = "equidistant",
) -> None:
    cam_dir = root / "mav0" / "cam0"
    imu_dir = root / "mav0" / "imu0"
    (cam_dir / "data").mkdir(parents=True)
    imu_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows = []
    for index in range(FRAME_COUNT):
        timestamp = int((index / 30.0) * NS) + NS
        image = (rng.random((IMAGE_SIZE[1], IMAGE_SIZE[0])) * 255).astype(np.uint8)
        cv2.imwrite(str(cam_dir / "data" / f"{timestamp}.png"), image)
        rows.append(f"{timestamp},{timestamp}.png")
    (cam_dir / "data.csv").write_text("#timestamp [ns],filename\n" + "\n".join(rows) + "\n")

    (cam_dir / "sensor.yaml").write_text(
        yaml.safe_dump(
            {
                "sensor_type": "camera",
                "T_BS": _transform(cam_rotation),
                "resolution": list(IMAGE_SIZE),
                "intrinsics": [90.0, 90.0, 80.0, 60.0],
                "distortion_model": distortion_model,
                "distortion_coefficients": [0.01, 0.001, 0.0, 0.0],
            }
        )
    )

    imu_times = np.arange(0, int(2.0 * NS), int(NS / 200))
    imu_rows = [
        f"{int(t)},{GYRO_RATE[0]},{GYRO_RATE[1]},{GYRO_RATE[2]},0.0,0.0,9.81" for t in imu_times
    ]
    (imu_dir / "data.csv").write_text("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n" + "\n".join(imu_rows) + "\n")
    (imu_dir / "sensor.yaml").write_text(
        yaml.safe_dump({"sensor_type": "imu", "T_BS": _transform(Rotation.identity())})
    )


def test_load_sequence_reads_frames_and_gyro(tmp_path):
    write_sequence(tmp_path)

    sequence = load_sequence(tmp_path)

    assert sequence.profile.distortion_model == FISHEYE_MODEL
    assert sequence.profile.calib_width == IMAGE_SIZE[0]
    assert len(list(sequence.source)) == FRAME_COUNT
    # Camera aligned with the IMU: rates pass through unchanged.
    np.testing.assert_allclose(sequence.gyro.angular_velocity[0], GYRO_RATE, atol=1e-9)


def test_gyro_is_rotated_into_the_camera_frame(tmp_path):
    """A camera mounted rotated must report the same physical motion in its own axes."""
    cam_rotation = Rotation.from_euler("z", 90, degrees=True)
    write_sequence(tmp_path, cam_rotation=cam_rotation)

    sequence = load_sequence(tmp_path)

    expected = cam_rotation.as_matrix().T @ GYRO_RATE
    np.testing.assert_allclose(sequence.gyro.angular_velocity[0], expected, atol=1e-9)


def test_radtan_sequences_are_supported(tmp_path):
    write_sequence(tmp_path, distortion_model="radial-tangential")

    sequence = load_sequence(tmp_path)

    assert sequence.profile.distortion_model == RADTAN_MODEL


def test_sequence_builds_shards(tmp_path):
    write_sequence(tmp_path)
    sequence = load_sequence(tmp_path)

    stats = build_from_source(sequence.source, sequence.profile, sequence.gyro, tmp_path / "shards")

    assert stats.frame_pairs == FRAME_COUNT - 1
    shard = next((tmp_path / "shards").glob("*.npz"))
    with np.load(shard) as payload:
        np.testing.assert_allclose(
            payload["angular_velocity"], np.tile(GYRO_RATE, (FRAME_COUNT - 1, 1)), atol=1e-4
        )


def test_missing_directories_are_reported(tmp_path):
    with pytest.raises(AslError, match="is this an ASL sequence"):
        load_sequence(tmp_path)
