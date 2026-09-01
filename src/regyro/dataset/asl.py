"""Reader for the ASL/EuRoC directory layout used by EuRoC MAV, TUM-VI and others.

    <sequence>/mav0/cam0/{sensor.yaml, data.csv, data/*.png}
    <sequence>/mav0/imu0/{sensor.yaml, data.csv}

Gyro rates are recorded in the IMU sensor frame; the model needs them in the camera
frame, so both sensors' body transforms are composed to rotate them across.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from regyro.dataset.frame_source import ImageSequenceFrameSource
from regyro.dataset.telemetry import GyroSamples
from regyro.errors import RegyroError
from regyro.lens_profile import LensProfile

logger = logging.getLogger(__name__)

NANOSECONDS_PER_SECOND = 1e9


class AslError(RegyroError):
    pass


@dataclass
class AslSequence:
    name: str
    source: ImageSequenceFrameSource
    gyro: GyroSamples
    profile: LensProfile


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise AslError("reading ASL sequences needs PyYAML: `uv sync --extra datasets`") from exc
    try:
        return yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise AslError(f"cannot read {path}: {exc}") from exc


def _transform_from_yaml(sensor: dict, path: Path) -> np.ndarray:
    block = sensor.get("T_BS")
    if block is None:
        logger.info("%s has no T_BS; assuming it is aligned with the body frame", path.name)
        return np.eye(4)
    data = np.asarray(block["data"], dtype=np.float64)
    if data.size != 16:
        raise AslError(f"{path}: T_BS must hold 16 values, got {data.size}")
    return data.reshape(4, 4)


def _read_csv_rows(path: Path) -> list[list[str]]:
    try:
        with path.open(newline="") as handle:
            rows = [row for row in csv.reader(handle) if row and not row[0].startswith("#")]
    except OSError as exc:
        raise AslError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise AslError(f"{path} contains no data rows")
    return rows


def _read_imu(imu_dir: Path) -> tuple[GyroSamples, np.ndarray]:
    sensor = _load_yaml(imu_dir / "sensor.yaml")
    rows = _read_csv_rows(imu_dir / "data.csv")

    try:
        timestamps = np.array([int(row[0]) for row in rows], dtype=np.float64)
        rates = np.array([[float(value) for value in row[1:4]] for row in rows])
    except (ValueError, IndexError) as exc:
        raise AslError(f"{imu_dir / 'data.csv'}: expected timestamp + 3 gyro columns") from exc

    samples = GyroSamples(timestamps / NANOSECONDS_PER_SECOND, rates)
    return samples, _transform_from_yaml(sensor, imu_dir / "sensor.yaml")


def _read_camera(cam_dir: Path, name: str) -> tuple[ImageSequenceFrameSource, np.ndarray, LensProfile]:
    sensor = _load_yaml(cam_dir / "sensor.yaml")
    rows = _read_csv_rows(cam_dir / "data.csv")

    image_dir = cam_dir / "data"
    try:
        timestamps = np.array([int(row[0]) for row in rows], dtype=np.float64)
        paths = [image_dir / row[1].strip() for row in rows]
    except (ValueError, IndexError) as exc:
        raise AslError(f"{cam_dir / 'data.csv'}: expected timestamp + filename columns") from exc

    missing = [path for path in paths[:5] if not path.exists()]
    if missing:
        raise AslError(f"{cam_dir}: image files are missing, e.g. {missing[0]}")

    source = ImageSequenceFrameSource(name, paths, timestamps / NANOSECONDS_PER_SECOND)
    return source, _transform_from_yaml(sensor, cam_dir / "sensor.yaml"), _profile_from_yaml(sensor, cam_dir)


def _profile_from_yaml(sensor: dict, cam_dir: Path) -> LensProfile:
    try:
        focal_x, focal_y, center_x, center_y = sensor["intrinsics"]
        width, height = sensor["resolution"]
        distortion = sensor.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0])
        model = sensor.get("distortion_model", "radial-tangential")
    except (KeyError, ValueError) as exc:
        raise AslError(f"{cam_dir / 'sensor.yaml'}: incomplete camera calibration: {exc}") from exc

    camera_matrix = np.array(
        [[focal_x, 0.0, center_x], [0.0, focal_y, center_y], [0.0, 0.0, 1.0]]
    )
    return LensProfile.from_intrinsics(camera_matrix, distortion, width, height, model)


def load_sequence(sequence_dir: Path, camera: str = "cam0") -> AslSequence:
    """Load one ASL sequence, with gyro already rotated into the camera frame."""
    root = sequence_dir / "mav0"
    if not root.is_dir():
        root = sequence_dir
    cam_dir, imu_dir = root / camera, root / "imu0"
    for directory in (cam_dir, imu_dir):
        if not directory.is_dir():
            raise AslError(f"{sequence_dir}: missing {directory.name}; is this an ASL sequence?")

    gyro, imu_to_body = _read_imu(imu_dir)
    name = f"{sequence_dir.name}-{camera}"
    source, cam_to_body, profile = _read_camera(cam_dir, name)

    # w_cam = R_body_cam^T * R_body_imu * w_imu
    rotation = cam_to_body[:3, :3].T @ imu_to_body[:3, :3]
    return AslSequence(name=name, source=source, gyro=gyro.rotate(rotation), profile=profile)
