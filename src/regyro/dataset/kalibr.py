"""Reader for Kalibr camchain YAML, the calibration format shipped with UZH-FPV.

Unlike ASL sensor.yaml this gives the camera-from-IMU transform directly, so no
composition through a body frame is needed.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from regyro.errors import RegyroError
from regyro.lens_profile import LensProfile


class KalibrError(RegyroError):
    pass


@dataclass(frozen=True)
class KalibrCalibration:
    profile: LensProfile
    rotation_cam_imu: np.ndarray


def load_calibration(path: Path, camera: str = "cam0") -> KalibrCalibration:
    try:
        import yaml
    except ImportError as exc:
        raise KalibrError("reading Kalibr YAML needs PyYAML: `uv sync --extra datasets`") from exc

    try:
        document = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise KalibrError(f"cannot read {path}: {exc}") from exc

    if not isinstance(document, dict) or camera not in document:
        available = sorted(document) if isinstance(document, dict) else []
        raise KalibrError(f"{path}: no entry for {camera!r}; found {available}")

    entry = document[camera]
    try:
        focal_x, focal_y, center_x, center_y = entry["intrinsics"]
        width, height = entry["resolution"]
        distortion = entry.get("distortion_coeffs", [0.0, 0.0, 0.0, 0.0])
        model = entry.get("distortion_model", "equidistant")
    except (KeyError, ValueError) as exc:
        raise KalibrError(f"{path}: incomplete calibration for {camera}: {exc}") from exc

    camera_matrix = np.array(
        [[focal_x, 0.0, center_x], [0.0, focal_y, center_y], [0.0, 0.0, 1.0]]
    )
    profile = LensProfile.from_intrinsics(camera_matrix, distortion, width, height, model)

    transform = entry.get("T_cam_imu")
    if transform is None:
        raise KalibrError(f"{path}: {camera} has no T_cam_imu, cannot align gyro to the camera")
    rotation = np.asarray(transform, dtype=np.float64)
    if rotation.shape != (4, 4):
        raise KalibrError(f"{path}: T_cam_imu must be 4x4, got {rotation.shape}")

    return KalibrCalibration(profile=profile, rotation_cam_imu=rotation[:3, :3])
