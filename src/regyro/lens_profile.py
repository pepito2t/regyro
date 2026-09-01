import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from regyro.errors import RegyroError

FISHEYE_MODEL = "opencv_fisheye"
RADTAN_MODEL = "pinhole_radtan"
SUPPORTED_MODELS = (FISHEYE_MODEL, RADTAN_MODEL)

# Names used by ASL/EuRoC and Kalibr calibration files.
MODEL_ALIASES = {
    "equidistant": FISHEYE_MODEL,
    "fisheye": FISHEYE_MODEL,
    "kannala_brandt": FISHEYE_MODEL,
    "opencv_fisheye": FISHEYE_MODEL,
    "radial-tangential": RADTAN_MODEL,
    "radtan": RADTAN_MODEL,
    "plumb_bob": RADTAN_MODEL,
    "pinhole_radtan": RADTAN_MODEL,
}


class LensProfileError(RegyroError):
    pass


def normalise_model_name(name: str) -> str:
    resolved = MODEL_ALIASES.get(name.strip().lower())
    if resolved is None:
        raise LensProfileError(f"unsupported distortion model {name!r}")
    return resolved


@dataclass(frozen=True)
class LensProfile:
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    calib_width: int
    calib_height: int
    distortion_model: str

    @classmethod
    def from_intrinsics(
        cls,
        camera_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
        width: int,
        height: int,
        distortion_model: str,
    ) -> "LensProfile":
        model = normalise_model_name(distortion_model)
        coefficients = np.asarray(distortion_coeffs, dtype=np.float64).ravel()
        if model == FISHEYE_MODEL and len(coefficients) < 4:
            coefficients = np.pad(coefficients, (0, 4 - len(coefficients)))
        return cls(
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            distortion_coeffs=coefficients[:4],
            calib_width=int(width),
            calib_height=int(height),
            distortion_model=model,
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "LensProfile":
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise LensProfileError(f"cannot read lens profile {path}: {exc}") from exc

        params = data.get("fisheye_params")
        if params is None:
            raise LensProfileError(f"no fisheye_params in lens profile {path}")

        calib = data.get("calib_dimension", {})
        width, height = calib.get("w"), calib.get("h")
        if not width or not height:
            raise LensProfileError(f"missing calib_dimension in lens profile {path}")

        return cls.from_intrinsics(
            camera_matrix=params["camera_matrix"],
            distortion_coeffs=params["distortion_coeffs"],
            width=width,
            height=height,
            distortion_model=data.get("distortion_model", FISHEYE_MODEL),
        )

    def camera_matrix_for(self, width: int, height: int) -> np.ndarray:
        scale_x = width / self.calib_width
        scale_y = height / self.calib_height
        scaled = self.camera_matrix.copy()
        scaled[0, :] *= scale_x
        scaled[1, :] *= scale_y
        return scaled

    def undistort_to_bearings(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        """Pixel coordinates (N, 2) -> unit bearing vectors (N, 3) in the camera frame."""
        camera_matrix = self.camera_matrix_for(width, height)
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        if self.distortion_model == FISHEYE_MODEL:
            normalized = cv2.fisheye.undistortPoints(pts, camera_matrix, self.distortion_coeffs)
        else:
            normalized = cv2.undistortPoints(pts, camera_matrix, self.distortion_coeffs)
        normalized = normalized.reshape(-1, 2)
        bearings = np.column_stack([normalized, np.ones(len(normalized))])
        return bearings / np.linalg.norm(bearings, axis=1, keepdims=True)

    def project_bearings(self, bearings: np.ndarray, width: int, height: int) -> np.ndarray:
        """Unit bearings (N, 3) -> pixel coordinates (N, 2) at the given resolution."""
        camera_matrix = self.camera_matrix_for(width, height)
        points = bearings.reshape(-1, 1, 3).astype(np.float64)
        zeros = np.zeros(3)
        if self.distortion_model == FISHEYE_MODEL:
            projected, _ = cv2.fisheye.projectPoints(
                points, zeros, zeros, camera_matrix, self.distortion_coeffs
            )
        else:
            projected, _ = cv2.projectPoints(
                points, zeros, zeros, camera_matrix, self.distortion_coeffs
            )
        return projected.reshape(-1, 2)
