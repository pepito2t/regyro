import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from regyro.errors import RegyroError


class LensProfileError(RegyroError):
    pass


@dataclass(frozen=True)
class LensProfile:
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    calib_width: int
    calib_height: int
    distortion_model: str

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

        return cls(
            camera_matrix=np.asarray(params["camera_matrix"], dtype=np.float64),
            distortion_coeffs=np.asarray(params["distortion_coeffs"], dtype=np.float64)[:4],
            calib_width=int(width),
            calib_height=int(height),
            distortion_model=data.get("distortion_model", "opencv_fisheye"),
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
        if self.distortion_model != "opencv_fisheye":
            raise LensProfileError(f"unsupported distortion model: {self.distortion_model}")
        camera_matrix = self.camera_matrix_for(width, height)
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        normalized = cv2.fisheye.undistortPoints(pts, camera_matrix, self.distortion_coeffs)
        normalized = normalized.reshape(-1, 2)
        bearings = np.column_stack([normalized, np.ones(len(normalized))])
        return bearings / np.linalg.norm(bearings, axis=1, keepdims=True)
