from dataclasses import dataclass

import cv2
import numpy as np

from regyro.lens_profile import LensProfile

CANONICAL_SIZE = 256
CANONICAL_FOV_DEG = 140.0


@dataclass(frozen=True)
class CanonicalProjector:
    """Remaps any lens onto a fixed equidistant fisheye, so the model sees one geometry.

    Without this, focal length and distortion differ per camera and the same pixel
    motion would mean different rotations across the dataset.
    """

    map_x: np.ndarray
    map_y: np.ndarray
    valid_mask: np.ndarray
    size: int

    @classmethod
    def build(cls, profile: LensProfile, width: int, height: int) -> "CanonicalProjector":
        bearings, valid_mask = _canonical_bearings(CANONICAL_SIZE, CANONICAL_FOV_DEG)
        camera_matrix = profile.camera_matrix_for(width, height)

        projected, _ = cv2.fisheye.projectPoints(
            bearings.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            profile.distortion_coeffs,
        )
        projected = projected.reshape(CANONICAL_SIZE, CANONICAL_SIZE, 2)
        map_x = np.where(valid_mask, projected[..., 0], -1.0).astype(np.float32)
        map_y = np.where(valid_mask, projected[..., 1], -1.0).astype(np.float32)
        return cls(map_x, map_y, valid_mask, CANONICAL_SIZE)

    def remap(self, gray: np.ndarray) -> np.ndarray:
        return cv2.remap(
            gray,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )


def _canonical_bearings(size: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit bearing per output pixel under an equidistant model, plus its validity mask."""
    center = (size - 1) / 2.0
    max_theta = np.deg2rad(fov_deg) / 2.0
    focal = center / max_theta

    grid = np.arange(size, dtype=np.float64) - center
    dx, dy = np.meshgrid(grid, grid)
    radius = np.hypot(dx, dy)
    # Only the inscribed circle is kept: square corners would exceed 90° and fold behind.
    valid_mask = radius <= center

    theta = np.clip(radius / focal, 0.0, max_theta)
    phi = np.arctan2(dy, dx)
    bearings = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=-1,
    )
    return bearings, valid_mask
