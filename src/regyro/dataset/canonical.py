from dataclasses import dataclass

import cv2
import numpy as np

from regyro.lens_profile import LensProfile

CANONICAL_SIZE = 256
CANONICAL_FOV_DEG = 140.0
# Bearings this far off-axis project unreliably, and for pinhole lenses they fold
# back into the image and would be sampled as if they were real content.
MAX_BEARING_ANGLE_DEG = 89.0


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
        bearings, in_circle = _canonical_bearings(CANONICAL_SIZE, CANONICAL_FOV_DEG)
        projected = profile.project_bearings(bearings.reshape(-1, 3), width, height)
        projected = projected.reshape(CANONICAL_SIZE, CANONICAL_SIZE, 2)

        # A source pixel must exist and be in front of the camera to be usable.
        in_bounds = (
            (projected[..., 0] >= 0)
            & (projected[..., 0] <= width - 1)
            & (projected[..., 1] >= 0)
            & (projected[..., 1] <= height - 1)
        )
        in_front = bearings[..., 2] > np.cos(np.deg2rad(MAX_BEARING_ANGLE_DEG))
        valid_mask = in_circle & in_bounds & in_front & np.isfinite(projected).all(axis=-1)

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

    def mask_image(self) -> np.ndarray:
        """The validity mask as a uint8 plane, stored alongside frames as a model input."""
        return (self.valid_mask * 255).astype(np.uint8)

    @property
    def coverage(self) -> float:
        return float(self.valid_mask.mean())


def _canonical_bearings(size: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit bearing per output pixel under an equidistant model, plus its validity mask."""
    center = (size - 1) / 2.0
    max_theta = np.deg2rad(fov_deg) / 2.0
    focal = center / max_theta

    grid = np.arange(size, dtype=np.float64) - center
    dx, dy = np.meshgrid(grid, grid)
    radius = np.hypot(dx, dy)
    # Only the inscribed circle is kept: square corners would exceed the modelled FOV.
    in_circle = radius <= center

    theta = np.clip(radius / focal, 0.0, max_theta)
    phi = np.arctan2(dy, dx)
    bearings = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=-1,
    )
    return bearings, in_circle
