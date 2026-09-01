import cv2
import numpy as np
from scipy.spatial.transform import Rotation

MOTION_BLUR_MAX_LEN = 15
BRIGHTNESS_GAIN_RANGE = (0.6, 1.4)
BRIGHTNESS_OFFSET_RANGE = (-40.0, 40.0)
FLICKER_GAIN_RANGE = (0.97, 1.03)
NOISE_SIGMA_MAX = 12.0
PROP_WEDGE_MAX = 3
PROP_WEDGE_MAX_DEG = 50.0


def roll_images(
    images: list[np.ndarray],
    target: np.ndarray,
    angle_deg: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Roll every plane about the optical axis and rotate the target to match.

    Under the canonical equidistant projection a camera roll is exactly an image
    rotation, so this is a geometrically valid way to multiply roll examples.
    cv2 moves content from azimuth phi to phi - angle, which corresponds to
    expressing the same motion in a camera frame rotated by -angle about z.
    The validity mask is rolled with the frames so it keeps describing them.
    """
    size = images[0].shape[0]
    center = ((size - 1) / 2.0, (size - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = [
        cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR) for image in images
    ]
    frame_rotation = Rotation.from_rotvec([0.0, 0.0, -np.deg2rad(angle_deg)])
    return rotated, frame_rotation.apply(target)


def apply_exposure(frames: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    """Shared gain/offset plus a small per-frame flicker, as a real sensor would show."""
    gain = rng.uniform(*BRIGHTNESS_GAIN_RANGE)
    offset = rng.uniform(*BRIGHTNESS_OFFSET_RANGE)
    adjusted = []
    for frame in frames:
        flicker = rng.uniform(*FLICKER_GAIN_RANGE)
        values = frame.astype(np.float32) * gain * flicker + offset
        adjusted.append(np.clip(values, 0, 255).astype(np.uint8))
    return adjusted


def apply_motion_blur(frames: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    length = int(rng.integers(3, MOTION_BLUR_MAX_LEN + 1))
    angle = rng.uniform(0.0, 180.0)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D(((length - 1) / 2.0, (length - 1) / 2.0), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    total = kernel.sum()
    if total <= 0:
        return frames
    kernel /= total
    return [cv2.filter2D(frame, -1, kernel) for frame in frames]


def apply_noise(frames: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    sigma = rng.uniform(1.0, NOISE_SIGMA_MAX)
    return [
        np.clip(frame.astype(np.float32) + rng.normal(0.0, sigma, frame.shape), 0, 255).astype(
            np.uint8
        )
        for frame in frames
    ]


def apply_prop_mask(frames: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    """Black out wedges at the image border, standing in for props in frame."""
    size = frames[0].shape[0]
    center = (size - 1) / 2.0
    grid = np.arange(size) - center
    dx, dy = np.meshgrid(grid, grid)
    azimuth = np.degrees(np.arctan2(dy, dx))
    radius = np.hypot(dx, dy)

    mask = np.ones((size, size), dtype=np.uint8)
    for _ in range(int(rng.integers(1, PROP_WEDGE_MAX + 1))):
        start = rng.uniform(-180.0, 180.0)
        width = rng.uniform(10.0, PROP_WEDGE_MAX_DEG)
        inner = rng.uniform(0.55, 0.9) * center
        delta = np.abs(((azimuth - start + 180.0) % 360.0) - 180.0)
        mask[(delta < width / 2.0) & (radius > inner)] = 0
    return [frame * mask for frame in frames]
