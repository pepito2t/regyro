"""Pins the geometric conventions of the roll augmentation.

A sign error here is invisible at runtime but teaches the network the opposite of
the truth for half the training set, so both halves of the chain are asserted:
how cv2 moves image content, and how the target must move to match.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from regyro.model.augment import (
    apply_exposure,
    apply_motion_blur,
    apply_noise,
    apply_prop_mask,
    roll_images,
)

SIZE = 65
CENTER = (SIZE - 1) / 2.0


def azimuth_of_brightest(image: np.ndarray) -> float:
    row, column = np.unravel_index(int(np.argmax(image)), image.shape)
    return float(np.degrees(np.arctan2(row - CENTER, column - CENTER)))


def test_cv2_rotation_moves_azimuth_negatively():
    """cv2's positive angle takes image content from azimuth phi to phi - angle."""
    image = np.zeros((SIZE, SIZE), dtype=np.uint8)
    image[int(CENTER), int(CENTER) + 20] = 255  # azimuth 0 degrees
    target = np.zeros(3)

    (rotated, _), _ = roll_images([image, image], target, angle_deg=30.0)

    np.testing.assert_allclose(azimuth_of_brightest(rotated), -30.0, atol=4.0)


def test_roll_target_uses_negative_z_rotation():
    image = np.zeros((SIZE, SIZE), dtype=np.uint8)
    target = np.array([1.0, 0.0, 0.0])

    _, rotated_target = roll_images([image, image], target, angle_deg=90.0)

    np.testing.assert_allclose(rotated_target, [0.0, -1.0, 0.0], atol=1e-9)


def test_roll_leaves_pure_roll_target_unchanged():
    """Rolling the frames cannot change a rotation that is already about the optical axis."""
    image = np.zeros((SIZE, SIZE), dtype=np.uint8)
    target = np.array([0.0, 0.0, 2.5])

    _, rotated_target = roll_images([image, image], target, angle_deg=37.0)

    np.testing.assert_allclose(rotated_target, target, atol=1e-9)


def test_roll_is_consistent_with_a_synthetic_pure_roll():
    """A camera roll of psi renders as a cv2 rotation of psi; augmenting must preserve it."""
    rng = np.random.default_rng(0)
    frame_a = (rng.random((SIZE, SIZE)) * 255).astype(np.uint8)
    roll_deg = 12.0
    matrix = cv2.getRotationMatrix2D((CENTER, CENTER), roll_deg, 1.0)
    frame_b = cv2.warpAffine(frame_a, matrix, (SIZE, SIZE))
    target = np.array([0.0, 0.0, np.deg2rad(roll_deg)])

    (augmented_a, augmented_b), augmented_target = roll_images([frame_a, frame_b], target, 55.0)

    # The relative motion between the two augmented frames is still the same roll.
    expected_b = cv2.warpAffine(
        augmented_a, cv2.getRotationMatrix2D((CENTER, CENTER), roll_deg, 1.0), (SIZE, SIZE)
    )
    interior = slice(12, SIZE - 12)
    difference = np.abs(
        augmented_b[interior, interior].astype(float) - expected_b[interior, interior].astype(float)
    )
    assert difference.mean() < 20.0
    np.testing.assert_allclose(augmented_target, target, atol=1e-9)


def test_target_rotation_matches_matrix_conjugation():
    """w' = R_z(-alpha) w is what conjugating the camera frame implies."""
    rng = np.random.default_rng(1)
    target = rng.normal(size=3)
    angle = 41.0
    image = np.zeros((SIZE, SIZE), dtype=np.uint8)

    _, rotated_target = roll_images([image, image], target, angle)

    expected = Rotation.from_rotvec([0.0, 0.0, -np.deg2rad(angle)]).apply(target)
    np.testing.assert_allclose(rotated_target, expected, atol=1e-12)


def test_photometric_augmentations_preserve_shape_and_dtype():
    rng = np.random.default_rng(2)
    frames = [(rng.random((SIZE, SIZE)) * 255).astype(np.uint8) for _ in range(2)]

    for augmentation in (apply_exposure, apply_motion_blur, apply_noise, apply_prop_mask):
        result = augmentation(frames, rng)
        assert len(result) == 2
        for frame in result:
            assert frame.shape == (SIZE, SIZE)
            assert frame.dtype == np.uint8


def test_prop_mask_blacks_out_some_border():
    rng = np.random.default_rng(3)
    frames = [np.full((SIZE, SIZE), 255, dtype=np.uint8)]

    masked = apply_prop_mask(frames, rng)[0]

    assert (masked == 0).any()
    assert masked[int(CENTER), int(CENTER)] == 255
