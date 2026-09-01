import json

import numpy as np
import pytest

from regyro.dataset.canonical import CANONICAL_SIZE, CanonicalProjector
from regyro.lens_profile import LensProfile


@pytest.fixture
def profile(tmp_path) -> LensProfile:
    data = {
        "calib_dimension": {"w": 1920, "h": 1080},
        "fisheye_params": {
            "camera_matrix": [[600.0, 0.0, 960.0], [0.0, 600.0, 540.0], [0.0, 0.0, 1.0]],
            "distortion_coeffs": [0.02, -0.005, 0.001, 0.0],
        },
    }
    path = tmp_path / "lens.json"
    path.write_text(json.dumps(data))
    return LensProfile.from_json(path)


def test_projector_maps_centre_to_optical_centre(profile):
    projector = CanonicalProjector.build(profile, 1920, 1080)
    middle = CANONICAL_SIZE // 2

    assert projector.map_x[middle, middle] == pytest.approx(960.0, abs=3.0)
    assert projector.map_y[middle, middle] == pytest.approx(540.0, abs=3.0)


def test_projector_masks_square_corners(profile):
    projector = CanonicalProjector.build(profile, 1920, 1080)

    assert not projector.valid_mask[0, 0]
    assert projector.valid_mask[CANONICAL_SIZE // 2, CANONICAL_SIZE // 2]


def test_remap_produces_canonical_shape(profile):
    rng = np.random.default_rng(0)
    frame = (rng.random((1080, 1920)) * 255).astype(np.uint8)
    projector = CanonicalProjector.build(profile, 1920, 1080)

    remapped = projector.remap(frame)

    assert remapped.shape == (CANONICAL_SIZE, CANONICAL_SIZE)
    assert remapped.dtype == np.uint8
    assert remapped[0, 0] == 0  # masked corner


def test_projector_is_rotationally_symmetric_about_the_axis(profile):
    """Equal angles from the axis must land at equal radii, whatever the azimuth."""
    projector = CanonicalProjector.build(profile, 1920, 1080)
    centre = (CANONICAL_SIZE - 1) / 2.0
    # Pixel centres straddle the true centre by half a pixel, so the sample points are
    # picked to sit at one shared canonical radius rather than on naive +/- offsets.
    low, high = int(centre - 0.5), int(centre + 0.5)
    offset = 60
    samples = [
        (low, low + offset),
        (low + offset, low),
        (high, high - offset),
        (high - offset, high),
    ]
    canonical_radii = {
        round(float(np.hypot(column - centre, row - centre)), 6) for row, column in samples
    }
    assert len(canonical_radii) == 1

    radii = [
        np.hypot(projector.map_x[row, column] - 960.0, projector.map_y[row, column] - 540.0)
        for row, column in samples
    ]
    assert np.std(radii) < 0.5
