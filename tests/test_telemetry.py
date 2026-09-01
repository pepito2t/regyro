import numpy as np
import pytest

from regyro.dataset.telemetry import (
    AxisMapping,
    GyroSamples,
    TelemetryError,
    find_axis_mapping,
    integrate_between_frames,
)


def test_axis_mapping_string_round_trip():
    mapping = AxisMapping((1, 0, 2), (1.0, -1.0, 1.0))
    assert mapping.to_string() == "y,-x,z"
    assert AxisMapping.from_string("y,-x,z") == mapping


def test_axis_mapping_rejects_incomplete():
    with pytest.raises(TelemetryError):
        AxisMapping.from_string("x,x,z")


def test_axis_mapping_inverse_round_trips():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(20, 3))
    mapping = AxisMapping((2, 0, 1), (-1.0, 1.0, -1.0))

    restored = mapping.inverse().apply(mapping.apply(values))

    np.testing.assert_allclose(restored, values)


def test_find_axis_mapping_recovers_known_mapping():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(200, 3))
    mapping = AxisMapping((2, 1, 0), (1.0, -1.0, -1.0))

    found, rmse = find_axis_mapping(source, mapping.apply(source))

    assert found == mapping
    assert rmse < 1e-12


def test_integrate_constant_rate_returns_that_rate():
    """A constant rotation about a fixed axis integrates back to itself exactly."""
    rate = np.array([0.1, -0.25, 0.4])
    timestamps = np.linspace(0.0, 2.0, 2001)
    gyro = GyroSamples(timestamps, np.tile(rate, (len(timestamps), 1)))
    frame_times = np.arange(0.0, 1.0, 1.0 / 30.0)

    result = integrate_between_frames(gyro, frame_times)

    assert result.shape == (len(frame_times) - 1, 3)
    np.testing.assert_allclose(result, np.tile(rate, (len(result), 1)), atol=1e-6)


def test_integrate_applies_time_offset():
    timestamps = np.linspace(0.0, 2.0, 2001)
    ramp = np.column_stack(
        [timestamps, np.zeros_like(timestamps), np.zeros_like(timestamps)]
    )
    gyro = GyroSamples(timestamps, ramp)
    frame_times = np.arange(0.0, 1.0, 1.0 / 30.0)

    shifted = integrate_between_frames(gyro, frame_times, time_offset_s=0.5)
    unshifted = integrate_between_frames(gyro, frame_times)

    np.testing.assert_allclose(shifted[:, 0] - unshifted[:, 0], 0.5, atol=1e-6)


def test_integrate_needs_two_timestamps():
    gyro = GyroSamples(np.array([0.0, 1.0]), np.zeros((2, 3)))
    with pytest.raises(TelemetryError):
        integrate_between_frames(gyro, np.array([0.0]))
