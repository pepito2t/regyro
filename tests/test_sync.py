import numpy as np
import pytest

from regyro.dataset.sync import SyncError, estimate_time_offset
from regyro.dataset.telemetry import GyroSamples

FPS = 60.0
DURATION_S = 20.0


def build_motion(rng: np.random.Generator, times: np.ndarray) -> np.ndarray:
    """Smooth, bursty motion resembling a flight rather than white noise."""
    raw = rng.normal(size=(len(times), 3))
    kernel = np.ones(15) / 15.0
    smoothed = np.column_stack([np.convolve(raw[:, axis], kernel, mode="same") for axis in range(3)])
    envelope = 1.0 + np.sin(2.0 * np.pi * times / 4.0) ** 2
    return smoothed * envelope[:, None] * 4.0


@pytest.mark.parametrize("true_offset", [0.0, 0.25, -0.4])
def test_estimate_time_offset_recovers_known_shift(true_offset):
    rng = np.random.default_rng(0)
    dense_times = np.arange(0.0, DURATION_S, 1.0 / 1000.0)
    dense_motion = build_motion(rng, dense_times)
    gyro = GyroSamples(dense_times, dense_motion)

    frame_times = np.arange(1.0, DURATION_S - 3.0, 1.0 / FPS)
    # The "estimated" signal is the gyro read at the shifted times.
    estimated = gyro.resample(frame_times + true_offset)

    found = estimate_time_offset(estimated, gyro, frame_times, FPS)

    assert found == pytest.approx(true_offset, abs=1.5 / FPS)


def test_estimate_time_offset_rejects_static_footage():
    times = np.arange(0.0, 5.0, 1.0 / 1000.0)
    gyro = GyroSamples(times, np.zeros((len(times), 3)))
    frame_times = np.arange(0.0, 3.0, 1.0 / FPS)
    estimated = np.zeros((len(frame_times), 3))

    with pytest.raises(SyncError):
        estimate_time_offset(estimated, gyro, frame_times, FPS)
