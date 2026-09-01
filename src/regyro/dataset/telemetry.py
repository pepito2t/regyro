import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import telemetry_parser
from scipy.spatial.transform import Rotation
from regyro.errors import RegyroError

AXIS_NAMES = ("x", "y", "z")
INTEGRATION_STEPS_PER_FRAME = 16


class TelemetryError(RegyroError):
    pass


@dataclass(frozen=True)
class AxisMapping:
    """Maps one 3-axis frame onto another by permuting and flipping axes."""

    permutation: tuple[int, int, int]
    signs: tuple[float, float, float]

    def apply(self, values: np.ndarray) -> np.ndarray:
        return values[:, list(self.permutation)] * np.array(self.signs)

    def inverse(self) -> "AxisMapping":
        order = np.argsort(self.permutation)
        return AxisMapping(
            permutation=tuple(order.tolist()),
            signs=tuple(np.array(self.signs)[order].tolist()),
        )

    def to_string(self) -> str:
        return ",".join(
            f"{'-' if sign < 0 else ''}{AXIS_NAMES[axis]}"
            for axis, sign in zip(self.permutation, self.signs)
        )

    @classmethod
    def from_string(cls, text: str) -> "AxisMapping":
        permutation, signs = [], []
        for token in text.split(","):
            token = token.strip()
            sign = -1.0 if token.startswith("-") else 1.0
            name = token.lstrip("+-")
            if name not in AXIS_NAMES:
                raise TelemetryError(f"invalid axis token {token!r} in mapping {text!r}")
            permutation.append(AXIS_NAMES.index(name))
            signs.append(sign)
        if len(permutation) != 3 or set(permutation) != {0, 1, 2}:
            raise TelemetryError(f"mapping {text!r} must permute all three axes")
        return cls(tuple(permutation), tuple(signs))

    @classmethod
    def identity(cls) -> "AxisMapping":
        return cls((0, 1, 2), (1.0, 1.0, 1.0))


@dataclass(frozen=True)
class GyroSamples:
    """Angular velocity samples in rad/s with timestamps in seconds."""

    timestamps: np.ndarray
    angular_velocity: np.ndarray

    def resample(self, target_times: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.interp(target_times, self.timestamps, self.angular_velocity[:, axis])
                for axis in range(3)
            ]
        )

    def remap_axes(self, mapping: AxisMapping) -> "GyroSamples":
        return GyroSamples(self.timestamps, mapping.apply(self.angular_velocity))

    def rotate(self, rotation_matrix: np.ndarray) -> "GyroSamples":
        """Express the rates in another frame.

        Camera-to-IMU extrinsics from public datasets are arbitrary rotations, not
        the axis swaps that action cameras happen to use.
        """
        matrix = np.asarray(rotation_matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise TelemetryError(f"expected a 3x3 rotation matrix, got {matrix.shape}")
        return GyroSamples(self.timestamps, self.angular_velocity @ matrix.T)

    @property
    def duration_s(self) -> tuple[float, float]:
        return float(self.timestamps[0]), float(self.timestamps[-1])


def load_gyro(video_path: Path | str) -> GyroSamples:
    """Read embedded IMU telemetry (GoPro GPMF, DJI, Insta360, ...) from a video."""
    try:
        samples = telemetry_parser.Parser(str(video_path)).normalized_imu()
    except Exception as exc:  # telemetry_parser raises bare exceptions from Rust
        raise TelemetryError(f"cannot read telemetry from {video_path}: {exc}") from exc

    usable = [s for s in samples if s.get("gyro") is not None]
    if not usable:
        raise TelemetryError(f"no gyro samples found in {video_path}")

    timestamps = np.array([s["timestamp_ms"] for s in usable], dtype=np.float64) / 1000.0
    gyro_deg = np.array([s["gyro"] for s in usable], dtype=np.float64)
    return GyroSamples(timestamps, np.deg2rad(gyro_deg))


def find_axis_mapping(source: np.ndarray, target: np.ndarray) -> tuple[AxisMapping, float]:
    """Brute-force the axis permutation and signs taking `source` closest to `target`."""
    if source.shape != target.shape:
        raise TelemetryError(f"shape mismatch: {source.shape} vs {target.shape}")

    best_rmse, best_mapping = np.inf, AxisMapping.identity()
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            candidate = AxisMapping(permutation, signs)
            rmse = float(np.sqrt(np.mean((candidate.apply(source) - target) ** 2)))
            if rmse < best_rmse:
                best_rmse, best_mapping = rmse, candidate
    return best_mapping, best_rmse


def integrate_between_frames(
    gyro: GyroSamples,
    frame_times: np.ndarray,
    time_offset_s: float = 0.0,
) -> np.ndarray:
    """Mean angular velocity (rad/s) over each frame interval, as a rotation vector.

    Composes sub-steps rather than summing, because summing angular velocity vectors
    ignores the non-commutativity of rotation and biases fast rolls.
    """
    if len(frame_times) < 2:
        raise TelemetryError("need at least two frame timestamps")

    starts = frame_times[:-1] + time_offset_s
    durations = np.diff(frame_times)
    if np.any(durations <= 0):
        raise TelemetryError("frame timestamps must be strictly increasing")
    steps = INTEGRATION_STEPS_PER_FRAME
    step_durations = durations / steps

    offsets = (np.arange(steps) + 0.5) / steps
    sub_times = starts[:, None] + durations[:, None] * offsets[None, :]
    sub_velocity = np.stack(
        [gyro.resample(sub_times[:, k]) for k in range(steps)], axis=1
    )

    accumulated = Rotation.identity(len(starts))
    for k in range(steps):
        accumulated = accumulated * Rotation.from_rotvec(
            sub_velocity[:, k, :] * step_durations[:, None]
        )
    return accumulated.as_rotvec() / durations[:, None]
