"""Compare estimated gyro against the real gyro embedded in a video.

Usage: uv run python scripts/evaluate.py <video-with-gyro> --lens <profile.json>

Searches all axis permutations/signs to find the best mapping between the
estimated camera-frame rates and the IMU frame, then reports per-axis RMSE
and writes an overlay plot next to the video.
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import telemetry_parser

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regyro.classical.video_pipeline import estimate_gyro
from regyro.lens_profile import LensProfile

AXIS_NAMES = ("x", "y", "z")


def extract_real_gyro(video_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_s, angular_velocities rad/s) from embedded telemetry."""
    samples = telemetry_parser.Parser(str(video_path)).normalized_imu()
    timestamps = np.array([s["timestamp_ms"] for s in samples]) / 1000.0
    gyro_deg = np.array([s["gyro"] for s in samples], dtype=np.float64)
    return timestamps, np.deg2rad(gyro_deg)


def resample_to(timestamps: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(target_times, timestamps, values[:, axis]) for axis in range(3)]
    )


def best_axis_mapping(estimated: np.ndarray, reference: np.ndarray) -> tuple[str, np.ndarray]:
    """Find the permutation + signs of estimated axes that best matches the reference."""
    best_rmse, best_label, best_mapped = np.inf, "", estimated
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            mapped = estimated[:, perm] * np.array(signs)
            rmse = float(np.sqrt(np.mean((mapped - reference) ** 2)))
            if rmse < best_rmse:
                label = ",".join(
                    f"{'-' if s < 0 else ''}{AXIS_NAMES[p]}" for p, s in zip(perm, signs)
                )
                best_rmse, best_label, best_mapped = rmse, label, mapped
    return best_label, best_mapped


def plot_overlay(times: np.ndarray, estimated: np.ndarray, reference: np.ndarray, path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for axis, ax in enumerate(axes):
        ax.plot(times, reference[:, axis], label="real", linewidth=0.8)
        ax.plot(times, estimated[:, axis], label="estimated", linewidth=0.8, alpha=0.8)
        ax.set_ylabel(f"g{AXIS_NAMES[axis]} (rad/s)")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(path, dpi=120)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--lens", type=Path, required=True)
    args = parser.parse_args()

    real_times, real_gyro = extract_real_gyro(args.video)
    if len(real_times) == 0:
        print("error: no gyro telemetry found in video", file=sys.stderr)
        return 1

    profile = LensProfile.from_json(args.lens)
    estimate = estimate_gyro(args.video, profile)
    frame_times = np.arange(len(estimate.angular_velocities)) / estimate.sample_rate_hz

    reference = resample_to(real_times, real_gyro, frame_times)
    mapping, mapped = best_axis_mapping(estimate.angular_velocities, reference)

    print(f"best axis mapping (estimated -> IMU): {mapping}")
    for axis in range(3):
        rmse = np.sqrt(np.mean((mapped[:, axis] - reference[:, axis]) ** 2))
        print(f"  g{AXIS_NAMES[axis]}: RMSE {rmse:.4f} rad/s")

    plot_path = args.video.with_suffix(".eval.png")
    plot_overlay(frame_times, mapped, reference, plot_path)
    print(f"plot: {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
