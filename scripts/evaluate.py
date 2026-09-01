"""Compare estimated gyro against the real gyro embedded in a video.

Usage:
    uv run python scripts/evaluate.py <video> --lens <profile.json>
    uv run python scripts/evaluate.py <video> --lens <profile.json> --method ml --checkpoint m.pt

Reports per-axis RMSE after resolving the time offset and axis mapping, and writes
an overlay plot next to the video.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regyro.classical.video_pipeline import estimate_gyro
from regyro.dataset.sync import estimate_time_offset
from regyro.dataset.telemetry import AXIS_NAMES, find_axis_mapping, load_gyro
from regyro.errors import RegyroError
from regyro.lens_profile import LensProfile


def plot_overlay(
    times: np.ndarray, estimated: np.ndarray, reference: np.ndarray, path: Path
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for axis, ax in enumerate(axes):
        ax.plot(times, reference[:, axis], label="real gyro", linewidth=0.8)
        ax.plot(times, estimated[:, axis], label="estimated", linewidth=0.8, alpha=0.8)
        ax.set_ylabel(f"g{AXIS_NAMES[axis]} (rad/s)")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(path, dpi=120)


def run_estimator(args: argparse.Namespace, profile: LensProfile):
    if args.method == "ml":
        if args.checkpoint is None:
            raise RegyroError("--method ml requires --checkpoint")
        from regyro.model.infer import estimate_gyro_ml

        return estimate_gyro_ml(args.video, profile, args.checkpoint, args.device)
    return estimate_gyro(args.video, profile)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--method", choices=("classical", "ml"), default="classical")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()

    try:
        gyro = load_gyro(args.video)
        profile = LensProfile.from_json(args.lens)
        estimate = run_estimator(args, profile)
    except RegyroError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    fps = estimate.sample_rate_hz
    estimated = estimate.angular_velocities
    frame_times = (np.arange(len(estimated)) + 0.5) / fps

    offset = estimate_time_offset(estimated, gyro, frame_times, fps)
    reference = gyro.resample(frame_times + offset)
    mapping, rmse = find_axis_mapping(reference, estimated)
    reference = mapping.apply(reference)

    print(f"method      : {args.method}")
    print(f"time offset : {offset:+.4f} s")
    print(f"axis mapping: {mapping.to_string()} (IMU -> camera)")
    print(f"overall RMSE: {rmse:.4f} rad/s")
    for axis in range(3):
        axis_rmse = np.sqrt(np.mean((estimated[:, axis] - reference[:, axis]) ** 2))
        magnitude = np.abs(reference[:, axis]).mean()
        print(f"  g{AXIS_NAMES[axis]}: RMSE {axis_rmse:.4f} rad/s (mean |real| {magnitude:.4f})")

    plot_path = args.video.with_suffix(f".{args.method}.png")
    plot_overlay(frame_times, estimated, reference, plot_path)
    print(f"plot        : {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
