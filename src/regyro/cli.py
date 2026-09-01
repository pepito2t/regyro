import argparse
import logging
import sys
from pathlib import Path

from regyro.classical.video_pipeline import VideoError, estimate_gyro
from regyro.gcsv import write_gcsv
from regyro.lens_profile import LensProfile, LensProfileError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regyro", description="Reconstruct gyro data from video for Gyroflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", help="Estimate gyro data from a video file")
    estimate.add_argument("video", type=Path, help="Input video file")
    estimate.add_argument("--lens", type=Path, required=True, help="Gyroflow lens profile JSON")
    estimate.add_argument("--output", "-o", type=Path, help="Output .gcsv path (default: next to video)")
    return parser


def run_estimate(args: argparse.Namespace) -> int:
    output = args.output or args.video.with_suffix(".gcsv")
    try:
        profile = LensProfile.from_json(args.lens)
        estimate = estimate_gyro(args.video, profile)
    except (LensProfileError, VideoError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_gcsv(
        output,
        estimate.angular_velocities,
        estimate.sample_rate_hz,
        video_filename=args.video.name,
    )
    total = len(estimate.angular_velocities)
    print(f"wrote {output} ({total} samples @ {estimate.sample_rate_hz:.2f} Hz, {estimate.failed_pairs} interpolated)")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.command == "estimate":
        return run_estimate(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
