import argparse
import logging
import sys
from pathlib import Path

from regyro.classical.video_pipeline import estimate_gyro
from regyro.dataset.build import DatasetError, build_dataset
from regyro.dataset.sync import calibrate_video
from regyro.errors import RegyroError
from regyro.gcsv import write_gcsv
from regyro.lens_profile import LensProfile


class MissingTorchError(RegyroError):
    pass


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise MissingTorchError(
            "this command needs PyTorch: install it with `uv sync --extra ml`"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regyro", description="Reconstruct gyro data from video for Gyroflow"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Log progress details")
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", help="Estimate gyro data from a video file")
    estimate.add_argument("video", type=Path)
    estimate.add_argument("--lens", type=Path, required=True, help="Gyroflow lens profile JSON")
    estimate.add_argument("--output", "-o", type=Path, help="Output .gcsv (default: next to video)")
    estimate.add_argument("--method", choices=("classical", "ml"), default="classical")
    estimate.add_argument("--checkpoint", type=Path, help="Trained model, required for --method ml")
    estimate.add_argument("--device", help="torch device override (cuda, mps, cpu)")

    calibrate = subparsers.add_parser(
        "calibrate", help="Report the IMU time offset and axis mapping of a video"
    )
    calibrate.add_argument("video", type=Path)
    calibrate.add_argument("--lens", type=Path, required=True)

    build = subparsers.add_parser(
        "build-dataset", help="Turn gyro-bearing videos into training shards"
    )
    build.add_argument("videos", type=Path, nargs="+")
    build.add_argument("--lens", type=Path, required=True)
    build.add_argument("--output", "-o", type=Path, required=True, help="Shard output directory")

    train = subparsers.add_parser("train", help="Train the rotation model on built shards")
    train.add_argument("--data", type=Path, required=True, help="Shard directory")
    train.add_argument("--output", "-o", type=Path, required=True, help="Checkpoint path")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--device")
    return parser


def run_estimate(args: argparse.Namespace) -> int:
    profile = LensProfile.from_json(args.lens)
    if args.method == "ml":
        if args.checkpoint is None:
            raise DatasetError("--method ml requires --checkpoint")
        _require_torch()
        from regyro.model.infer import estimate_gyro_ml

        estimate = estimate_gyro_ml(args.video, profile, args.checkpoint, args.device)
    else:
        estimate = estimate_gyro(args.video, profile)

    output = args.output or args.video.with_suffix(".gcsv")
    write_gcsv(
        output,
        estimate.angular_velocities,
        estimate.sample_rate_hz,
        video_filename=args.video.name,
    )
    print(
        f"wrote {output} ({len(estimate.angular_velocities)} samples @ "
        f"{estimate.sample_rate_hz:.2f} Hz, {estimate.failed_pairs} interpolated)"
    )
    return 0


def run_calibrate(args: argparse.Namespace) -> int:
    profile = LensProfile.from_json(args.lens)
    calibration = calibrate_video(args.video, profile)
    print(f"time offset : {calibration.time_offset_s:+.4f} s")
    print(f"axis mapping: {calibration.axis_mapping.to_string()} (IMU -> camera)")
    print(f"residual    : {calibration.rmse_rad_s:.4f} rad/s")
    return 0


def run_build_dataset(args: argparse.Namespace) -> int:
    profile = LensProfile.from_json(args.lens)
    stats = build_dataset(args.videos, profile, args.output)
    print(
        f"{stats.shards} shards, {stats.frame_pairs} frame pairs, "
        f"{stats.skipped_videos} videos skipped -> {args.output}"
    )
    return 0 if stats.frame_pairs else 1


def run_train(args: argparse.Namespace) -> int:
    _require_torch()
    from regyro.model.train import TrainConfig, train

    config = TrainConfig(
        data_dir=args.data,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        workers=args.workers,
        device=args.device,
    )
    train(config)
    return 0


COMMANDS = {
    "estimate": run_estimate,
    "calibrate": run_calibrate,
    "build-dataset": run_build_dataset,
    "train": run_train,
}


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        return COMMANDS[args.command](args)
    except RegyroError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
