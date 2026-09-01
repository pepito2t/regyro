import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from regyro.classical.video_pipeline import VideoError, read_frame_rate
from regyro.dataset.canonical import CanonicalProjector
from regyro.dataset.sync import SyncCalibration, SyncError, calibrate_video
from regyro.dataset.telemetry import (
    GyroSamples,
    TelemetryError,
    integrate_between_frames,
    load_gyro,
)
from regyro.lens_profile import LensProfile
from regyro.errors import RegyroError

logger = logging.getLogger(__name__)

FRAMES_PER_SHARD = 128
MAX_SYNC_RMSE_RAD_S = 1.5


class DatasetError(RegyroError):
    pass


@dataclass
class ShardStats:
    shards: int
    frame_pairs: int
    skipped_videos: int


def _shard_paths(output_dir: Path, stem: str, index: int) -> Path:
    return output_dir / f"{stem}-{index:05d}.npz"


def _write_shard(path: Path, frames: list[np.ndarray], targets: np.ndarray) -> None:
    np.savez_compressed(
        path,
        frames=np.stack(frames).astype(np.uint8),
        angular_velocity=targets.astype(np.float32),
    )


def build_from_video(
    video_path: Path,
    profile: LensProfile,
    output_dir: Path,
    calibration: SyncCalibration | None = None,
) -> ShardStats:
    """Turn one gyro-bearing video into training shards of canonical frames + targets."""
    if calibration is None:
        calibration = calibrate_video(video_path, profile)
    if calibration.rmse_rad_s > MAX_SYNC_RMSE_RAD_S:
        raise DatasetError(
            f"{video_path.name}: sync rmse {calibration.rmse_rad_s:.3f} rad/s exceeds "
            f"{MAX_SYNC_RMSE_RAD_S}; refusing to emit mislabelled data"
        )

    gyro = load_gyro(video_path).remap_axes(calibration.axis_mapping)
    fps = read_frame_rate(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoError(f"cannot open video {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    projector: CanonicalProjector | None = None
    buffer: list[np.ndarray] = []
    shard_index, total_pairs = 0, 0
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if projector is None:
                projector = CanonicalProjector.build(profile, gray.shape[1], gray.shape[0])
            buffer.append(projector.remap(gray))
            frame_index += 1

            if len(buffer) == FRAMES_PER_SHARD:
                start = frame_index - FRAMES_PER_SHARD
                targets = _targets_for(gyro, start, len(buffer), fps, calibration.time_offset_s)
                _write_shard(_shard_paths(output_dir, stem, shard_index), buffer, targets)
                shard_index += 1
                total_pairs += len(targets)
                # Carry the last frame so pairs straddling shard boundaries are not lost.
                buffer = buffer[-1:]

        if len(buffer) >= 2:
            start = frame_index - len(buffer)
            targets = _targets_for(gyro, start, len(buffer), fps, calibration.time_offset_s)
            _write_shard(_shard_paths(output_dir, stem, shard_index), buffer, targets)
            shard_index += 1
            total_pairs += len(targets)
    finally:
        capture.release()

    if total_pairs == 0:
        raise DatasetError(f"{video_path.name}: too short to produce any frame pair")
    return ShardStats(shards=shard_index, frame_pairs=total_pairs, skipped_videos=0)


def _targets_for(
    gyro: GyroSamples,
    start_frame: int,
    frame_count: int,
    fps: float,
    time_offset_s: float,
) -> np.ndarray:
    frame_times = (np.arange(start_frame, start_frame + frame_count)) / fps
    return integrate_between_frames(gyro, frame_times, time_offset_s)


def build_dataset(
    videos: list[Path],
    profile: LensProfile,
    output_dir: Path,
) -> ShardStats:
    """Build shards for every video, skipping the ones that cannot be synchronised."""
    total = ShardStats(shards=0, frame_pairs=0, skipped_videos=0)
    manifest = []

    for video in videos:
        try:
            stats = build_from_video(video, profile, output_dir)
        except (DatasetError, SyncError, TelemetryError, VideoError) as exc:
            logger.warning("skipping %s: %s", video.name, exc)
            total.skipped_videos += 1
            continue
        total.shards += stats.shards
        total.frame_pairs += stats.frame_pairs
        manifest.append({"video": video.name, **asdict(stats)})
        logger.info("%s: %d shards, %d pairs", video.name, stats.shards, stats.frame_pairs)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return total
