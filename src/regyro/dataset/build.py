import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from regyro.dataset.canonical import CanonicalProjector
from regyro.dataset.frame_source import FrameSource, FrameSourceError, VideoFrameSource
from regyro.dataset.sync import SyncCalibration, SyncError, calibrate_video
from regyro.dataset.telemetry import (
    GyroSamples,
    TelemetryError,
    integrate_between_frames,
    load_gyro,
)
from regyro.dataset.provenance import OWN_FOOTAGE_LICENSE
from regyro.errors import RegyroError
from regyro.lens_profile import LensProfile

logger = logging.getLogger(__name__)

FRAMES_PER_SHARD = 128
MAX_SYNC_RMSE_RAD_S = 1.5
MIN_COVERAGE = 0.05


class DatasetError(RegyroError):
    pass


@dataclass
class ShardStats:
    shards: int
    frame_pairs: int
    skipped_videos: int = 0


def _write_shard(
    path: Path,
    frames: list[np.ndarray],
    targets: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        frames=np.stack(frames).astype(np.uint8),
        angular_velocity=targets.astype(np.float32),
        valid_mask=valid_mask,
    )


def _flush_shard(
    output_dir: Path,
    name: str,
    index: int,
    frames: list[np.ndarray],
    timestamps: list[float],
    gyro: GyroSamples,
    time_offset_s: float,
    valid_mask: np.ndarray,
) -> int:
    targets = integrate_between_frames(gyro, np.asarray(timestamps), time_offset_s)
    _write_shard(output_dir / f"{name}-{index:05d}.npz", frames, targets, valid_mask)
    return len(targets)


def build_from_source(
    source: FrameSource,
    profile: LensProfile,
    gyro: GyroSamples,
    output_dir: Path,
    time_offset_s: float = 0.0,
) -> ShardStats:
    """Turn a timestamped frame stream plus its gyro into training shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gyro_start, gyro_end = gyro.duration_s

    projector: CanonicalProjector | None = None
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    shard_index, total_pairs, dropped = 0, 0, 0

    for frame in source:
        # Frames outside the gyro span would silently take a flat extrapolated target.
        shifted = frame.timestamp_s + time_offset_s
        if not gyro_start <= shifted <= gyro_end:
            dropped += 1
            continue
        if projector is None:
            projector = CanonicalProjector.build(
                profile, frame.gray.shape[1], frame.gray.shape[0]
            )
            logger.info("%s: canonical coverage %.1f%%", source.name, projector.coverage * 100)
            if projector.coverage < MIN_COVERAGE:
                raise DatasetError(
                    f"{source.name}: lens covers only {projector.coverage:.1%} of the "
                    "canonical field of view; check the calibration"
                )

        frames.append(projector.remap(frame.gray))
        timestamps.append(frame.timestamp_s)

        if len(frames) == FRAMES_PER_SHARD:
            total_pairs += _flush_shard(
                output_dir,
                source.name,
                shard_index,
                frames,
                timestamps,
                gyro,
                time_offset_s,
                projector.mask_image(),
            )
            shard_index += 1
            # Carry the last frame so pairs straddling shard boundaries are not lost.
            frames, timestamps = frames[-1:], timestamps[-1:]

    if projector is not None and len(frames) >= 2:
        total_pairs += _flush_shard(
            output_dir,
            source.name,
            shard_index,
            frames,
            timestamps,
            gyro,
            time_offset_s,
            projector.mask_image(),
        )
        shard_index += 1

    if dropped:
        logger.warning("%s: dropped %d frames outside the gyro span", source.name, dropped)
    if total_pairs == 0:
        raise DatasetError(f"{source.name}: produced no usable frame pair")
    return ShardStats(shards=shard_index, frame_pairs=total_pairs)


def build_from_video(
    video_path: Path,
    profile: LensProfile,
    output_dir: Path,
    calibration: SyncCalibration | None = None,
) -> ShardStats:
    """Build shards from a camera file whose gyro is embedded in the container."""
    if calibration is None:
        calibration = calibrate_video(video_path, profile)
    if calibration.rmse_rad_s > MAX_SYNC_RMSE_RAD_S:
        raise DatasetError(
            f"{video_path.name}: sync rmse {calibration.rmse_rad_s:.3f} rad/s exceeds "
            f"{MAX_SYNC_RMSE_RAD_S}; refusing to emit mislabelled data"
        )

    gyro = load_gyro(video_path).remap_axes(calibration.axis_mapping)
    source = VideoFrameSource(video_path)
    return build_from_source(source, profile, gyro, output_dir, calibration.time_offset_s)


def build_dataset(
    videos: list[Path],
    profile: LensProfile,
    output_dir: Path,
    license_name: str = OWN_FOOTAGE_LICENSE,
) -> ShardStats:
    """Build shards for every video, skipping the ones that cannot be synchronised."""
    total = ShardStats(shards=0, frame_pairs=0, skipped_videos=0)
    manifest = []

    for video in videos:
        try:
            stats = build_from_video(video, profile, output_dir)
        except (DatasetError, SyncError, TelemetryError, FrameSourceError) as exc:
            logger.warning("skipping %s: %s", video.name, exc)
            total.skipped_videos += 1
            continue
        total.shards += stats.shards
        total.frame_pairs += stats.frame_pairs
        manifest.append({"source": video.name, "license": license_name, **asdict(stats)})
        logger.info("%s: %d shards, %d pairs", video.name, stats.shards, stats.frame_pairs)

    append_manifest(output_dir, manifest)
    return total


def append_manifest(output_dir: Path, entries: list[dict]) -> None:
    """Keep a running record so imports from several datasets accumulate."""
    path = output_dir / "manifest.json"
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("manifest at %s was unreadable and has been replaced", path)
    path.write_text(json.dumps([*existing, *entries], indent=2))
