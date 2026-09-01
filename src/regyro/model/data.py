import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from regyro.model.augment import (
    apply_exposure,
    apply_motion_blur,
    apply_noise,
    apply_prop_mask,
    roll_images,
)
from regyro.model.network import TARGET_SCALE_RAD_S
from regyro.errors import RegyroError

logger = logging.getLogger(__name__)

ROLL_PROBABILITY = 0.5
BLUR_PROBABILITY = 0.3
NOISE_PROBABILITY = 0.3
PROP_PROBABILITY = 0.25
EXPOSURE_PROBABILITY = 0.8


class DataError(RegyroError):
    pass


@dataclass(frozen=True)
class ShardSplit:
    train: list[Path]
    validation: list[Path]


def _video_key(shard: Path) -> str:
    return shard.stem.rsplit("-", 1)[0]


def split_shards_by_video(shard_dir: Path, val_fraction: float, seed: int = 0) -> ShardSplit:
    """Hold out whole videos, never individual pairs.

    Consecutive frames are near-duplicates, so a pair-level split would leak the
    validation set into training and report a flattering, meaningless score.
    """
    shards = sorted(shard_dir.glob("*.npz"))
    if not shards:
        raise DataError(f"no shards found in {shard_dir}")

    keys = sorted({_video_key(shard) for shard in shards})
    if len(keys) < 2:
        raise DataError(f"need at least two source videos to split, found {len(keys)}")

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(keys))
    held_out = max(1, int(round(len(keys) * val_fraction)))
    validation_keys = set(shuffled[:held_out])

    return ShardSplit(
        train=[s for s in shards if _video_key(s) not in validation_keys],
        validation=[s for s in shards if _video_key(s) in validation_keys],
    )


def _augment(
    frames: list[np.ndarray],
    mask: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    if rng.random() < ROLL_PROBABILITY:
        rolled, target = roll_images([*frames, mask], target, rng.uniform(-180.0, 180.0))
        frames, mask = rolled[:2], rolled[2]
    # Photometric augmentations touch the frames only: the mask records where sensor
    # data exists, which occlusion and exposure do not change.
    if rng.random() < EXPOSURE_PROBABILITY:
        frames = apply_exposure(frames, rng)
    if rng.random() < BLUR_PROBABILITY:
        frames = apply_motion_blur(frames, rng)
    if rng.random() < PROP_PROBABILITY:
        frames = apply_prop_mask(frames, rng)
    if rng.random() < NOISE_PROBABILITY:
        frames = apply_noise(frames, rng)
    return frames, mask, target


def _to_tensors(
    frames: list[np.ndarray], mask: np.ndarray, target: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    images = np.stack(frames).astype(np.float32) / 127.5 - 1.0
    mask_plane = (mask.astype(np.float32) / 255.0)[None, ...]
    stacked = np.concatenate([images, mask_plane], axis=0)
    return (
        torch.from_numpy(stacked),
        torch.from_numpy((target / TARGET_SCALE_RAD_S).astype(np.float32)),
    )


class ShardDataset(IterableDataset):
    """Streams frame pairs, shuffling shard order and pair order within each shard.

    Shards are read whole because they are compressed; shuffling at two levels gives
    enough mixing without the random-access cost of an uncompressed layout.
    """

    def __init__(self, shards: list[Path], augment: bool, seed: int = 0) -> None:
        if not shards:
            raise DataError("shard list is empty")
        self.shards = shards
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _worker_shards(self) -> list[Path]:
        info = get_worker_info()
        if info is None:
            return list(self.shards)
        return list(self.shards[info.id :: info.num_workers])

    def __iter__(self):
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        rng = np.random.default_rng([self.seed, self.epoch, worker_id])

        shards = self._worker_shards()
        for index in rng.permutation(len(shards)):
            shard = shards[int(index)]
            try:
                with np.load(shard) as payload:
                    frames = payload["frames"]
                    targets = payload["angular_velocity"]
                    mask = payload["valid_mask"]
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("skipping unreadable shard %s: %s", shard.name, exc)
                continue

            for pair in rng.permutation(len(targets)):
                pair = int(pair)
                pair_frames = [frames[pair], frames[pair + 1]]
                pair_mask = mask
                target = targets[pair].astype(np.float64)
                if self.augment:
                    pair_frames, pair_mask, target = _augment(pair_frames, mask, target, rng)
                yield _to_tensors(pair_frames, pair_mask, target)
