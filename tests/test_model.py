import numpy as np
import pytest
import torch

from regyro.model.data import DataError, ShardDataset, split_shards_by_video
from regyro.model.network import INPUT_CHANNELS, RotationNet, count_parameters
from regyro.model.train import TrainConfig, amp_is_worthwhile, select_device, train

FRAME_SIZE = 64
FRAMES_PER_SHARD = 12


def write_shard(directory, stem: str, index: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    frames = (rng.random((FRAMES_PER_SHARD, FRAME_SIZE, FRAME_SIZE)) * 255).astype(np.uint8)
    targets = rng.normal(scale=1.5, size=(FRAMES_PER_SHARD - 1, 3)).astype(np.float32)
    mask = np.full((FRAME_SIZE, FRAME_SIZE), 255, dtype=np.uint8)
    np.savez_compressed(
        directory / f"{stem}-{index:05d}.npz",
        frames=frames,
        angular_velocity=targets,
        valid_mask=mask,
    )


@pytest.fixture
def shard_dir(tmp_path):
    for video_index, stem in enumerate(["flightA", "flightB", "flightC", "flightD"]):
        for shard_index in range(2):
            write_shard(tmp_path, stem, shard_index, seed=video_index * 10 + shard_index)
    return tmp_path


def test_network_output_shape():
    model = RotationNet()
    batch = torch.zeros(3, INPUT_CHANNELS, FRAME_SIZE, FRAME_SIZE)

    assert model(batch).shape == (3, 3)


def test_network_stays_small_enough_for_one_gpu():
    assert count_parameters(RotationNet()) < 20_000_000


def test_split_holds_out_whole_videos(shard_dir):
    split = split_shards_by_video(shard_dir, val_fraction=0.25)

    train_videos = {path.stem.rsplit("-", 1)[0] for path in split.train}
    val_videos = {path.stem.rsplit("-", 1)[0] for path in split.validation}

    assert not train_videos & val_videos
    assert len(val_videos) == 1
    assert len(split.validation) == 2  # both shards of the held-out video


def test_split_rejects_single_video(tmp_path):
    write_shard(tmp_path, "only", 0, seed=0)
    with pytest.raises(DataError):
        split_shards_by_video(tmp_path, val_fraction=0.5)


def test_split_rejects_empty_directory(tmp_path):
    with pytest.raises(DataError):
        split_shards_by_video(tmp_path, val_fraction=0.5)


def test_dataset_yields_normalised_pairs(shard_dir):
    shards = sorted(shard_dir.glob("*.npz"))
    dataset = ShardDataset(shards, augment=False)

    frames, target = next(iter(dataset))

    assert frames.shape == (INPUT_CHANNELS, FRAME_SIZE, FRAME_SIZE)
    assert frames.dtype == torch.float32
    assert -1.01 <= float(frames.min()) and float(frames.max()) <= 1.01
    assert target.shape == (3,)


def test_dataset_with_augmentation_is_still_well_formed(shard_dir):
    shards = sorted(shard_dir.glob("*.npz"))
    dataset = ShardDataset(shards, augment=True, seed=3)

    for index, (frames, target) in enumerate(dataset):
        assert torch.isfinite(frames).all()
        assert torch.isfinite(target).all()
        if index > 30:
            break


def test_dataset_covers_every_pair(shard_dir):
    shards = sorted(shard_dir.glob("*.npz"))
    dataset = ShardDataset(shards, augment=False)

    count = sum(1 for _ in dataset)

    assert count == len(shards) * (FRAMES_PER_SHARD - 1)


def test_amp_is_disabled_off_cuda():
    assert not amp_is_worthwhile(torch.device("cpu"))


def test_select_device_honours_request():
    assert select_device("cpu") == torch.device("cpu")


def test_training_loop_runs_and_saves_a_checkpoint(shard_dir, tmp_path):
    checkpoint = tmp_path / "model.pt"
    config = TrainConfig(
        data_dir=shard_dir,
        output=checkpoint,
        epochs=1,
        batch_size=4,
        val_fraction=0.25,
        workers=0,
        device="cpu",
    )

    train(config)

    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "state_dict" in payload
    assert payload["target_scale_rad_s"] > 0
