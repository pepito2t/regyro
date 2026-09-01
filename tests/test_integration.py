"""End-to-end chain: build shards from video, train on them, infer back to a .gcsv.

Real footage is needed to judge accuracy, but this pins the plumbing: frame/target
alignment, shard layout, checkpoint round-trip and output length.
"""

import json

import cv2
import numpy as np
import pytest
import torch

from regyro.dataset.build import build_from_video
from regyro.dataset.canonical import CANONICAL_SIZE
from regyro.dataset.sync import SyncCalibration
from regyro.dataset.telemetry import AxisMapping, GyroSamples
from regyro.gcsv import write_gcsv
from regyro.lens_profile import LensProfile
from regyro.model.infer import estimate_gyro_ml
from regyro.model.train import TrainConfig, train

FPS = 30.0
FRAME_COUNT = 20
VIDEO_SIZE = (320, 240)
CONSTANT_RATE = np.array([0.2, -0.3, 0.5])


@pytest.fixture
def lens(tmp_path) -> LensProfile:
    data = {
        "calib_dimension": {"w": VIDEO_SIZE[0], "h": VIDEO_SIZE[1]},
        "fisheye_params": {
            "camera_matrix": [[150.0, 0.0, 160.0], [0.0, 150.0, 120.0], [0.0, 0.0, 1.0]],
            "distortion_coeffs": [0.01, 0.0, 0.0, 0.0],
        },
    }
    path = tmp_path / "lens.json"
    path.write_text(json.dumps(data))
    return LensProfile.from_json(path)


def write_video(path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    texture = (rng.random((VIDEO_SIZE[1] + 60, VIDEO_SIZE[0] + 60)) * 255).astype(np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, VIDEO_SIZE, False)
    for index in range(FRAME_COUNT):
        x, y = index * 2, index
        writer.write(texture[y : y + VIDEO_SIZE[1], x : x + VIDEO_SIZE[0]])
    writer.release()


@pytest.fixture
def constant_gyro(monkeypatch):
    timestamps = np.linspace(0.0, FRAME_COUNT / FPS + 1.0, 2000)
    samples = GyroSamples(timestamps, np.tile(CONSTANT_RATE, (len(timestamps), 1)))
    monkeypatch.setattr("regyro.dataset.build.load_gyro", lambda _path: samples)
    return samples


def test_build_produces_aligned_frames_and_targets(tmp_path, lens, constant_gyro):
    video = tmp_path / "flight.mp4"
    write_video(video, seed=0)
    calibration = SyncCalibration(0.0, AxisMapping.identity(), rmse_rad_s=0.0)

    stats = build_from_video(video, lens, tmp_path / "shards", calibration)

    assert stats.shards == 1
    assert stats.frame_pairs == FRAME_COUNT - 1

    shard = next((tmp_path / "shards").glob("*.npz"))
    with np.load(shard) as payload:
        frames, targets = payload["frames"], payload["angular_velocity"]
        mask = payload["valid_mask"]

    assert mask.shape == (CANONICAL_SIZE, CANONICAL_SIZE)
    assert mask.dtype == np.uint8

    assert frames.shape == (FRAME_COUNT, CANONICAL_SIZE, CANONICAL_SIZE)
    assert targets.shape == (FRAME_COUNT - 1, 3)
    # A constant gyro rate must come back unchanged through the integration.
    np.testing.assert_allclose(targets, np.tile(CONSTANT_RATE, (len(targets), 1)), atol=1e-4)


def test_build_refuses_badly_synchronised_video(tmp_path, lens, constant_gyro):
    video = tmp_path / "flight.mp4"
    write_video(video, seed=0)
    unusable = SyncCalibration(0.0, AxisMapping.identity(), rmse_rad_s=99.0)

    with pytest.raises(Exception, match="sync rmse"):
        build_from_video(video, lens, tmp_path / "shards", unusable)


def test_train_then_infer_round_trip(tmp_path, lens, constant_gyro):
    shard_dir = tmp_path / "shards"
    calibration = SyncCalibration(0.0, AxisMapping.identity(), rmse_rad_s=0.0)
    for index, name in enumerate(["flightA", "flightB"]):
        video = tmp_path / f"{name}.mp4"
        write_video(video, seed=index)
        build_from_video(video, lens, shard_dir, calibration)

    checkpoint = tmp_path / "model.pt"
    train(
        TrainConfig(
            data_dir=shard_dir,
            output=checkpoint,
            epochs=1,
            batch_size=4,
            val_fraction=0.5,
            workers=0,
            device="cpu",
        )
    )
    assert checkpoint.exists()

    video = tmp_path / "flightA.mp4"
    estimate = estimate_gyro_ml(video, lens, checkpoint, device_name="cpu", batch_size=8)

    assert estimate.angular_velocities.shape == (FRAME_COUNT - 1, 3)
    assert estimate.sample_rate_hz == pytest.approx(FPS)
    assert np.isfinite(estimate.angular_velocities).all()

    output = tmp_path / "out.gcsv"
    write_gcsv(output, estimate.angular_velocities, estimate.sample_rate_hz)
    assert output.read_text().startswith("GYROFLOW IMU LOG")


def test_checkpoint_rejects_mismatched_projection(tmp_path, lens):
    checkpoint = tmp_path / "stale.pt"
    torch.save(
        {"state_dict": {}, "canonical_size": 128, "canonical_fov_deg": 90.0},
        checkpoint,
    )
    video = tmp_path / "flight.mp4"
    write_video(video, seed=0)

    with pytest.raises(Exception, match="canonical|projection"):
        estimate_gyro_ml(video, lens, checkpoint, device_name="cpu")
