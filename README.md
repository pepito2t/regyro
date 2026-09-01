# regyro

Reconstruct gyroscope data from video, for [Gyroflow](https://gyroflow.xyz) stabilization when gyro data is lost or missing.

## How it works

Two interchangeable estimators, both writing the same `.gcsv` Gyroflow reads:

1. **Classical** — sparse optical flow between consecutive frames, undistorted through a Gyroflow lens profile, rotation-only fit with RANSAC to reject parallax, props and moving objects.
2. **Learned** — a compact CNN trained on video/gyro pairs. Any clip that still has valid gyro is a free training example, so the dataset is self-supervised.

## Usage

```bash
# Reconstruct gyro from a clip that lost it
uv run regyro estimate flight.mp4 --lens lens-profile.json
uv run regyro estimate flight.mp4 --lens lens-profile.json --method ml --checkpoint model.pt

# Inspect how a clip's IMU lines up with its images
uv run regyro calibrate flight.mp4 --lens lens-profile.json

# Build training shards from clips that still have gyro
uv run regyro build-dataset rushes/*.mp4 --lens lens-profile.json -o data/shards

# Add public research datasets to the same shard directory
uv run regyro import-dataset euroc/MH_01/ --format asl -o data/shards
uv run regyro import-dataset uzh/*.bag --format rosbag --calibration camchain.yaml -o data/shards

# Train
uv run regyro train --data data/shards -o model.pt --epochs 30

# Measure either method against the real embedded gyro
uv run python scripts/evaluate.py flight.mp4 --lens lens-profile.json
```

## Design notes

**Canonical projection.** Every frame is remapped onto a fixed 140° equidistant fisheye before it reaches the network. Without this, the same pixel motion would mean different rotations depending on the camera's focal length and distortion, and GoPro, DJI and public datasets could not share a model. A validity mask travels with each shard as a third input channel, so the network can tell a narrow lens's empty periphery from genuinely dark image content.

**Public datasets.** Two importers cover what research datasets actually ship: the ASL/EuRoC directory layout (EuRoC MAV, TUM-VI) and ROS bags (UZH-FPV, Blackbird), the latter read through the pure-Python `rosbags` package so no ROS install is needed. Both rotate the gyro into the camera frame using the dataset's own extrinsics — an arbitrary rotation, not the axis swap that action cameras happen to use.

**Synchronisation.** The IMU clock and the image clock disagree. `calibrate` cross-correlates motion magnitude from the classical estimator against the embedded gyro to recover the offset, then brute-forces the axis permutation and signs. Clips whose residual stays too high are skipped rather than turned into mislabelled training data.

**Validation split by video, not by pair.** Consecutive frames are near-duplicates; splitting at pair level would leak validation into training and report a meaningless score.

**Roll augmentation.** Under the canonical projection a camera roll is exactly an image rotation, so rolling both frames and rotating the target is a geometrically valid way to multiply the fast-roll examples that FPV footage is short on. The sign conventions are pinned by tests in `tests/test_augment.py`.

## Hardware note

The training box is a GTX 1080 Ti (Pascal, sm_61). PyTorch 2.8 dropped Pascal kernels, so `torch` is pinned below 2.8. Mixed precision is deliberately disabled on Pascal, where fp16 runs slower than fp32.

## Development

```bash
uv sync --extra ml --extra datasets
uv run pytest
```

## Status

The classical pipeline works end to end. The learned path is implemented and its plumbing is tested, but it is untrained: it needs real footage before it means anything. The next step is running `scripts/evaluate.py` on clips with valid gyro to see whether the classical baseline is already good enough.
