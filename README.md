# regyro

Reconstruct gyroscope data from video, for [Gyroflow](https://gyroflow.xyz) stabilization when gyro data is lost or missing.

## How it works

1. **Classical pipeline** — sparse optical flow between consecutive frames, undistorted through a Gyroflow lens profile, rotation-only fit with RANSAC → angular velocities → `.gcsv` file readable by Gyroflow.
2. **ML pipeline** (planned) — a compact CNN trained on video/gyro pairs (own FPV footage + public datasets) to outperform optical flow on hard cases: motion blur, low light, props in frame.

## Usage

```bash
uv run regyro estimate flight.mp4 --lens lens-profile.json
# → flight.gcsv, ready to load in Gyroflow
```

## Development

```bash
uv sync
uv run pytest
```
