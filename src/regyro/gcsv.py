from pathlib import Path

import numpy as np

GCSV_VERSION = "1.3"


def write_gcsv(
    path: Path | str,
    angular_velocities: np.ndarray,
    sample_rate_hz: float,
    orientation: str = "xyz",
    video_filename: str | None = None,
) -> None:
    """Write angular velocities (N, 3) in rad/s to a Gyroflow .gcsv file."""
    if angular_velocities.ndim != 2 or angular_velocities.shape[1] != 3:
        raise ValueError(f"expected (N, 3) angular velocities, got {angular_velocities.shape}")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample rate must be positive, got {sample_rate_hz}")

    header = [
        "GYROFLOW IMU LOG",
        f"version,{GCSV_VERSION}",
        "id,regyro",
        f"orientation,{orientation}",
        f"tscale,{1.0 / sample_rate_hz:.9f}",
        "gscale,1.0",
        "ascale,1.0",
    ]
    if video_filename:
        header.append(f"videofilename,{video_filename}")
    header.append("t,gx,gy,gz")

    rows = (
        f"{index},{gx:.6f},{gy:.6f},{gz:.6f}"
        for index, (gx, gy, gz) in enumerate(angular_velocities)
    )
    Path(path).write_text("\n".join([*header, *rows]) + "\n")
