from dataclasses import dataclass

import numpy as np


@dataclass
class GyroEstimate:
    """Per-frame angular velocities in rad/s, expressed in the camera frame."""

    angular_velocities: np.ndarray
    sample_rate_hz: float
    failed_pairs: int = 0
