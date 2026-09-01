import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from regyro.classical.video_pipeline import VideoError, read_frame_rate
from regyro.dataset.canonical import CANONICAL_FOV_DEG, CANONICAL_SIZE, CanonicalProjector
from regyro.lens_profile import LensProfile
from regyro.model.network import TARGET_SCALE_RAD_S, RotationNet
from regyro.model.train import select_device
from regyro.result import GyroEstimate
from regyro.errors import RegyroError

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


class InferenceError(RegyroError):
    pass


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[RotationNet, dict]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (OSError, RuntimeError) as exc:
        raise InferenceError(f"cannot load checkpoint {checkpoint_path}: {exc}") from exc

    size = checkpoint.get("canonical_size", CANONICAL_SIZE)
    fov = checkpoint.get("canonical_fov_deg", CANONICAL_FOV_DEG)
    if (size, fov) != (CANONICAL_SIZE, CANONICAL_FOV_DEG):
        raise InferenceError(
            f"checkpoint was trained on {size}px/{fov}deg projection but this build uses "
            f"{CANONICAL_SIZE}px/{CANONICAL_FOV_DEG}deg"
        )

    model = RotationNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _predict_batch(model: RotationNet, pairs: list[np.ndarray], device: torch.device) -> np.ndarray:
    stacked = np.stack(pairs).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(stacked).to(device)
    with torch.no_grad():
        outputs = model(tensor)
    return outputs.cpu().numpy() * TARGET_SCALE_RAD_S


def estimate_gyro_ml(
    video_path: Path | str,
    profile: LensProfile,
    checkpoint_path: Path,
    device_name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> GyroEstimate:
    """Estimate per-frame angular velocities with the trained network."""
    device = select_device(device_name)
    model, _ = load_model(checkpoint_path, device)
    fps = read_frame_rate(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoError(f"cannot open video {video_path}")

    projector: CanonicalProjector | None = None
    previous: np.ndarray | None = None
    pending: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if projector is None:
                projector = CanonicalProjector.build(profile, gray.shape[1], gray.shape[0])
            current = projector.remap(gray)

            if previous is not None:
                pending.append(np.stack([previous, current]))
                if len(pending) == batch_size:
                    predictions.append(_predict_batch(model, pending, device))
                    pending = []
            previous = current

        if pending:
            predictions.append(_predict_batch(model, pending, device))
    finally:
        capture.release()

    if not predictions:
        raise VideoError(f"video {video_path} has fewer than 2 frames")

    return GyroEstimate(
        angular_velocities=np.concatenate(predictions, axis=0),
        sample_rate_hz=fps,
    )
