import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from regyro.dataset.canonical import CANONICAL_FOV_DEG, CANONICAL_SIZE
from regyro.model.data import ShardDataset, split_shards_by_video
from regyro.model.network import TARGET_SCALE_RAD_S, RotationNet, count_parameters

logger = logging.getLogger(__name__)

# Pascal runs fp16 at a fraction of fp32 throughput, so mixed precision costs time there.
MIN_AMP_CAPABILITY = (7, 0)
LOG_EVERY_STEPS = 50


@dataclass
class TrainConfig:
    data_dir: Path
    output: Path
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    workers: int = 4
    device: str | None = None
    seed: int = 0


def select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_is_worthwhile(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device) >= MIN_AMP_CAPABILITY


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss, total_samples = 0.0, 0

    for step, (frames, targets) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device.type, enabled=scaler is not None):
                predictions = model(frames)
                loss = criterion(predictions, targets)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch = targets.shape[0]
        total_loss += loss.item() * batch
        total_samples += batch
        if training and step % LOG_EVERY_STEPS == 0:
            logger.info("step %d loss %.5f", step, loss.item())

    if total_samples == 0:
        raise RuntimeError("epoch produced no samples; check the shard directory")
    return total_loss / total_samples


def train(config: TrainConfig) -> Path:
    device = select_device(config.device)
    split = split_shards_by_video(config.data_dir, config.val_fraction, config.seed)
    logger.info(
        "device %s | %d train shards | %d val shards",
        device,
        len(split.train),
        len(split.validation),
    )

    train_set = ShardDataset(split.train, augment=True, seed=config.seed)
    val_set = ShardDataset(split.validation, augment=False, seed=config.seed)
    loader_args = {"batch_size": config.batch_size, "num_workers": config.workers}
    train_loader = DataLoader(train_set, **loader_args, drop_last=True)
    val_loader = DataLoader(val_set, **loader_args)

    model = RotationNet().to(device)
    logger.info("model has %d parameters", count_parameters(model))
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler(device.type) if amp_is_worthwhile(device) else None

    best_loss = float("inf")
    config.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        train_set.set_epoch(epoch)
        started = time.monotonic()
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss = _run_epoch(model, val_loader, criterion, device, None, None)
        schedule.step()

        logger.info(
            "epoch %d/%d train %.5f val %.5f (%.1fs)",
            epoch + 1,
            config.epochs,
            train_loss,
            val_loss,
            time.monotonic() - started,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(config.output, model, epoch, val_loss)
            logger.info("saved checkpoint to %s", config.output)

    return config.output


def save_checkpoint(path: Path, model: nn.Module, epoch: int, val_loss: float) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "target_scale_rad_s": TARGET_SCALE_RAD_S,
            "canonical_size": CANONICAL_SIZE,
            "canonical_fov_deg": CANONICAL_FOV_DEG,
        },
        path,
    )
