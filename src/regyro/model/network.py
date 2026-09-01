import torch
from torch import nn

# Targets are divided by this before the loss so outputs sit near unit scale.
# 5 rad/s (~290 deg/s) is a typical hard FPV rate.
TARGET_SCALE_RAD_S = 5.0

STAGE_CHANNELS = (64, 96, 128, 192, 256)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        needs_projection = stride != 1 or in_channels != out_channels
        self.shortcut: nn.Module = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if needs_projection
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class RotationNet(nn.Module):
    """Predicts camera angular velocity from a pair of canonically projected frames.

    The two frames enter as channels rather than through a siamese encoder: the
    network only needs their difference, and early fusion keeps it small enough
    to train on a single mid-range GPU.
    """

    def __init__(self, in_frames: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_frames, STAGE_CHANNELS[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(STAGE_CHANNELS[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        for index in range(1, len(STAGE_CHANNELS)):
            stages.append(ResidualBlock(STAGE_CHANNELS[index - 1], STAGE_CHANNELS[index], stride=2))
            stages.append(ResidualBlock(STAGE_CHANNELS[index], STAGE_CHANNELS[index], stride=1))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(STAGE_CHANNELS[-1], 3)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.stages(self.stem(frames))
        return self.head(self.pool(features).flatten(1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
