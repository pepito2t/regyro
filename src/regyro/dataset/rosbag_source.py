"""Reader for ROS bags, the format UZH-FPV and Blackbird distribute.

Uses the pure-Python `rosbags` package, so no ROS installation is needed. Images are
streamed rather than collected, because a racing sequence does not fit in memory.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from regyro.dataset.frame_source import Frame, FrameSource
from regyro.dataset.telemetry import GyroSamples
from regyro.errors import RegyroError

logger = logging.getLogger(__name__)

IMU_TYPES = ("sensor_msgs/msg/Imu",)
IMAGE_TYPES = ("sensor_msgs/msg/Image",)


class RosbagError(RegyroError):
    pass


def _reader(path: Path):
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RosbagError("reading rosbags needs the rosbags package: `uv sync --extra datasets`") from exc
    if not path.exists():
        raise RosbagError(f"no such bag: {path}")
    return AnyReader([path])


def _stamp_seconds(message) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _pick_topic(reader, message_types: tuple[str, ...], requested: str | None, kind: str) -> str:
    available = sorted({c.topic for c in reader.connections if c.msgtype in message_types})
    if requested:
        if requested not in available:
            raise RosbagError(f"{kind} topic {requested!r} not in bag; available: {available}")
        return requested
    if not available:
        raise RosbagError(f"bag has no {kind} topic of type {message_types[0]}")
    if len(available) > 1:
        raise RosbagError(f"bag has several {kind} topics {available}; pick one explicitly")
    return available[0]


def _decode_image(message) -> np.ndarray:
    encoding = message.encoding.lower()
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if encoding in ("mono8", "8uc1"):
        return raw.reshape(message.height, message.step)[:, : message.width]
    if encoding in ("bgr8", "rgb8", "8uc3"):
        colour = raw.reshape(message.height, message.step // 3, 3)[:, : message.width]
        conversion = cv2.COLOR_RGB2GRAY if encoding == "rgb8" else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(colour, conversion)
    raise RosbagError(f"unsupported image encoding {message.encoding!r}")


def read_gyro(bag_path: Path, imu_topic: str | None = None) -> GyroSamples:
    timestamps, rates = [], []
    with _reader(bag_path) as reader:
        topic = _pick_topic(reader, IMU_TYPES, imu_topic, "IMU")
        connections = [c for c in reader.connections if c.topic == topic]
        for connection, _, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            timestamps.append(_stamp_seconds(message))
            rates.append(
                [
                    message.angular_velocity.x,
                    message.angular_velocity.y,
                    message.angular_velocity.z,
                ]
            )

    if len(timestamps) < 2:
        raise RosbagError(f"{bag_path.name}: fewer than two IMU samples on the chosen topic")
    order = np.argsort(timestamps)
    return GyroSamples(np.asarray(timestamps)[order], np.asarray(rates)[order])


class RosbagFrameSource(FrameSource):
    def __init__(self, bag_path: Path, image_topic: str | None = None) -> None:
        self.bag_path = Path(bag_path)
        self.name = self.bag_path.stem
        with _reader(self.bag_path) as reader:
            self.image_topic = _pick_topic(reader, IMAGE_TYPES, image_topic, "image")

    def __iter__(self) -> Iterator[Frame]:
        with _reader(self.bag_path) as reader:
            connections = [c for c in reader.connections if c.topic == self.image_topic]
            for connection, _, rawdata in reader.messages(connections=connections):
                message = reader.deserialize(rawdata, connection.msgtype)
                yield Frame(_stamp_seconds(message), _decode_image(message))


@dataclass
class BagSequence:
    name: str
    source: RosbagFrameSource
    gyro: GyroSamples


def load_bag(
    bag_path: Path,
    image_topic: str | None = None,
    imu_topic: str | None = None,
) -> BagSequence:
    source = RosbagFrameSource(bag_path, image_topic)
    gyro = read_gyro(bag_path, imu_topic)
    logger.info(
        "%s: images on %s, %d gyro samples", source.name, source.image_topic, len(gyro.timestamps)
    )
    return BagSequence(name=source.name, source=source, gyro=gyro)
