import abc
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from regyro.errors import RegyroError


class FrameSourceError(RegyroError):
    pass


@dataclass(frozen=True)
class Frame:
    timestamp_s: float
    gray: np.ndarray


class FrameSource(abc.ABC):
    """A timestamped stream of grayscale frames.

    Public datasets ship as image folders with irregular timestamps while cameras
    ship as containers with a nominal frame rate; the builder only needs the stream.
    """

    name: str

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError


class VideoFrameSource(FrameSource):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.name = self.path.stem
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise FrameSourceError(f"cannot open video {self.path}")
        try:
            self.frame_rate = capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
        if self.frame_rate <= 0:
            raise FrameSourceError(f"invalid frame rate in {self.path}")

    def __iter__(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise FrameSourceError(f"cannot open video {self.path}")
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield Frame(index / self.frame_rate, gray)
                index += 1
        finally:
            capture.release()


class ImageSequenceFrameSource(FrameSource):
    """Frames stored as individual image files with explicit timestamps."""

    def __init__(self, name: str, paths: list[Path], timestamps_s: np.ndarray) -> None:
        if len(paths) != len(timestamps_s):
            raise FrameSourceError(
                f"{len(paths)} images but {len(timestamps_s)} timestamps for {name}"
            )
        if len(paths) < 2:
            raise FrameSourceError(f"{name}: need at least two frames, got {len(paths)}")
        self.name = name
        self.paths = paths
        self.timestamps_s = timestamps_s

    def __iter__(self) -> Iterator[Frame]:
        for path, timestamp in zip(self.paths, self.timestamps_s):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FrameSourceError(f"cannot read image {path}")
            yield Frame(float(timestamp), gray)
