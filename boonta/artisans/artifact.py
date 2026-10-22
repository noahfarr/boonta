from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Artifact:
    name: str
    data: Any


@dataclass(frozen=True)
class Text(Artifact):
    pass


@dataclass(frozen=True)
class Video(Artifact):
    fps: int = 16

    def encode(self, path) -> str:
        from PIL import Image as PILImage

        frames = np.asarray(self.data, dtype=np.uint8)
        if frames.ndim == 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        first, *rest = [PILImage.fromarray(frame) for frame in frames]
        first.save(
            path,
            save_all=True,
            append_images=rest,
            duration=max(int(1000 / max(self.fps, 1)), 1),
            loop=0,
        )
        return str(path)


@dataclass(frozen=True)
class Image(Artifact):

    def encode(self, path) -> str:
        from PIL import Image as PILImage

        pixels = np.asarray(self.data, dtype=np.uint8)
        PILImage.fromarray(pixels).save(path)
        return str(path)


@dataclass(frozen=True)
class Checkpoint(Artifact):
    score: float = float("-inf")


@dataclass(frozen=True)
class Metrics(Artifact):
    pass
