from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


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
        frames = np.asarray(self.data, dtype=np.uint8)
        if frames.ndim == 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        first, *rest = [Image.fromarray(frame) for frame in frames]
        first.save(
            path,
            save_all=True,
            append_images=rest,
            duration=max(int(1000 / max(self.fps, 1)), 1),
            loop=0,
        )
        return str(path)


@dataclass(frozen=True)
class Checkpoint(Artifact):
    score: float = float("-inf")


@dataclass(frozen=True)
class Metrics(Artifact):
    pass
