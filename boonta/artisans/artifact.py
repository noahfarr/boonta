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

    def frames(self) -> np.ndarray:
        frames = np.asarray(self.data, dtype=np.uint8)
        if frames.ndim == 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        return frames

    def encode(self, path) -> str:
        from pathlib import Path

        path = Path(path)
        if path.suffix.lower() == ".gif":
            return self.encode_gif(path)
        try:
            return self.encode_mp4(path)
        except Exception:
            return self.encode_gif(path.with_suffix(".gif"))

    def encode_mp4(self, path) -> str:
        import imageio.v2 as imageio

        imageio.mimwrite(
            path,
            self.frames(),
            format="FFMPEG",
            fps=max(self.fps, 1),
            codec="libx264",
            pixelformat="yuv420p",
        )
        return str(path)

    def encode_gif(self, path) -> str:
        from PIL import Image

        frames = self.frames()
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
