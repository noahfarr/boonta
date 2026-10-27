import shutil
import subprocess
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

    @staticmethod
    def suffix() -> str:
        return "mp4" if shutil.which("ffmpeg") else "gif"

    def encode(self, path) -> str:
        from PIL import Image as PILImage

        frames = np.asarray(self.data, dtype=np.uint8)
        if frames.ndim == 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        if str(path).endswith((".mp4", ".webm")):
            return self.encode_ffmpeg(frames, path)
        first, *rest = [PILImage.fromarray(frame) for frame in frames]
        first.save(
            path,
            save_all=True,
            append_images=rest,
            duration=max(int(1000 / max(self.fps, 1)), 1),
            loop=0,
        )
        return str(path)

    def encode_ffmpeg(self, frames, path) -> str:
        _, height, width, _ = frames.shape
        codec = ["-c:v", "libvpx-vp9"] if str(path).endswith(".webm") else ["-c:v", "libx264"]
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", str(max(self.fps, 1)), "-i", "-",
                # yuv420p needs even sides
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                *codec, "-crf", "12", "-pix_fmt", "yuv420p", str(path),
            ],
            input=np.ascontiguousarray(frames).tobytes(),
            check=True,
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
