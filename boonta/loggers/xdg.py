import os
import shutil
import subprocess
from pathlib import Path

import jax

from boonta.artisans import Image, Video
from boonta.utils import PyTree


class XdgLogger:
    def __init__(self, directory: str = ".", **kwargs):
        self.directory = Path(directory)

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        pass

    def log_artifact(self, artifact, step: int, **kwargs) -> None:
        if jax.process_index() != 0 or not isinstance(artifact, (Image, Video)):
            return
        if not (os.environ.get("DISPLAY") and shutil.which("xdg-open")):
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        suffix = "png" if isinstance(artifact, Image) else "gif"
        path = artifact.encode(self.directory / f"{artifact.name}-{step}.{suffix}")
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def log_summary(self, data: PyTree, **kwargs) -> None:
        pass

    def finish(self) -> None:
        pass
