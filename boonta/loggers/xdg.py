import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax

from boonta.artisans import Video
from boonta.utils import PyTree


class XdgLogger:
    def __init__(self, directory: str = ".", **kwargs):
        self.directory = Path(directory)

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        pass

    def log_artifact(self, artifact, step: int, **kwargs) -> None:
        if jax.process_index() != 0 or not isinstance(artifact, Video):
            return
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        if not shutil.which(opener):
            return
        if sys.platform != "darwin" and not os.environ.get("DISPLAY"):
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = artifact.encode(self.directory / f"{artifact.name}-{step}.gif")
        subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def log_summary(self, data: PyTree, **kwargs) -> None:
        pass

    def finish(self) -> None:
        pass
