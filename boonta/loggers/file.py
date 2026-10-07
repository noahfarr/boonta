from collections import defaultdict
from pathlib import Path

import numpy as np

from boonta.utils import PyTree, Video


class FileLogger:
    def __init__(self, directory: str = ".", filename: str = "metrics.npz", **kwargs):
        self.directory = Path(directory)
        self.filename = filename
        self.history = defaultdict(list)
        self.steps = []

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        index = len(self.steps)
        self.steps.append(int(np.asarray(steps).max()))
        for key, value in data.items():
            self.history[key].append((index, np.array([np.mean(row) for row in value])))
        self.save()

    def log_artifact(self, artifact, step: int, **kwargs) -> None:
        if isinstance(artifact, Video):
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{artifact.name}-{step}.gif"
            artifact.encode(path)

    def log_summary(self, data: PyTree, **kwargs) -> None:
        pass

    def align(self, entries):
        _, first = entries[0]
        series = np.full((len(self.steps), *first.shape), np.nan)
        for index, value in entries:
            series[index] = value
        return series

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        arrays = {key: self.align(entries) for key, entries in self.history.items()}
        path = self.directory / self.filename
        temporary = path.with_suffix(".npz.tmp")
        with open(temporary, "wb") as handle:
            np.savez(handle, steps=np.asarray(self.steps), **arrays)
        temporary.replace(path)

    def finish(self) -> None:
        self.save()
