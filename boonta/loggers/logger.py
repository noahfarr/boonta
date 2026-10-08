from typing import Protocol

import jax

from boonta.artisans import Artifact
from boonta.utils import PyTree


class Logger(Protocol):
    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None: ...
    def log_summary(self, data: PyTree, **kwargs) -> None: ...
    def log_artifact(self, artifact: Artifact, step: int, **kwargs) -> None: ...
    def finish(self) -> None: ...


class Silent:
    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        pass

    def log_summary(self, data: PyTree, **kwargs) -> None:
        pass

    def log_artifact(self, artifact: Artifact, step: int, **kwargs) -> None:
        pass

    def finish(self) -> None:
        pass


class Primary:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls) if jax.process_index() == 0 else Silent()


class MultiLogger:
    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        data, steps = jax.device_get((data, steps))
        for logger in self.loggers:
            logger.log(data, steps, **kwargs)

    def log_summary(self, data: PyTree, **kwargs) -> None:
        data = jax.device_get(data)
        for logger in self.loggers:
            logger.log_summary(data, **kwargs)

    def log_artifact(self, artifact: Artifact, step: int, **kwargs) -> None:
        for logger in self.loggers:
            logger.log_artifact(artifact, step, **kwargs)

    def finish(self) -> None:
        for logger in self.loggers:
            logger.finish()
