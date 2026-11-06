from typing import Protocol, runtime_checkable

from .artifact import Artifact


@runtime_checkable
class Artisan(Protocol):
    name: str

    def craft(self, key, podracer, podracer_state, logs) -> Artifact: ...
