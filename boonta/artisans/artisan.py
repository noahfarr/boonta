from typing import Protocol, runtime_checkable

from boonta.utils import Artifact


@runtime_checkable
class Artisan(Protocol):
    name: str

    def craft(self, algorithm, environment, state, logs) -> Artifact: ...
