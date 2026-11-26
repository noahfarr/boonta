from typing import Protocol

from boonta.algorithms import Algorithm
from boonta.podracers.podracer import Lap, Pit
from boonta.utils.typing import Environment


class Curriculum(Protocol):
    def __call__(
        self, algorithm: Algorithm, environment: Environment, **kwargs
    ) -> tuple[Algorithm, Environment, Pit, Lap]: ...
