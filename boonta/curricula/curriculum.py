from collections.abc import Callable
from typing import Protocol

from boonta.algorithms import Algorithm
from boonta.podracers.podracer import Lap, Pit
from boonta.utils.typing import Environment, PyTree


class Curriculum(Protocol):
    def __call__(
        self, algorithm: Algorithm, environment: Environment, **kwargs
    ) -> tuple[Algorithm, Environment, Pit, Lap]: ...


def within(state: PyTree, kind: type, change: Callable[[PyTree], PyTree]) -> PyTree:
    if isinstance(state, kind):
        return change(state)
    return state.replace(env_state=within(state.env_state, kind, change))
