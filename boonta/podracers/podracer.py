from collections.abc import Callable
from typing import Protocol, Self, TypeVar

from boonta.utils import Key, PyTree, Timestep, Transition
from boonta.utils.typing import EnvState

State = TypeVar("State")


class PodracerState(Protocol):
    @property
    def timestep(self) -> Timestep: ...

    @property
    def environment_state(self) -> EnvState: ...

    @property
    def algorithm_state(self) -> PyTree: ...

    def replace(self, **changes) -> Self: ...


Boarded = TypeVar("Boarded", bound=PodracerState)

Pit = Callable[[Boarded, Transition | None], Boarded]
Lap = Callable[[Boarded], Boarded]


class Podracer(Protocol[State]):

    @property
    def batch_size(self) -> int: ...

    def init(self, key: Key) -> State: ...

    def train(
        self, state: State, key: Key, num_updates: int
    ) -> tuple[State, PyTree]: ...

    def evaluate(
        self, state: State, key: Key, num_steps: int
    ) -> tuple[State, PyTree]: ...

    def close(self, state: State) -> State: ...
