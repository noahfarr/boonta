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

    @property
    def pit_state(self) -> PyTree: ...

    @property
    def lap_state(self) -> PyTree: ...

    def replace(self, **changes) -> Self: ...


class Lap:
    def init(self, key: Key, state: PodracerState) -> PyTree:
        return None

    def __call__(self, key: Key, state: PodracerState) -> PodracerState:
        return state


class Pit(Lap):
    def update(
        self, state: PyTree, key: Key, transitions: Transition, algorithm_state: PyTree
    ) -> PyTree:
        return state


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
