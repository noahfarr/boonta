from typing import Protocol, TypeVar

from boonta.utils import Array, Key, PyTree, Timestep, Transition

State = TypeVar("State")


class Algorithm(Protocol[State]):
    def init(self, key: Key, timestep: Timestep) -> tuple[State, PyTree]: ...

    def step(
        self,
        state: State,
        carry: PyTree,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[State, PyTree, Array, PyTree]: ...

    def update(self, state: State, key: Key, transitions: Transition) -> State: ...
