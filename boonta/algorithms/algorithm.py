from typing import Protocol, TypeVar

from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep, Transition

State = TypeVar("State")


@struct.dataclass(frozen=True)
class Carries:
    first: PyTree = None
    last: PyTree = None


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

    def update(
        self, state: State, key: Key, transitions: Transition, carries: Carries
    ) -> State: ...
